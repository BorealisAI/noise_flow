#!/usr/bin/env python
import logging
import os
import queue
import socket
import sys
import time
from datetime import datetime
from os import path
from threading import Thread

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from borealisflows.noise_flow_model import NoiseFlow
from borealisflows.utils import ResultLogger
from borealisflows.utils import get_its
from borealisflows.utils import hps_logger
from mylogger import add_logging_level
from sidd.ArgParser import arg_parser
from sidd.Initialization import initialize_data_stats_queues_baselines_histograms
from sidd.data_loader import check_download_sidd
from sidd.sidd_utils import sidd_filenames_que_inst, restore_last_model, \
    divide_parts, calc_train_test_stats, print_train_test_stats, sample_sidd_tf, \
    calc_kldiv_mb, kl_div_3_data, restore_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train_multithread(sess, num_epoch, tr_batch_que,
                      loss, sd_z, summ, writer, train_op,
                      x, y, nlf0, nlf1, iso, cam, lr, is_training,
                      _lr, n_processed_que, train_epoch_loss_que, sd_z_que,
                      train_its, nthr=8, requeue=False):
    divs = divide_parts(train_its, nthr)
    threads = []
    for thr_id in range(nthr):
        threads.append(Thread(target=train_thread,
                              args=(num_epoch, thr_id, divs[thr_id], sess, tr_batch_que,
                                    loss, sd_z, summ, writer, train_op,
                                    x, y, nlf0, nlf1, iso, cam, lr, is_training,
                                    _lr, n_processed_que, train_epoch_loss_que, sd_z_que, requeue)
                              )
                       )
        threads[thr_id].start()
    for thr_id in range(nthr):
        threads[thr_id].join()


def train_thread(num_epoch, thr_id, niter, sess, tr_batch_que,
                 loss, sd_z, summ, writer, train_op,
                 x, y, nlf0, nlf1, iso, cam, lr, is_training,
                 _lr, n_processed_que, train_epoch_loss_que, sd_z_que, requeue=False):
    for k in range(niter):
        tr_mb_dict = tr_batch_que.get()  # blocking
        _x = tr_mb_dict['_x']
        _y = tr_mb_dict['_y']
        _iso = tr_mb_dict['iso']
        _cam = tr_mb_dict['cam']

        feed_dict = {x: _x, y: _y, iso: _iso, cam: _cam, lr: _lr, is_training: True}

        if 'nlf0' in tr_mb_dict.keys():
            _nlf0 = tr_mb_dict['nlf0']
            _nlf1 = tr_mb_dict['nlf1']
            feed_dict.update({
                nlf0: _nlf0,
                nlf1: _nlf1
            })
        if hps.sidd_cond == 'condSDN':
            train_loss, sd_z_val, s = sess.run(
                [loss, sd_z, summ], feed_dict=feed_dict)
        else:
            _, train_loss, sd_z_val, s = sess.run(
                [train_op, loss, sd_z, summ], feed_dict=feed_dict)
        writer.add_summary(s, k  + (niter * (num_epoch - 1)))
        if requeue:
            tr_batch_que.put(tr_mb_dict)
        sd_z_que.put(sd_z_val)

        n_processed_que.put(hps.n_batch_train)
        train_epoch_loss_que.put(train_loss)


def test_multithread(sess, ts_batch_que,
                     loss, sd_z,
                     x, y, nlf0, nlf1, iso, cam, is_training,
                     test_epoch_loss_que, sd_z_que,
                     test_its, nthr=8, requeue=False):
    divs = divide_parts(test_its, nthr)
    threads = []
    for thr_id in range(nthr):
        threads.append(Thread(target=test_thread,
                              args=(thr_id, divs[thr_id], sess, ts_batch_que,
                                    loss, sd_z, x, y, nlf0, nlf1, iso, cam, is_training,
                                    test_epoch_loss_que, sd_z_que, requeue)
                              )
                       )
        threads[thr_id].start()
    for thr_id in range(nthr):
        threads[thr_id].join()


def test_thread(thr_id, niter, sess, ts_batch_que,
                loss, sd_z, x, y, nlf0, nlf1, iso, cam, is_training,
                test_epoch_loss_que, sd_z_que,
                requeue=False):
    for k in range(niter):
        ts_mb_dict = ts_batch_que.get()  # blocking
        _x = ts_mb_dict['_x']
        _y = ts_mb_dict['_y']
        _iso = ts_mb_dict['iso']
        _cam = ts_mb_dict['cam']

        feed_dict = {x: _x, y: _y, iso: _iso, cam: _cam, is_training: False}

        if 'nlf0' in ts_mb_dict.keys():
            _nlf0 = ts_mb_dict['nlf0']
            _nlf1 = ts_mb_dict['nlf1']
            feed_dict.update({
                nlf0: _nlf0,
                nlf1: _nlf1
            })

        test_loss, sd_z_val = sess.run([loss, sd_z], feed_dict=feed_dict)
        if requeue:
            ts_batch_que.put(ts_mb_dict)
        test_epoch_loss_que.put(test_loss)
        sd_z_que.put(sd_z_val)


def sample_multithread(sess, ts_batch_que,
                       loss, sd_z,
                       x, x_sample, y, nlf0, nlf1, iso, cam, is_training,
                       sample_epoch_loss_que, sd_z_que, kldiv_que,
                       test_its, nthr=8, requeue=False, sc_sd=1, epoch=0):
    divs = divide_parts(test_its, nthr)
    threads = []
    for thr_id in range(nthr):
        threads.append(Thread(target=sample_thread,
                              args=(thr_id, divs[thr_id], sess, ts_batch_que,
                                    loss, sd_z, x, x_sample, y, nlf0, nlf1, iso, cam, is_training,
                                    sample_epoch_loss_que, sd_z_que, kldiv_que, requeue, sc_sd, epoch)
                              )
                       )
        threads[thr_id].start()
    for thr_id in range(nthr):
        threads[thr_id].join()


def sample_thread(thr_id, niter, sess, ts_batch_que,
                  loss, sd_z, x, x_sample, y, nlf0, nlf1, iso, cam, is_training,
                  sample_epoch_loss_que, sd_z_que, kldiv_que, requeue=False, sc_sd=1, epoch=0):
    is_fix = True  # to fix the camera and ISO
    iso_vals = [100, 400, 800, 1600, 3200]
    iso_fix = [100]
    cam_fix = [['IP', 'GP', 'S6', 'N6', 'G4'].index('S6')]
    nlf_s6 = [[0.000479, 0.000002], [0.001774, 0.000002], [0.003696, 0.000002], [0.008211, 0.000002],
              [0.019930, 0.000002]]
    # for S6, for ISO 100, 400, 800, 1600, 3200

    for k in range(niter):
        ts_mb_dict = ts_batch_que.get()  # blocking

        _x = ts_mb_dict['_x']
        _y = ts_mb_dict['_y']
        if is_fix:
            _iso = iso_fix
            _cam = cam_fix
            _nlf0 = [nlf_s6[iso_vals.index(iso_fix[0])][0]]
            _nlf1 = [nlf_s6[iso_vals.index(iso_fix[0])][0]]
        else:
            _iso = ts_mb_dict['iso']
            _cam = ts_mb_dict['cam']
            _nlf0 = ts_mb_dict['nlf0']
            _nlf1 = ts_mb_dict['nlf1']

        # sample (forward)
        x_sample_val = sess.run(x_sample, feed_dict={y: _y, nlf0: _nlf0, nlf1: _nlf1,
                                                     iso: _iso, cam: _cam, is_training: False})

        # (optional) compute KL divergence between _x and x_sample_val
        kldiv3 = kl_div_3_data(_x, x_sample_val)  # slow

        # compute NLL (inverse)
        sample_loss, sd_z_val = sess.run([loss, sd_z], feed_dict={x: x_sample_val, y: _y, nlf0: _nlf0, nlf1: _nlf1,
                                                                  iso: _iso, cam: _cam, is_training: False})
        # marginal KL divergence
        vis_mbs_dir = os.path.join(hps.logdir, 'samples_epoch_%04d' % epoch, 'samples_%.1f' % hps.temp)
        kldiv3 = calc_kldiv_mb(ts_mb_dict, x_sample_val, vis_mbs_dir, sc_sd)

        if requeue:
            ts_batch_que.put(ts_mb_dict)
        sample_epoch_loss_que.put(sample_loss)
        sd_z_que.put(sd_z_val)
        kldiv_que.put(kldiv3)


def get_optimizer(hps, lr, loss_val):
    train_op = None
    if hps.sidd_cond != 'condSDN':
        if hps.optim == 'adam':
            train_op = tf.train.AdamOptimizer(learning_rate=lr,
                                              beta1=0.9,
                                              beta2=0.999,
                                              epsilon=1e-08).minimize(loss_val)
        elif hps.optim == 'sgd':
            train_op = tf.train.MomentumOptimizer(lr, 0.9).minimize(loss_val)
        tf.add_to_collection('train_op', train_op)
    return train_op


def init_params(hps1):
    npcam = 3
    if hps1.arch.__contains__('sdn5'):
        npcam = 3
    elif hps1.arch.__contains__('sdn6'):
        npcam = 1
    c_i = 1.0
    beta1_i = -5.0 / c_i
    beta2_i = 0.0
    gain_params_i = np.ndarray([5])
    gain_params_i[:] = -5.0 / c_i
    cam_params_i = np.ndarray([npcam, 5])
    cam_params_i[:, :] = 1.0
    hps1.param_inits = (c_i, beta1_i, beta2_i, gain_params_i, cam_params_i)


def main(hps):
    # Download SIDD_Medium_Raw?
    check_download_sidd()

    total_time = time.time()
    host = socket.gethostname()
    tf.set_random_seed(hps.seed)
    np.random.seed(hps.seed)

    # set up a custom logger
    add_logging_level('TRACE', 100)
    logging.getLogger(__name__).setLevel("TRACE")
    logging.basicConfig(level=logging.TRACE)

    hps.n_bins = 2. ** hps.n_bits_x

    logging.trace('SIDD path = %s' % hps.sidd_path)
    logging.trace('Num GPUs Available: %s' % len(tf.config.experimental.list_physical_devices('GPU')))
    # prepare data file names
    tr_fns, hps.n_tr_inst = sidd_filenames_que_inst(hps.sidd_path, 'train', hps.start_tr_im_idx, hps.end_tr_im_idx,
                                                    hps.camera, hps.iso)
    logging.trace('# training scene instances (cam = %s, iso = %s) = %d' %
                  (str(hps.camera), str(hps.iso), hps.n_tr_inst))
    ts_fns, hps.n_ts_inst = sidd_filenames_que_inst(hps.sidd_path, 'test', hps.start_ts_im_idx, hps.end_ts_im_idx,
                                                    hps.camera, hps.iso)
    logging.trace('# testing scene instances (cam = %s, iso = %s) = %d' %
                  (str(hps.camera), str(hps.iso), hps.n_ts_inst))

    # training/testing data stats
    calc_train_test_stats(hps)

    # output log dir
    logdir = os.path.abspath(os.path.join('experiments', hps.problem, hps.logdir)) + '/'
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    hps.logdirname = hps.logdir
    hps.logdir = logdir

    train_its, test_its = get_its(hps.n_batch_train, hps.n_batch_test, hps.n_train, hps.n_test)
    hps.train_its = train_its
    hps.test_its = test_its

    x_shape = [None, hps.patch_height, hps.patch_height, hps.n_channels]
    hps.x_shape = x_shape
    hps.n_dims = np.prod(x_shape[1:])

    # calculate data stats and baselines
    logging.trace('calculating data stats and baselines...')
    hps.calc_pat_stats_and_baselines_only = True
    pat_stats, nll_gauss, _, nll_sdn, _, tr_batch_sampler, ts_batch_sampler = initialize_data_stats_queues_baselines_histograms(hps, logdir)
    hps.nll_gauss = nll_gauss
    hps.nll_sdn = nll_sdn

    # prepare get data queues
    hps.mb_requeue = True  # requeue minibatches for future epochs
    logging.trace('preparing data queues...')
    hps.calc_pat_stats_and_baselines_only = False
    tr_im_que, ts_im_que, tr_pat_que, ts_pat_que, tr_batch_que, ts_batch_que = \
        initialize_data_stats_queues_baselines_histograms(hps, logdir, tr_batch_sampler=tr_batch_sampler, ts_batch_sampler=ts_batch_sampler)
    # hps.save_batches = True

    print_train_test_stats(hps)

    input_shape = x_shape

    # Build noise flow graph
    logging.trace('Building NoiseFlow...')
    is_training = tf.placeholder(tf.bool, name='is_training')
    x = tf.placeholder(tf.float32, x_shape, name='noise_image')
    y = tf.placeholder(tf.float32, x_shape, name='clean_image')
    nlf0 = tf.placeholder(tf.float32, [None], name='nlf0')
    nlf1 = tf.placeholder(tf.float32, [None], name='nlf1')
    iso = tf.placeholder(tf.float32, [None], name='iso')
    cam = tf.placeholder(tf.float32, [None], name='cam')
    lr = tf.placeholder(tf.float32, None, name='learning_rate')

    # initialization of signal, gain, and camera parameters
    if hps.sidd_cond == 'mix':
        init_params(hps)

    # NoiseFlow model
    nf = NoiseFlow(input_shape[1:], is_training, hps)
    loss_val, sd_z = nf.loss(x, y, nlf0=nlf0, nlf1=nlf1, iso=iso, cam=cam)
    x_sample = nf.sample(y, 1.0, y, nlf0, nlf1, iso, cam)

    logging.trace('preparing optimizer')
    train_op = get_optimizer(hps, lr, loss_val)

    # save variable names and number of parameters
    vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    vars_files = os.path.join(hps.logdir, 'model_vars.txt')
    with open(vars_files, 'w') as vf:
        vf.write(str(vs))
    hps.num_params = int(np.sum([np.prod(v.get_shape().as_list())
                                 for v in tf.trainable_variables()]))
    logging.trace('number of parameters = %d' % hps.num_params)
    hps_logger(logdir + 'hps.txt', hps, nf.get_layer_names(), hps.num_params)

    # create session
    sess = tf.Session()
    n_processed = 0
    train_time = 0.0
    test_loss_best = np.inf

    # create a saver.
    saver = tf.train.Saver(max_to_keep=0)  # keep all models

    # checkpoint directory
    ckpt_dir = os.path.join(hps.logdir, 'ckpt')
    ckpt_path = os.path.join(ckpt_dir, 'model.ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    # sampling temperature (default = 1.0)
    if hps.temp is None:
        hps.temp = 1.0

    # setup the output log
    train_logger = test_logger = None
    log_columns = ['epoch', 'NLL']
    # NLL: negative log likelihood
    # NLL_G: for Gaussian baseline
    # NLL_SDN: for camera NLF baseline
    # sdz: standard deviation of the base measure (sanity check)
    log_columns = log_columns + ['NLL_G', 'NLL_SDN', 'sdz']
    if hps.do_sample:
        log_columns.append('sample_time')
    else:
        train_logger = ResultLogger(logdir + 'train.txt', log_columns + ['train_time'], hps.continue_training)
        test_logger = ResultLogger(logdir + 'test.txt', log_columns + ['msg'], hps.continue_training)
    sample_logger = ResultLogger(logdir + 'sample.txt', log_columns + ['KLD_G', 'KLD_NLF', 'KLD_NF', 'KLD_R'],
                                 hps.continue_training)

    tcurr = time.time()
    train_results = []
    test_results = []
    sample_results = []

    logging.trace('initializing variables')
    sess.run(tf.global_variables_initializer())

    # continue training?
    start_epoch = 1
    logging.trace('continue_training = ' + str(hps.continue_training))
    if hps.continue_training:
        if hps.restore_epoch:
            last_epoch = hps.restore_epoch
            restore_model(ckpt_dir, sess, saver, last_epoch)
        else:
            last_epoch = restore_last_model(ckpt_dir, sess, saver)
        start_epoch = 1 + last_epoch

    _lr = hps.lr
    _nlf0 = None
    _nlf1 = None
    t_train = t_test = t_sample = dsample = is_best = sd_z_tr = sd_z_ts = 0
    kldiv3 = None

    # Epochs
    logging.trace('totoal number 0f variables = %d' % len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
    logging.trace('Starting training/testing/samplings.')
    logging.trace('Logging to ' + logdir)

    summ = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./tensorboard_data_5/SRGB_noise_flow_fixed_image_loader_12")
    writer.add_graph(sess.graph)

    for epoch in range(start_epoch, hps.epochs + 1):

        # Testing
        if (not hps.do_sample) and \
                (epoch < 10 or (epoch < 100 and epoch % 10 == 0) or epoch % hps.epochs_full_valid == 0.):
            t = time.time()
            test_epoch_loss = []

            # multi-thread testing (faster)
            test_epoch_loss_que = queue.Queue()
            sd_z_que_ts = queue.Queue()
            sd_z_ts = 0

            test_multithread(sess, ts_batch_que, loss_val, sd_z, x, y, nlf0, nlf1, iso, cam, is_training,
                             test_epoch_loss_que, sd_z_que_ts, test_its, nthr=hps.n_train_threads,
                             requeue=not hps.mb_requeue)

            assert test_epoch_loss_que.qsize() == test_its
            for tt in range(test_its):
                test_epoch_loss.append(test_epoch_loss_que.get())
                sd_z_ts += sd_z_que_ts.get()
            sd_z_ts /= test_its

            mean_test_loss = np.mean(test_epoch_loss)
            test_results.append(mean_test_loss)

            # Save checkpoint
            saver.save(sess, ckpt_path, global_step=epoch)

            # best model?
            if test_results[-1] < test_loss_best:
                test_loss_best = test_results[-1]
                saver.save(sess, ckpt_path + '.best')
                is_best = 1
            else:
                is_best = 0

            # log
            log_dict = {'epoch': epoch, 'NLL': test_results[-1], 'NLL_G': nll_gauss, 'NLL_SDN': nll_sdn, 'sdz': sd_z_ts,
                        'msg': is_best}
            test_logger.log(log_dict)

            t_test = time.time() - t

        # End testing if & loop
        import ipdb;ipdb.set_trace()
        # Sampling (optional)
        do_sampling = False  # make this true to perform sampling
        if do_sampling and ((epoch < 10 or (epoch < 100 and epoch % 10 == 0) or  # (is_best == 1) or
                             epoch % hps.epochs_full_valid * 2 == 0.)):
            for temp in [1.0]:  # using only default temperature
                t_sample = time.time()
                hps.temp = float(temp)
                sample_epoch_loss = []

                # multi-thread sampling (faster)
                sample_epoch_loss_que = queue.Queue()
                sd_z_que_sam = queue.Queue()
                kldiv_que = queue.Queue()
                sd_z_sam = 0.0
                kldiv1 = np.ndarray([4])
                kldiv1[:] = 0.0
                kldiv3 = np.zeros(4)

                is_cond = hps.sidd_cond != 'uncond'

                # sample (forward)
                x_sample = sample_sidd_tf(sess, nf, is_training, hps.temp, y, nlf0, nlf1, iso, cam, is_cond)

                sample_multithread(sess, ts_batch_que, loss_val, sd_z, x, x_sample, y, nlf0, nlf1, iso, cam,
                                   is_training, sample_epoch_loss_que, sd_z_que_sam, kldiv_que,
                                   test_its, nthr=hps.n_train_threads, requeue=not hps.mb_requeue,
                                   sc_sd=pat_stats['sc_in_sd'], epoch=epoch)

                # assert sample_epoch_loss_que.qsize() == test_its
                nqs = sample_epoch_loss_que.qsize()
                for tt in range(nqs):
                    sample_epoch_loss.append(sample_epoch_loss_que.get())
                    sd_z_sam += sd_z_que_sam.get()
                    kldiv3 += kldiv_que.get()
                sd_z_sam /= nqs
                kldiv3 /= np.repeat(nqs, len(kldiv3))

                mean_sample_loss = np.mean(sample_epoch_loss)
                sample_results.append(mean_sample_loss)

                t_sample = time.time() - t_sample

                # log
                log_dict = {'epoch': epoch, 'NLL': sample_results[-1], 'NLL_G': nll_gauss,
                            'NLL_SDN': nll_sdn, 'sdz': sd_z_sam, 'sample_time': t_sample, 'KLD_G': kldiv3[0],
                            'KLD_NLF': kldiv3[1], 'KLD_NF': kldiv3[2], 'KLD_R': kldiv3[3]}
                sample_logger.log(log_dict)

        # Training loop
        t_curr = 0
        if not hps.do_sample:
            t = time.time()
            train_epoch_loss = []

            # multi-thread training (faster)
            train_epoch_loss_que = queue.Queue()
            sd_z_que_tr = queue.Queue()
            n_processed_que = queue.Queue()
            sd_z_tr = 0

            train_multithread(sess, epoch, tr_batch_que, loss_val, sd_z, summ, writer, train_op, x, y, nlf0, nlf1, iso, cam,
                              lr, is_training, _lr, n_processed_que, train_epoch_loss_que, sd_z_que_tr,
                              train_its, nthr=hps.n_train_threads, requeue=not hps.mb_requeue)

            assert train_epoch_loss_que.qsize() == train_its
            for tt in range(train_its):
                train_epoch_loss.append(train_epoch_loss_que.get())
                n_processed += n_processed_que.get()
                sd_z_tr += sd_z_que_tr.get()
            sd_z_tr /= train_its

            t_curr = time.time() - tcurr
            tcurr = time.time()

            mean_train_loss = np.mean(train_epoch_loss)
            train_results.append(mean_train_loss)
            # print("epoch: %s, losses:%s" % (epoch, np.isnan(np.sum(train_epoch_loss))))
            t_train = time.time() - t
            train_time += t_train

            train_logger.log({'epoch': epoch, 'train_time': int(train_time),
                              'NLL': train_results[-1], 'NLL_G': nll_gauss, 'NLL_SDN': nll_sdn, 'sdz': sd_z_tr})
        # End training

        # print results of train/test/sample
        tr_l = train_results[-1] if len(train_results) > 0 else 0
        ts_l = test_results[-1] if len(test_results) > 0 else 0
        sam_l = sample_results[-1] if len(sample_results) > 0 else 0
        if epoch < 10 or (epoch < 100 and epoch % 10 == 0) or \
                epoch % hps.epochs_full_valid == 0.:
            # E: epoch
            # tr, ts, tsm, tv: time of training, testing, sampling, visualization
            # T: total time
            # tL, sL, smL: loss of training, testing, sampling
            # SDr, SDs: std. dev. of base measure in training and testing
            # B: 1 if best model, 0 otherwise
            print('%s %s %s E=%d tr=%.1f ts=%.1f tsm=%.1f tv=%.1f T=%.1f '
                  'tL=%5.1f sL=%5.1f smL=%5.1f SDr=%.1f SDs=%.1f B=%d' %
                  (str(datetime.now())[11:16], host, hps.logdirname, epoch, t_train, t_test, t_sample, dsample, t_curr,
                   tr_l, ts_l, sam_l, sd_z_tr, sd_z_ts, is_best),
                  end='')
            if kldiv3 is not None:
                print(' ', end='')
                # marginal KL divergence of noise samples from: Gaussian, camera-NLF, and NoiseFlow, respectively
                print(','.join('{0:.3f}'.format(kk) for kk in kldiv3), end='')
            print('', flush=True)

    total_time = time.time() - total_time
    logging.trace('Total time = %f' % total_time)
    with open(path.join(logdir, 'total_time.txt'), 'w') as f:
        f.write('total_time (s) = %f' % total_time)
    logging.trace("Finished!")


if __name__ == "__main__":
    import signal

    # This enables a ctr-C without triggering errors
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))
    hps = arg_parser()

    main(hps)
