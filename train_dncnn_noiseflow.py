# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import datetime
import glob
import os
import queue
import random
import re
from threading import Thread

import keras.backend as krs
import numpy as np
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Subtract
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils.training_utils import multi_gpu_model
# from keras.utils import multi_gpu_model

import sidd.data_loader as loader
from borealisflows.NoiseFlowWrapper import NoiseFlowWrapper

import tensorflow as tf
import pandas as pd

# from SIDD:
min_est_sigma = 0.24186
max_est_sigma = 11.507
min_cam_nlf = [0.00011841, 2.0024e-06]
max_cam_nlf = [0.021949, 0.0017506]


def load_cam_iso_nlf():
    cin = pd.read_csv('cam_iso_nlf.txt')
    cin = cin.drop_duplicates()
    cin = cin.set_index('cam_iso', drop=False)
    return cin


cam_iso_nlf = load_cam_iso_nlf()
iso_vals = [100.0, 400.0, 800.0, 1600.0, 3200.0]
cam_ids = [0, 1, 3, 3, 4]  # IP, GP, S6, N6, G4
cam_vals = ['IP', 'GP', 'S6', 'N6', 'G4']

noise_flow_path = '../models/NoiseFlow/ckpt/'

# Params
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--train_data', type=str, help='path of train data')
parser.add_argument('--max_epoch', default=2000, type=int, help='number of train epochs')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_every', default=1, type=int, help='save model at every x epochs')
parser.add_argument('--fine_tune', action='store_true',
                    help='whether to fine tune DnCNN, needs to specify --pre_model_path')
parser.add_argument('--pre_model_path', type=str, default='',
                    help='path to previous model for fine-tuning')
parser.add_argument('--num_gpus', type=int, default=1, help='number of GPUs')
args = parser.parse_args()
print('args:')
print(args)
print('===')

save_dir = os.path.join('models', args.model)

if not os.path.exists(save_dir):
    os.mkdir(save_dir)


def dncnn(depth, filters=64, image_channels=4, use_bnorm=True):
    layer_count = 0
    inpt = Input(shape=(None, None, image_channels), name='input' + str(layer_count))
    # 1st layer, Conv+relu
    layer_count += 1
    x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
               name='conv' + str(layer_count))(inpt)
    layer_count += 1
    x = Activation('relu', name='relu' + str(layer_count))(x)
    # depth-2 layers, Conv+BN+relu
    for i in range(depth - 2):
        layer_count += 1
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal', padding='same',
                   use_bias=False, name='conv' + str(layer_count))(x)
        if use_bnorm:
            layer_count += 1
            # x = BatchNormalization(axis=3, momentum=0.1,epsilon=0.0001, name = 'bn'+str(layer_count))(x)
            x = BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='bn' + str(layer_count))(x)
        layer_count += 1
        x = Activation('relu', name='relu' + str(layer_count))(x)
        # last layer, Conv
    layer_count += 1
    x = Conv2D(filters=image_channels, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal',
               padding='same', use_bias=False, name='conv' + str(layer_count))(x)
    layer_count += 1
    x = Subtract(name='subtract' + str(layer_count))([inpt, x])  # input - noise
    model = Model(inputs=inpt, outputs=x)

    return model


def find_last_checkpoint(save_dir1):
    file_list = glob.glob(os.path.join(save_dir1, 'model_*.hdf5'))  # get name list of all .hdf5 files
    # file_list = os.listdir(save_dir)
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).hdf5.*", file_)
            # print(result[0])
            epochs_exist.append(int(result[0]))
        initial_epoch1 = max(epochs_exist)
    else:
        initial_epoch1 = 0
    return initial_epoch1


def log(*args1, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args1, **kwargs, flush=True)


def lr_schedule(epoch):
    initial_lr = args.lr
    if epoch <= 30:
        lr = initial_lr
    elif epoch <= 60:
        lr = initial_lr / 10
    elif epoch <= 80:
        lr = initial_lr / 20
    else:
        lr = initial_lr / 20
    log('current learning rate is %2.8f' % lr)
    return lr


def sample_thread(idx, noise_flow, x_pat, b1, b2, iso, cam, out_que):
    out_que.put(noise_flow.sample_noise_nf(x_pat, b1, b2, iso, cam))


def sample_thread_cont(noise_flow, cam_iso_nlf, in_que, out_que):
    n_cam_iso = cam_iso_nlf['cam_iso'].count()
    while True:
        (i, batch_x, batch_info) = in_que.get()

        cam_iso_idx = random.randint(0, n_cam_iso - 1)
        row = cam_iso_nlf.iloc[cam_iso_idx]
        cam = cam_vals.index(row['cam_iso'][:2])
        iso = float(row['cam_iso'][3:])

        aug_gauss = random.uniform(0.0, 1.0) > 0.5
        if aug_gauss:
            sig = random.uniform(min_est_sigma, max_est_sigma)  # Gaussian sigma in [0, 255]
            noise = np.random.normal(0, sig / 255.0, batch_x.shape)  # noise
        else:
            noise = noise_flow.sample_noise_nf(batch_x, 0.0, 0.0, iso, cam)

        batch_y = batch_x + noise
        batch_y = np.clip(batch_y, 0.0, 1.0)
        out_que.put((batch_y, batch_x))


def enqueue_thread(xs, cam_iso_info, indices, batch_size, in_que):
    while True:
        np.random.shuffle(indices)  # shuffle
        for i in range(0, len(indices), batch_size):
            batch_x = xs[indices[i:i + batch_size]]
            batch_info = cam_iso_info[i:i + batch_size]
            in_que.put((i, batch_x, batch_info))


def train_datagen(epoch_iter=2000, epoch_num=5, batch_size=128, data_dir=args.train_data, noise_flow=None):
    xs_noisy = None
    if args.model.__contains__('_Real'):
        xs, cam_iso_info, xs_noisy = loader.load_data_threads_with_noisy(data_dir)
    else:
        xs, cam_iso_info = loader.load_data_threads(data_dir)
    print('dataset size: %s' % str(xs.shape))
    assert len(xs) % args.batch_size == 0, \
        log(
            'make sure the last iteration has a full batchsize, '
            'this is important if you use batch normalization!')
    indices = list(range(xs.shape[0]))

    out_que = None
    if args.model.__contains__('DnCNN_NF'):
        in_que = queue.Queue(maxsize=1000)
        out_que = queue.Queue(maxsize=1000)
        enq_thr = Thread(target=enqueue_thread, args=(xs, cam_iso_info, indices, batch_size, in_que))
        enq_thr.start()
        thrs = [None] * 32  # number of sampling threads
        for k in range(len(thrs)):
            thrs[k] = Thread(target=sample_thread_cont, args=(noise_flow, cam_iso_nlf, in_que, out_que))
            thrs[k].start()

    while True:
        for _ in range(1):  # epoch_num):
            if args.model.__contains__('_Real'):
                for i in range(0, len(indices), batch_size):
                    batch_x = xs[indices[i:i + batch_size]]  # clean
                    batch_y = xs_noisy[indices[i:i + batch_size]]  # noisy
                    yield batch_y, batch_x
            elif args.model.__contains__('_NF'):
                for i in range(0, len(indices), batch_size):
                    yield out_que.get()  # batch_y, batch_x
            else:
                # np.random.shuffle(indices)  # shuffle
                for i in range(0, len(indices), batch_size):
                    batch_x = xs[indices[i:i + batch_size]]
                    if args.model.__contains__('_Gauss'):
                        sig = random.uniform(min_est_sigma, max_est_sigma)  # Gaussian sigma in [0, 255]
                        noise = np.random.normal(0, sig / 255.0, batch_x.shape)  # noise
                    elif args.model.__contains__('_SDN'):
                        b1 = random.uniform(min_cam_nlf[0], max_cam_nlf[0])
                        b2 = random.uniform(min_cam_nlf[1], max_cam_nlf[1])
                        sig = np.sqrt(b1 * batch_x + b2)  # in [0, 1]
                        noise = np.random.normal(0.0, sig, batch_x.shape)
                    elif args.model.__contains__('_CamNLF'):
                        idx = random.choice(cam_iso_nlf.index)
                        row = cam_iso_nlf.loc[idx]
                        b1 = row['beta1']
                        b2 = row['beta2']
                        if type(b1) != np.float64 and type(b1) != np.float32:
                            b1 = b1.iloc[0]
                            b2 = b2.iloc[0]
                        try:
                            sig = np.sqrt(b1 * batch_x + b2)  # in [0, 1]
                        except Exception as ex:
                            print(str(ex))
                        noise = np.random.normal(0.0, sig, batch_x.shape)
                    else:
                        raise Exception('Invalid model name')
                    # noise =  K.random_normal(ge_batch_y.shape, mean=0, stddev=args.sigma/255.0)
                    batch_y = batch_x + noise
                    batch_y = np.clip(batch_y, 0.0, 1.0)
                    yield batch_y, batch_x


def sum_squared_error(y_true, y_pred):
    return krs.sum(krs.square(y_pred - y_true)) / 2


def select_model():
    if args.fine_tune:
        # load trained model to fine-tune
        model1 = load_model(args.pre_model_path, compile=False)
    else:
        # model1 = dncnn(depth=17, filters=64, image_channels=4, use_bnorm=True)
        model1 = dncnn(depth=9, filters=32, image_channels=4, use_bnorm=True)
    return model1


if __name__ == '__main__':

    graph2 = tf.Graph()
    with graph2.as_default():
        if args.model.__contains__('DnCNN_NF'):
            nf_model = NoiseFlowWrapper(noise_flow_path)
        else:
            nf_model = None

    graph1 = tf.Graph()
    sess1 = tf.Session(graph=graph1)
    with graph1.as_default():
        with sess1.as_default():

            if args.num_gpus > 1:
                print('multi-gpu training...')
                with tf.device("/cpu:0"):
                    model1 = select_model()
                model1 = multi_gpu_model(model1, gpus=args.num_gpus)
            else:
                model1 = select_model()
            model1.summary()

            # load the last model in matconvnet style
            initial_epoch = find_last_checkpoint(save_dir1=save_dir)
            if initial_epoch > 0:
                print('resuming by loading epoch %03d' % initial_epoch)
                model1 = load_model(os.path.join(save_dir, 'model_%03d.hdf5' % initial_epoch), compile=False)

            # compile the model
            model1.compile(optimizer=Adam(0.001), loss=sum_squared_error)

            # use call back functions
            checkpointer = ModelCheckpoint(os.path.join(save_dir, 'model_{epoch:03d}.hdf5'),
                                           verbose=1, save_weights_only=False, period=args.save_every)
            csv_logger = CSVLogger(os.path.join(save_dir, 'log.csv'), append=True, separator=',')
            lr_scheduler = LearningRateScheduler(lr_schedule)

            print('start loading data and training...')
            stps = 103808 / args.batch_size  # dataset size / mini-batch size
            history = model1.fit_generator(
                train_datagen(batch_size=args.batch_size * args.num_gpus, noise_flow=nf_model),
                epochs=args.max_epoch, verbose=2, initial_epoch=initial_epoch, steps_per_epoch=stps,
                callbacks=[checkpointer, csv_logger, lr_scheduler])
