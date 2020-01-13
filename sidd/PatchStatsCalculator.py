# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import os
import queue
import sys
import time
from threading import Thread

import numpy as np
from numpy import save, load

from sidd.sidd_utils import bpd


class PatchStatsCalculator:

    def __init__(self, mb_queue, patch_height=256, n_channels=4, save_dir='.', file_postfix='', n_threads=1, hps=None):
        # queue to pull mini batches dictionaries from
        self.mb_queue = mb_queue
        self.patch_height = patch_height
        self.n_channels = n_channels
        self.save_dir = save_dir
        self.file_postfix = file_postfix
        self.n_threads = n_threads
        self.hps = hps

        self.n_pat = 0
        self.threads = []
        self.stats = None

        self.init_pat_stats()

        # self.xxque = queue.Queue()

    def init_pat_stats(self):
        # initialize patch stats
        self.stats = dict({
            # patch-wise
            'in_mu': np.zeros((1, self.patch_height, self.patch_height, self.n_channels)),
            'gt_mu': np.zeros((1, self.patch_height, self.patch_height, self.n_channels)),
            'in_vr': np.zeros((1, self.patch_height, self.patch_height, self.n_channels)),
            'gt_vr': np.zeros((1, self.patch_height, self.patch_height, self.n_channels)),
            'in_sd': np.zeros((1, self.patch_height, self.patch_height, self.n_channels)),
            'gt_sd': np.zeros((1, self.patch_height, self.patch_height, self.n_channels)),
            'n_pat': 0,
            # scalars
            'sc_in_mu': 0, 'sc_gt_mu': 0, 'sc_in_vr': 0, 'sc_gt_vr': 0, 'sc_in_sd': 0, 'sc_gt_sd': 0, 'n_pix': 0
        })

    @staticmethod
    def stats_exist(save_dir, file_postfix):
        if os.path.exists(os.path.join(save_dir, 'pat_stats%s.npy') % file_postfix):
            return True
        return False

    @staticmethod
    def nll_files_exist(save_dir, file_postfix):
        if os.path.exists(os.path.join(save_dir, 'nll_bpd_gauss%s.npy') % file_postfix):
            return True
        return False

    def calc_stats(self, just_save=False):
        if just_save:
            logging.trace('just_save')
            save(os.path.join(self.save_dir, 'pat_stats%s.npy') % self.file_postfix, self.stats)
            return self.stats

        t0 = time.time()
        divs = self.divide_parts(self.hps.train_its, self.n_threads)
        out_que = queue.Queue(self.n_threads)
        cnt_mb_que = queue.Queue()
        # initialize threads
        for t in range(self.n_threads):
            self.threads.append(Thread(
                target=self.calc_patch_stats, args=(t, self.mb_queue, divs[t], out_que, cnt_mb_que)))
            self.threads[t].start()
        print('')
        for t in range(self.n_threads):
            self.threads[t].join()
        self.weighted_stats(out_que)
        self.calc_scalar_stats()
        save(os.path.join(self.save_dir, 'pat_stats%s.npy') % self.file_postfix, self.stats)
        logging.trace('calc. stats: time = %3.0f s ' % (time.time() - t0))
        return self.stats

    def calc_baselines(self, ts_mb_que):

        # out = np.zeros((self.hps.test_its, 4))
        nll_gauss_lst = []
        nll_sdn_lst = []
        for i in range(self.hps.test_its):
            mb = ts_mb_que.get()

            x = mb['_x']
            y = mb['_y']
            nlf0 = mb['nlf0']
            nlf1 = mb['nlf1']
            vr = y * nlf0 + nlf1

            vr_gauss = self.stats['sc_in_vr']
            nll_mb_gauss = 0.5 * \
                 (np.log(2 * np.pi) + np.log(vr_gauss) + (x) ** 2 / vr_gauss)
            nll_mb_gauss = np.sum(nll_mb_gauss, axis=(1, 2, 3))
            nll_gauss_lst.append(nll_mb_gauss)

            nll_mb = 0.5 * \
                (np.log(2 * np.pi) + np.log(vr) + (x) ** 2 / vr)
            nll_mb = np.sum(nll_mb, axis=(1, 2, 3))
            nll_sdn_lst.append(nll_mb)

        nll_sdn = np.mean(nll_sdn_lst)
        # print(str(np.mean(nll_sdn_lst)), end='******')
        # np.savetxt(os.path.join(self.save_dir, 'out.txt'), out, fmt='%f')
        nll_gauss = np.mean(nll_gauss_lst)
        save(os.path.join(self.save_dir, 'nll_bpd_gauss%s.npy') % self.file_postfix, (nll_gauss, 0))
        save(os.path.join(self.save_dir, 'nll_bpd_sdn%s.npy') % self.file_postfix, (nll_sdn, 0))
        return nll_gauss, 0, nll_sdn, 0

        t0 = time.time()
        divs = self.divide_parts(self.hps.test_its, self.n_threads)
        out_que = queue.Queue(self.n_threads)
        cnt_mb_que = queue.Queue()
        # initialize threads
        self.threads = []
        for t in range(self.n_threads):
            self.threads.append(Thread(
                target=self.calc_baselines_thread, args=(t, ts_mb_que, divs[t], out_que, cnt_mb_que)))
            self.threads[t].start()
        for t in range(self.n_threads):
            self.threads[t].join()
        nll_gauss, bpd_gauss, nll_sdn, bpd_sdn = self.weighted_baselines(out_que)
        save(os.path.join(self.save_dir, 'nll_bpd_gauss%s.npy') % self.file_postfix, (nll_gauss, bpd_gauss))
        save(os.path.join(self.save_dir, 'nll_bpd_sdn%s.npy') % self.file_postfix, (nll_sdn, bpd_sdn))
        print('calc_baselines: time = %3.0f s ' % (time.time() - t0))
        return nll_gauss, bpd_gauss, nll_sdn, bpd_sdn

    def weighted_baselines(self, out_que):
        sum_g = 0  # for Gaussian NLL
        sum_s = 0  # for signal-dependent noise NLL
        sum_v = 0  # for signal-dependent noise NLL
        qsz = out_que.qsize()
        for i in range(qsz):
            d = out_que.get()
            sum_g += d['sum_g']
            sum_v += d['sum_v']
            sum_s += d['sum_s']
        # n = number of test patches, or pixels?
        n = self.hps.test_its * self.hps.n_batch_test
        print('***** n (patches) = %f' % n)
        ndims = self.hps.patch_height * self.patch_height * 4
        nll_gauss = (ndims / 2.0) * np.log(2 * np.pi * self.stats['sc_in_vr']) + (sum_g / n)  # mean NLL
        nll_sdn = (ndims / 2.0) * np.log(2 * np.pi) + (sum_v / n) + (sum_s / n)  # mean NLL
        bpd_gauss = bpd(nll_gauss, self.hps.n_bins, self.hps.n_dims)
        bpd_sdn = bpd(nll_sdn, self.hps.n_bins, self.hps.n_dims)
        return nll_gauss, bpd_gauss, nll_sdn, bpd_sdn

    def calc_baselines_thread(self, thr_id, ts_mb_que, n_batch, out_que, cnt_mb_que):
        sum_g = 0  # for Gaussian NLL
        sum_s = 0  # for signal-dependent noise NLL
        sum_v = 0  # for signal-dependent noise NLL
        mb = None
        for i in range(n_batch):
            # print('\rcalc. baselines %3.1f%% ' % (cnt_mb_que.qsize() * 100. / self.hps.test_its), end='', flush=True)
            mb = ts_mb_que.get()

            nlf0 = mb['nlf0']
            nlf1 = mb['nlf1']

            for p in range(self.hps.n_batch_test):  # for each patch

                x_pat_sqr = mb['_x'][p, :, :, :] ** 2  # mean subtracted in ImageLoader

                sum_g += 0.5 * np.sum(x_pat_sqr) / self.stats['sc_in_vr']

                vr = mb['_y'][p, :, :, :] * nlf0[p, :, :, :] + nlf1[p, :, :, :]

                sum_v += 0.5 * np.sum(np.log(vr))

                sum_s += 0.5 * np.sum(x_pat_sqr / vr)

            cnt_mb_que.put(1)  # just a counter
        out_que.put({'sum_g': sum_g, 'sum_v': sum_v, 'sum_s': sum_s})

    def calc_patch_stats(self, thread_id, mb_que, n_batch, out_que, cnt_mb_que):
        # t0 = time.time()
        # initialize patch stats
        stats = dict({
            # patch-wise
            'in_mu': np.zeros((1, self.patch_height, self.patch_height, self.n_channels)),
            'gt_mu': np.zeros((1, self.patch_height, self.patch_height, self.n_channels)),
            'in_vr': np.zeros((1, self.patch_height, self.patch_height, self.n_channels)),
            'gt_vr': np.zeros((1, self.patch_height, self.patch_height, self.n_channels)),
            'in_sd': np.zeros((1, self.patch_height, self.patch_height, self.n_channels)),
            'gt_sd': np.zeros((1, self.patch_height, self.patch_height, self.n_channels)),
            'n_pat': 0,
            # scalars
            'sc_in_mu': 0, 'sc_gt_mu': 0, 'sc_in_vr': 0, 'sc_gt_vr': 0, 'sc_in_sd': 0, 'sc_gt_sd': 0, 'n_pix': 0
        })
        n_pat = 0  # number of patches so far
        # ** first mini batch
        mb_dict = mb_que.get()  # blocking
        _x = mb_dict['_x']
        _y = mb_dict['_y']
        # v_batch = tr_mb_dict['_v']
        # n_batch = tr_mb_dict['_nlf']
        stats['in_mu'] = self.online_mean_step(stats['in_mu'], n_pat, _x[0, :, :, :])
        stats['gt_mu'] = self.online_mean_step(stats['gt_mu'], n_pat, _y[0, :, :, :])
        n_pat += 1
        for p in range(1, _x.shape[0]):  # note: starting from 1, not 0, for sample variance
            # this is sample variance, take square root at the end
            stats['in_vr'] = self.online_var_step(stats['in_vr'], stats['in_mu'], n_pat, _x[p, :, :, :])
            stats['gt_vr'] = self.online_var_step(stats['gt_vr'], stats['gt_mu'], n_pat, _y[p, :, :, :])
            stats['in_mu'] = self.online_mean_step(stats['in_mu'], n_pat, _x[p, :, :, :])
            stats['gt_mu'] = self.online_mean_step(stats['gt_mu'], n_pat, _y[p, :, :, :])
            n_pat += 1
        # ** second to last mini batch
        n_mb = 1
        for i in range(1, n_batch):  # for each mini batch
            # dequeue mini batch dictionary
            mb_dict = mb_que.get()  # blocking
            _x = mb_dict['_x']
            _y = mb_dict['_y']
            # v_batch = tr_mb_dict['_v']
            # n_batch = tr_mb_dict['_nlf']
            # calculate stats
            for p in range(0, _x.shape[0]):
                # this is sample variance, take square root at the end
                stats['in_vr'] = self.online_var_step(stats['in_vr'], stats['in_mu'], n_pat, _x[p, :, :, :])
                stats['gt_vr'] = self.online_var_step(stats['gt_vr'], stats['gt_mu'], n_pat, _y[p, :, :, :])
                stats['in_mu'] = self.online_mean_step(stats['in_mu'], n_pat, _x[p, :, :, :])
                stats['gt_mu'] = self.online_mean_step(stats['gt_mu'], n_pat, _y[p, :, :, :])
                n_pat += 1
            n_mb += 1
            cnt_mb_que.put(1)
        stats['n_pat'] = n_pat
        out_que.put(stats)

    def calc_patch_stats_one_thread(self):
        self.init_pat_stats()
        t0 = time.time()
        self.n_pat = 0  # number of patches so far
        # ** first mini batch
        mb_dict = self.mb_queue.get()  # blocking
        _x = mb_dict['_x']
        _y = mb_dict['_y']
        # v_batch = tr_mb_dict['_v']
        # n_batch = tr_mb_dict['_nlf']
        self.stats['in_mu'] = self.online_mean_step(self.stats['in_mu'], self.n_pat, _x[0, :, :, :])
        self.stats['gt_mu'] = self.online_mean_step(self.stats['gt_mu'], self.n_pat, _y[0, :, :, :])
        self.n_pat += 1
        for p in range(1, _x.shape[0]):  # note: starting from 1, not 0, for sample variance
            # this is sample variance, take square root at the end
            self.stats['in_vr'] = self.online_var_step(self.stats['in_vr'], self.stats['in_mu'],
                                                       self.n_pat, _x[p, :, :, :])
            self.stats['gt_vr'] = self.online_var_step(self.stats['gt_vr'], self.stats['gt_mu'],
                                                       self.n_pat, _y[p, :, :, :])
            self.stats['in_mu'] = self.online_mean_step(self.stats['in_mu'], self.n_pat, _x[p, :, :, :])
            self.stats['gt_mu'] = self.online_mean_step(self.stats['gt_mu'], self.n_pat, _y[p, :, :, :])
            self.n_pat += 1
        # ** second to last mini batch
        n_mb = 1
        for i in range(1, self.hps.train_its):  # for each mini batch
            # dequeue mini batch dictionary
            mb_dict = self.mb_queue.get()  # blocking
            _x = mb_dict['_x']
            _y = mb_dict['_y']
            # v_batch = tr_mb_dict['_v']
            # n_batch = tr_mb_dict['_nlf']
            # calculate stats
            for p in range(0, _x.shape[0]):
                # this is sample variance, take square root at the end
                self.stats['in_vr'] = self.online_var_step(self.stats['in_vr'], self.stats['in_mu'],
                                                           self.n_pat, _x[p, :, :, :])
                self.stats['gt_vr'] = self.online_var_step(self.stats['gt_vr'], self.stats['gt_mu'],
                                                           self.n_pat, _y[p, :, :, :])
                self.stats['in_mu'] = self.online_mean_step(self.stats['in_mu'], self.n_pat, _x[p, :, :, :])
                self.stats['gt_mu'] = self.online_mean_step(self.stats['gt_mu'], self.n_pat, _y[p, :, :, :])
                self.n_pat += 1
            n_mb += 1
            t_cur = time.time() - t0
            rem_time = (t_cur / n_mb) * (self.hps.train_its - n_mb)
            # print('\rcalc_patch_stats_one_thread: batch #%5d/%5d   time = %3.0f s   rem. time = %3.0f s' %
            #       (n_mb, self.hps.train_its, t_cur, rem_time), end='')
        # print('')
        self.stats['n_pat'] = self.n_pat
        self.calc_scalar_stats()
        # save
        # save('pat_stats%s.npy' % self.file_postfix, self.pat_stats)
        save(os.path.join(self.save_dir, 'pat_stats%s.npy') % self.file_postfix, self.stats)
        return self.stats

    def calc_scalar_stats(self):
        # scalar mean and scalar sample variance
        k = self.stats['n_pat']
        g = self.patch_height * self.patch_height * self.n_channels
        self.stats['n_pix'] = k * g

        self.stats['sc_in_mu'] = np.mean(self.stats['in_mu'])
        self.stats['sc_gt_mu'] = np.mean(self.stats['gt_mu'])

        t_sum = np.sum(self.stats['in_vr']) + np.var(self.stats['in_mu']) * (k * (g - 1)) / (k - 1)
        self.stats['sc_in_vr'] = t_sum * (k - 1) / (k * g - 1)
        t_sum = np.sum(self.stats['gt_vr']) + np.var(self.stats['gt_mu']) * (k * (g - 1)) / (k - 1)
        self.stats['sc_gt_vr'] = t_sum * (k - 1) / (k * g - 1)

        if self.stats['sc_in_vr'] < sys.float_info.epsilon:
            self.stats['sc_in_vr'] = sys.float_info.epsilon
        if self.stats['sc_gt_vr'] < sys.float_info.epsilon:
            self.stats['sc_gt_vr'] = sys.float_info.epsilon

        # standard deviation (sd): take square root
        self.stats['in_sd'] = np.sqrt(self.stats['in_vr'])
        self.stats['gt_sd'] = np.sqrt(self.stats['gt_vr'])
        self.stats['sc_in_sd'] = np.sqrt(self.stats['sc_in_vr'])
        self.stats['sc_gt_sd'] = np.sqrt(self.stats['sc_gt_vr'])

    def calc_scalar_stats_p(self, stats, n_pat):
        # scalar mean and scalar sample variance
        k = n_pat
        g = self.patch_height * self.patch_height * self.n_channels
        stats['n_pix'] = k * g

        stats['sc_in_mu'] = np.mean(stats['in_mu'])
        stats['sc_gt_mu'] = np.mean(stats['gt_mu'])

        t_sum = np.sum(stats['in_vr']) + np.var(stats['in_mu']) * (k * (g - 1)) / (k - 1)
        stats['sc_in_vr'] = t_sum * (k - 1) / (k * g - 1)
        t_sum = np.sum(stats['gt_vr']) + np.var(stats['gt_mu']) * (k * (g - 1)) / (k - 1)
        stats['sc_gt_vr'] = t_sum * (k - 1) / (k * g - 1)

        # standard deviation (sd): take square root
        stats['in_sd'] = np.sqrt(stats['in_vr'])
        stats['gt_sd'] = np.sqrt(stats['gt_vr'])
        stats['sc_in_sd'] = np.sqrt(stats['sc_in_vr'])
        stats['sc_gt_sd'] = np.sqrt(stats['sc_gt_vr'])

    @staticmethod
    def online_mean_step(cur_mean, cur_n, new_point):
        if cur_n < 0:
            raise Exception
        return cur_mean + (new_point - cur_mean) / (cur_n + 1)

    @staticmethod
    def online_var_step(cur_var, cur_mean, cur_n, new_point):
        if cur_n < 1:
            raise Exception
        return cur_var + ((new_point - cur_mean) ** 2 / (cur_n + 1)) - (cur_var / cur_n)

    @staticmethod
    def weighted_mean(means, weights):
        w_mean = np.zeros(means[0].shape)
        g = len(means)  # number of partitions
        sum_w = 0
        for i in range(g):
            w_mean += means[i] * weights[i]
            sum_w += weights[i]
        return w_mean / sum_w

    @staticmethod
    def weighted_var(means, weighted_mean, vars, weights):
        n = np.sum(weights)
        w_var = np.zeros(vars[0].shape)
        g = len(means)  # number of partitions
        for i in range(g):
            w_var += (weights[i] - 1) * vars[i]
            w_var += weights[i] * (means[i] - weighted_mean) ** 2
        w_var /= (n - 1)
        return w_var

    def weighted_stats(self, stats_que):
        tmp_que = queue.Queue(stats_que.qsize())
        # mean
        g = stats_que.qsize()  # number of partitions
        self.stats['in_mu'] = np.zeros((1, self.hps.patch_height, self.hps.patch_height, self.n_channels))
        self.stats['gt_mu'] = np.zeros((1, self.hps.patch_height, self.hps.patch_height, self.n_channels))
        tot_pats = 0
        for i in range(0, g):
            st = stats_que.get()
            tmp_que.put(st)
            self.stats['in_mu'] += st['in_mu'] * st['n_pat']
            self.stats['gt_mu'] += st['gt_mu'] * st['n_pat']
            tot_pats += st['n_pat']
        self.stats['in_mu'] /= tot_pats
        self.stats['gt_mu'] /= tot_pats
        self.stats['n_pat'] = tot_pats
        # variance
        self.stats['in_vr'] = np.zeros(st['in_vr'].shape)
        self.stats['gt_vr'] = np.zeros(st['gt_vr'].shape)
        for i in range(0, g):
            st = tmp_que.get()
            self.stats['in_vr'] += (st['n_pat'] - 1) * st['in_vr']
            self.stats['in_vr'] += st['n_pat'] * (st['in_mu'] - self.stats['in_mu']) ** 2
            self.stats['gt_vr'] += (st['n_pat'] - 1) * st['gt_vr']
            self.stats['gt_vr'] += st['n_pat'] * (st['gt_mu'] - self.stats['gt_mu']) ** 2
        self.stats['in_vr'] /= (tot_pats - 1)
        self.stats['gt_vr'] /= (tot_pats - 1)

    @staticmethod
    def divide_parts(n, n_parts):
        """divide a number into a list of parts"""
        (div, rem) = divmod(n, n_parts)
        divs = [div] * n_parts
        if rem != 0:
            for r in range(rem):
                divs[r] += 1
        return divs

    def load_pat_stats(self):
        self.stats = np.load(os.path.join(self.save_dir, 'pat_stats%s.npy') % self.file_postfix,
                             allow_pickle=True).item()  # dictionary
        return self.stats

    def load_gauss_baseline(self):
        return load(os.path.join(self.save_dir, 'nll_bpd_gauss%s.npy') % self.file_postfix)

    def load_sdn_baseline(self):
        return load(os.path.join(self.save_dir, 'nll_bpd_sdn%s.npy') % self.file_postfix)
