# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from os.path import exists
import numpy as np
import matplotlib.pyplot as plt
from sidd.Initialization import get_image_ques


class HistogramCalculator:
    def __init__(self, hps, logdir):
        self.hps = hps
        self.logdir = logdir

    def calc_hists(self):
        """Calculates 3 4-channel histograms of n_im noisy, clean, and noise images pulled from im_que"""

        if exists(os.path.join(self.logdir, 'tr_hists.npy')):
            tr_hists, ts_hists = self.load_hists()
            self.plot_hists(tr_hists, ts_hists, self.logdir)
            print('calc_hists: loaded.')
            return tr_hists, ts_hists

        tr_im_que, ts_im_que = get_image_ques(self.hps, requeue=False)
        tr_hists = self.calc_hists_ex(tr_im_que, self.hps.n_tr_inst * self.hps.n_train_per_scene)
        ts_hists = self.calc_hists_ex(ts_im_que, self.hps.n_ts_inst * self.hps.n_test_per_scene)
        # save
        self.save_hists(tr_hists, ts_hists)
        # plot
        self.plot_hists(tr_hists, ts_hists, self.logdir)
        print('calc_hists: done.')
        return tr_hists, ts_hists

    @staticmethod
    def calc_hists_ex(im_que, n_im):
        n_bins = 100
        bin_width = 1. / n_bins
        bin_edges = np.arange(0, 1 + bin_width, bin_width)
        bin_edges_noise = np.arange(-0.5, 0.5 + bin_width, bin_width)
        hists = {'noisy': np.zeros(n_bins),
                 'clean': np.zeros(n_bins),
                 'noise': np.zeros(n_bins),
                 'bin_edges': bin_edges,
                 'bin_edges_noise': bin_edges_noise,
                 'bin_width': bin_width,
                 'n_pix': 0}
        n_pix = 0
        for i in range(n_im):
            im_dict = im_que.get()
            noise = im_dict['in']
            clean = im_dict['gt']
            noisy = clean + noise
            bin_counts, _ = np.histogram(noisy, bin_edges)
            hists['noisy'] += bin_counts
            bin_counts, _ = np.histogram(clean, bin_edges)
            hists['clean'] += bin_counts
            bin_counts, _ = np.histogram(noise, bin_edges_noise)
            hists['noise'] += bin_counts
            n_pix += np.prod(noisy.shape)
            print('\rcalc_hists_ex: image %5d / %5d' % (i + 1, n_im), end='', flush=True)
        print('', flush=True)
        hists['n_pix'] = n_pix
        # hists['noisy'] /= n_pix
        # hists['clean'] /= n_pix
        # hists['noise'] /= n_pix
        return hists

    def save_hists(self, tr_hists, ts_hists):
        np.save(os.path.join(self.logdir, 'tr_hists.npy'), tr_hists)
        np.save(os.path.join(self.logdir, 'ts_hists.npy'), ts_hists)

    def load_hists(self):
        tr_hists = np.load(os.path.join(self.logdir, 'tr_hists.npy')).item()
        ts_hists = np.load(os.path.join(self.logdir, 'ts_hists.npy')).item()
        return tr_hists, ts_hists

    @staticmethod
    def plot_hists(tr_hists, ts_hists, logdir):

        plt.rcParams.update({'font.size': 14})

        fig1 = plt.figure()
        x = tr_hists['bin_edges'][:-1] + (.5 * tr_hists['bin_width'])
        plt.bar(x, tr_hists['noisy'], alpha=.5, width=tr_hists['bin_width'], label='Training subset')
        plt.bar(x, ts_hists['noisy'], alpha=.5, width=tr_hists['bin_width'], label='Testing subset')
        plt.legend()
        plt.title('Noisy images')
        plt.xlabel('Intensity')
        plt.ylabel('Pixel count')
        fig1.savefig(os.path.join(logdir, 'hists_noisy.png'))

        fig11 = plt.figure()
        x = tr_hists['bin_edges'][:-1] + (.5 * tr_hists['bin_width'])
        plt.bar(x, tr_hists['noisy'] / tr_hists['n_pix'], alpha=.5, width=tr_hists['bin_width'], label='Training subset')
        plt.bar(x, ts_hists['noisy'] / ts_hists['n_pix'], alpha=.5, width=ts_hists['bin_width'], label='Testing subset')
        plt.legend()
        plt.title('Noisy images (normalized)')
        plt.xlabel('Intensity')
        plt.ylabel('Pixel percentage')
        fig11.savefig(os.path.join(logdir, 'hists_noisy_norm.png'))

        fig2 = plt.figure()
        x = tr_hists['bin_edges'][:-1] + (.5 * tr_hists['bin_width'])
        plt.bar(x, tr_hists['clean'], alpha=.5, width=tr_hists['bin_width'], label='Training subset')
        plt.bar(x, ts_hists['clean'], alpha=.5, width=ts_hists['bin_width'], label='Testing subset')
        plt.legend()
        plt.title('Clean images')
        plt.xlabel('Intensity')
        plt.ylabel('Pixel count')
        fig2.savefig(os.path.join(logdir, 'hists_clean.png'))

        fig22 = plt.figure()
        x = tr_hists['bin_edges'][:-1] + (.5 * tr_hists['bin_width'])
        plt.bar(x, tr_hists['clean'] / tr_hists['n_pix'], alpha=.5, width=tr_hists['bin_width'], label='Training subset')
        plt.bar(x, ts_hists['clean'] / ts_hists['n_pix'], alpha=.5, width=ts_hists['bin_width'], label='Testing subset')
        plt.legend()
        plt.title('Clean images (normalized)')
        plt.xlabel('Intensity')
        plt.ylabel('Pixel percentage')
        fig22.savefig(os.path.join(logdir, 'hists_clean_norm.png'))

        fig3 = plt.figure()
        x = tr_hists['bin_edges_noise'][:-1] + (.5 * tr_hists['bin_width'])
        plt.bar(x, tr_hists['noise'], alpha=.5, width=tr_hists['bin_width'], label='Training subset')
        plt.bar(x, ts_hists['noise'], alpha=.5, width=ts_hists['bin_width'], label='Testing subset')
        plt.legend()
        plt.title('Noise layers')
        plt.xlabel('Intensity')
        plt.ylabel('Pixel count')
        # plt.autoscale(enable=True, axis='x')
        plt.xlim((-.2, .2))
        fig3.savefig(os.path.join(logdir, 'hists_noise.png'))

        fig33 = plt.figure()
        x = tr_hists['bin_edges_noise'][:-1] + (.5 * tr_hists['bin_width'])
        plt.bar(x, tr_hists['noise'] / tr_hists['n_pix'], alpha=.5, width=tr_hists['bin_width'], label='Training subset')
        plt.bar(x, ts_hists['noise'] / ts_hists['n_pix'], alpha=.5, width=ts_hists['bin_width'], label='Testing subset')
        plt.legend()
        plt.title('Noise layers (normalized)')
        plt.xlabel('Intensity')
        plt.ylabel('Pixel percentage')
        # plt.autoscale(enable=True, axis='x')
        plt.xlim((-.2, .2))
        fig33.savefig(os.path.join(logdir, 'hists_noise_norm.png'))
