# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from broken_ax import broken_ax


def get_sample_data(path):
    try:
        df_sample = pd.read_csv(path + '/sample.txt', sep='\t')
        return df_sample
    except Exception as ex:
        print(str(ex))
        print('error reading: %s' % path + '/sample.txt')
        return None


def get_sample_kld(df_sample):
    xs = df_sample['epoch']
    kld_g = df_sample['KLD_G']
    kld_nlf = df_sample['KLD_NLF']
    kld_nf = df_sample['KLD_NF']
    kld_r = df_sample['KLD_R']
    return xs, kld_g, kld_nlf, kld_nf, kld_r


def plot_kld(data_path, plot_dict):
    subdir = plot_dict['folders'][0]['folder']
    fpath = os.path.join(data_path, subdir)
    df_sample = get_sample_data(fpath)
    if df_sample is None:
        return
    xs, kld_g, kld_nlf, kld_nf, kld_r = get_sample_kld(df_sample)

    fig = plt.figure(figsize=plot_dict['figsize'])
    plt.plot(xs, kld_g, label='Gaussian')
    plt.plot(xs, kld_nlf, label='Camera NLF')
    plt.plot(xs, kld_nf, label='Noise Flow')
    plt.plot(xs, kld_r, label='Real noise')
    if plot_dict['xlims'] is not None:
        plt.xlim(plot_dict['xlims'])
    if plot_dict['ylims'] is not None:
        plt.ylim(plot_dict['ylims'])
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    plt.xlabel('Epoch')
    plt.ylabel('KL divergence')
    plt.title(plot_dict['folders'][0]['folder'])
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(fpath, 'kld.png'))
    # plt.show()


def plot_kld_multi(data_path, plot_dict, clrs):
    n_folders = len(plot_dict['folders'])
    xs = [None] * n_folders
    kld_g = None
    kld_nlf = None
    kld_nf = [None] * n_folders
    kld_r = None
    fpath = None
    maxlen = 0
    for i in range(n_folders):
        subdir = plot_dict['folders'][i]['folder']
        fpath = os.path.join(data_path, subdir)
        df_sample = get_sample_data(fpath)
        if df_sample is None:
            continue
        xs[i], kld_g_, kld_nlf_, kld_nf[i], kld_r_ = get_sample_kld(df_sample)
        if maxlen < len(kld_r_):
            maxlen = len(kld_r_)
            xg = xs[i]
            kld_g = kld_g_
            kld_nlf = kld_nlf_
            kld_r = kld_r_

    fig = plt.figure(figsize=plot_dict['figsize'])

    # best NLL
    bnll = []

    plt.plot(xg, kld_g, label='Gaussian', linestyle='-.', color=clrs[0])
    plt.plot(xg, kld_nlf, label='Camera NLF', linestyle='-.', color=clrs[1])

    bnll.append(np.min(kld_g))
    bnll.append(np.min(kld_nlf))

    for i in range(n_folders):
        if xs[i] is None:
            print('skipping %d, %s' % (i, plot_dict['folders'][i]['folder']))
            continue
        plt.plot(xs[i], kld_nf[i], label=plot_dict['folders'][i]['legend'], color=clrs[i + 2], linestyle='-')

        bnll.append(np.min(kld_nf[i]))

    plt.plot(xg, kld_r, label='Real noise', linestyle='--', color=clrs[3])

    # bnll.append(np.min(kld_r))
    improv = []
    for t in range(len(bnll)):
        improv.append((np.absolute(bnll[-1]) - np.absolute(bnll[t])) / np.absolute(bnll[t]))
    # bnll.append(np.absolute(bnll[2]) - np.absolute(bnll[0]))
    # bnll.append(np.absolute(bnll[2]) - np.absolute(bnll[1]))
    # bnll.append(bnll[4] / np.absolute(bnll[0]))
    # bnll.append(bnll[5] / np.absolute(bnll[1]))
    np.savetxt(os.path.join(fpath, 'best_kld.txt'), bnll)
    np.savetxt(os.path.join(fpath, 'best_improv.txt'), improv)

    if plot_dict['xlims'] is not None:
        plt.xlim(plot_dict['xlims'])
    if plot_dict['ylims'] is not None:
        plt.ylim(plot_dict['ylims'])
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    plt.xlabel('Epoch')
    plt.ylabel(r'Marginal $D_{KL}$')
    if plot_dict['title'] is not None:
        plt.title(plot_dict['title'])
    plt.legend(loc='best', bbox_to_anchor=None, prop={'size': 14}, ncol=1, fancybox=False, shadow=False)
    plt.tight_layout()
    fig.savefig(os.path.join(fpath, '..', 'kld_' + plot_dict['fig_fn'] + '.png'))
    fig.savefig(os.path.join(fpath, '..', 'kld_' + plot_dict['fig_fn'] + '.pdf'))
    # plt.show()
