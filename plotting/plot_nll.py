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


def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret[n - 1:] = ret[n - 1:] / n
    for i in range(1, n - 1):
        ret[i] /= (i + 1)
    return ret


def get_train_test_data(path):
    try:
        df_train = pd.read_csv(path + '/train.txt', sep='\t')
        df_test = pd.read_csv(path + '/test.txt', sep='\t')
        return df_train, df_test
    except:
        print('error reading: %s' % path + 'train/test.txt')
        return None, None


def get_train_test_nll(df_train, df_test):
    xtr = df_train['epoch']
    ytr = df_train['NLL']
    xts = df_test['epoch']
    yts = df_test['NLL']
    nll_g = df_test['NLL_G']
    nll_nlf = df_test['NLL_SDN']
    return xtr, ytr, xts, yts, nll_g, nll_nlf


def plot_nll(data_path, plot_dict):
    subdir = plot_dict['folders'][0]['folder']
    fpath = os.path.join(data_path, subdir)
    df_train, df_test = get_train_test_data(fpath)
    if df_train is None:
        return
    xtr, ytr, xts, yts, nll_g, nll_nlf = get_train_test_nll(df_train, df_test)

    fig = plt.figure(figsize=plot_dict['figsize'])
    plt.plot(xtr, ytr, label='Train')
    plt.plot(xts, yts, label='Test')
    plt.plot(xts, nll_g, label='Gaussian')
    plt.plot(xts, nll_nlf, label='Camera NLF')
    if plot_dict['xlims'] is not None:
        plt.xlim(plot_dict['xlims'])
    if plot_dict['ylims'] is not None:
        plt.ylim(plot_dict['ylims'])
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    plt.xlabel('Epoch')
    plt.ylabel('NLL')
    plt.title(plot_dict['folders'][0]['folder'])
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(fpath, 'nll.png'))
    # plt.show()


def plot_nll_multi(data_path, plot_dict, clrs):
    n_folders = len(plot_dict['folders'])
    xtr = [None] * n_folders
    ytr = [None] * n_folders
    xts = [None] * n_folders
    yts = [None] * n_folders
    xg = nll_g = nll_nlf = None
    fpath = None
    maxlen = 0
    for i in range(n_folders):
        subdir = plot_dict['folders'][i]['folder']
        fpath = os.path.join(data_path, subdir)
        df_train, df_test = get_train_test_data(fpath)
        if df_train is None:
            continue
        xtr[i], ytr[i], xts[i], yts[i], gg, hh = get_train_test_nll(df_train, df_test)
        if maxlen < len(gg):
            maxlen = len(gg)
            xg = xts[i]
            nll_g = gg
            nll_nlf = hh
        # normalize per Pixel
        n_dims = 64 * 64
        ytr[i] /= n_dims
        yts[i] /= n_dims

    fig = plt.figure(figsize=np.asarray(plot_dict['figsize']) * plot_dict['figscale'])
    plt.rcParams.update({'font.size': plot_dict['fsz']})

    if 'skip_postfix' in plot_dict.keys() and plot_dict['skip_postfix']:
        postfix = ['', '']
    else:
        postfix = [' - Train', ' - Test']

    coff = np.asarray(range(n_folders))
    if 'clr_offset' in plot_dict.keys():
        coff = plot_dict['clr_offset']

    # best NLL
    bnll = []

    if not('skip_base' in plot_dict.keys() and plot_dict['skip_base']):
        plt.plot(xg, nll_g / n_dims, label='Gaussian - Test', linestyle='-.', color=clrs[0])
        plt.plot(xg, nll_nlf / n_dims, label='Camera NLF - Test', linestyle='-.', color=clrs[1])
    bnll.append(np.min(nll_g / n_dims))
    bnll.append(np.min(nll_nlf / n_dims))

    for i in range(n_folders):
        if xtr[i] is None:
            continue

        if 'movavg' in plot_dict.keys() and plot_dict['movavg']:
            ytr[i] = moving_average(np.asarray(ytr[i]))
            yts[i] = moving_average(np.asarray(yts[i]))

        if plot_dict['train_or_test'] == 'both' or plot_dict['train_or_test'] == 'train':
            plt.plot(xtr[i], ytr[i], label=plot_dict['folders'][i]['legend'] + postfix[0], color=clrs[2+coff[i]], linestyle='--')
        if plot_dict['train_or_test'] == 'both' or plot_dict['train_or_test'] == 'test':
            plt.plot(xts[i], yts[i], label=plot_dict['folders'][i]['legend'] + postfix[1], color=clrs[2+coff[i]], linestyle='-')

        bnll.append(np.min(yts[i]))

    lastnll = bnll[-1]
    bnll.append(np.absolute(bnll[0]) - np.absolute(lastnll))
    bnll.append(np.absolute(bnll[1]) - np.absolute(lastnll))
    bnll.append(bnll[-2] / np.absolute(bnll[0]))
    bnll.append(bnll[-2] / np.absolute(bnll[1]))
    np.savetxt(os.path.join(fpath, 'best_nll.txt'), bnll)

    if plot_dict['xlims'] is not None:
        plt.xlim(plot_dict['xlims'])
    if plot_dict['ylims'] is not None:
        plt.ylim(plot_dict['ylims'])
    if 'ystep' in plot_dict.keys():
        plt.yticks(np.arange(plot_dict['ylims'][0], plot_dict['ylims'][1], step=plot_dict['ystep']))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    plt.xlabel('Epoch')
    plt.ylabel(r'$NLL$ (per dimension)')
    if plot_dict['title'] is not None:
        plt.title(plot_dict['title'])
    plt.legend(loc='best', bbox_to_anchor=None, prop={'size': 14}, ncol=1, fancybox=False, shadow=False)
    plt.tight_layout()
    fig.savefig(os.path.join(fpath, '..', 'nll_' + plot_dict['fig_fn'] + '.png'))
    fig.savefig(os.path.join(fpath, '..', 'nll_' + plot_dict['fig_fn'] + '.pdf'), bbox_inches='tight')
    # plt.show()
