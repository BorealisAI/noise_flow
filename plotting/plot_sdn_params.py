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


def get_sdn_params_data(path):
    df_sdn_params = pd.read_csv(path + '/vars.txt', sep='\t')
    return df_sdn_params


def get_sdn_params(df_sdn_params, sdn_param_name):
    if sdn_param_name is None:
        sdn_param_name = 'beta'
    xtr = df_sdn_params['epoch']
    ytr = []
    for i in range(1, 3):
        ytr.append(df_sdn_params['beta' + str(i)])
    return xtr, ytr


def plot_sdn_params(data_path, plot_dict):
    subdir = plot_dict['folders'][0]['folder']
    fpath = os.path.join(data_path, subdir)
    df_sdn_params = get_sdn_params_data(fpath)
    xtr, ytr = get_sdn_params(df_sdn_params, None)

    plt.rcParams.update({'font.size': 18})

    fig = plt.figure()
    for i in range(2):
        plt.plot(xtr, ytr[i], label=r'$\beta_' + str(i + 1) + '$')
    if plot_dict['xlims'] is not None:
        plt.xlim(plot_dict['xlims'])
    if plot_dict['ylims'] is not None:
        plt.ylim(plot_dict['ylims'])
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    plt.xlabel('Epoch')
    plt.ylabel('Signal-dependent parameters')
    if plot_dict['title'] is not None:
        plt.title(plot_dict['title'])
    plt.legend(prop={'size': 16}, ncol=4, fancybox=False, shadow=False)
    plt.tight_layout()
    fig.savefig(os.path.join(fpath, 'sdn_params.png'))
    fig.savefig(os.path.join(fpath, 'sdn_params.pdf'))
    # plt.show()

    fig = plt.figure()
    for i in range(2):
        ytr_ = moving_average(np.exp(plot_dict['c'] * np.asarray(ytr[i])), n=10)
        plt.plot(xtr, ytr_, label=r'$\exp(\beta_' + str(i + 1) + ')$')
    if plot_dict['xlims'] is not None:
        plt.xlim(plot_dict['xlims'])
    if plot_dict['ylims'] is not None:
        plt.ylim(plot_dict['ylims'])
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    plt.xlabel('Epoch')
    plt.ylabel('Signal-dependent parameters')
    if plot_dict['title'] is not None:
        plt.title(plot_dict['title'])
    plt.legend(loc='best', bbox_to_anchor=None, prop={'size': 16}, ncol=1, fancybox=False, shadow=False)
    plt.tight_layout()
    fig.savefig(os.path.join(fpath, 'sdn_params_exp.png'))
    fig.savefig(os.path.join(fpath, 'sdn_params_exp.pdf'))
    # plt.show()
