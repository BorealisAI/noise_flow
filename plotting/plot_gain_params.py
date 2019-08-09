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


def load_cam_iso_nlf():
    cin = pd.read_csv('cam_iso_nlf_all.txt')
    cin = cin.drop_duplicates()
    cin = cin.set_index('cam_iso', drop=False)
    return cin


def get_cam_beta1_profiles(cam_iso_nlf):
    cam_ids = ['IP', 'GP', 'S6', 'N6', 'G4']
    n_cam = len(cam_ids)
    cam_beta1_all = []
    dbeta1 = np.ndarray([n_cam])
    for i in range(n_cam):
        cam_beta1_all.append([])
    for i in range(cam_iso_nlf.shape[0]):
        row = cam_iso_nlf.iloc[i]
        cid = row['cam_iso'][:2]
        cidx = cam_ids.index(cid)
        iso = float(row['cam_iso'][3:])
        cam_beta1_all[cidx].append([iso, row['beta1']])
    for i in range(n_cam):
        dbeta1[i] = (np.asarray(cam_beta1_all[i][1][1]) - np.asarray(cam_beta1_all[i][0][1])) / (400 - 100)
        # dbeta1[i] = np.asarray(cam_beta1_all[i][0][1])
    return cam_beta1_all, dbeta1


def get_gain_params_data(path):
    try:
        df_gain_params = pd.read_csv(path + '/vars.txt', sep='\t')
        return df_gain_params
    except Exception as ex:
        print(str(ex))
        print('error reading: %s' % path + 'vars.txt')
        return None

def get_gain_params(df_gain_params, gain_param_name):
    iso_vals = [100, 400, 800, 1600, 3200]
    xtr = df_gain_params['epoch']
    ytr = []
    for i in range(5):
        if gain_param_name == 'g':
            ytr.append(df_gain_params['g' + str(iso_vals[i])])
        else:
            ytr.append(df_gain_params['gain_params' + str(i)])
    return xtr, ytr


def get_cam_gain_params(df_params):
    xtr = df_params['epoch']
    ytr = []
    for i in range(5):
        ytr.append(df_params['cam_params2' + str(i)])
    return xtr, ytr


def plot_gain_params(data_path, plot_dict):
    subdir = plot_dict['folders'][0]['folder']
    fpath = os.path.join(data_path, subdir)
    df_gain_params = get_gain_params_data(fpath)
    xtr, ytr = get_gain_params(df_gain_params, plot_dict['gain_param_name'])
    iso_vals = [100, 400, 800, 1600, 3200]

    plt.rcParams.update({'font.size': 18})

    fig = plt.figure()
    for i in range(len(iso_vals)):
        plt.plot(xtr, ytr[i], label='g' + str(iso_vals[i]))
    if plot_dict['xlims'] is not None:
        plt.xlim(plot_dict['xlims'])
    if plot_dict['ylims'] is not None:
        plt.ylim(plot_dict['ylims'])
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    plt.xlabel('Epoch')
    plt.ylabel('Gain parameters')
    plt.title(plot_dict['folders'][0]['folder'])
    plt.legend(loc='best', bbox_to_anchor=None, prop={'size': 16}, ncol=2, fancybox=False, shadow=False)
    plt.tight_layout()
    fig.savefig(os.path.join(fpath, 'gain_params.png'))
    fig.savefig(os.path.join(fpath, 'gain_params.pdf'))
    # plt.show()

    fig = plt.figure()
    for i in range(len(iso_vals)):
        if 'skip_iso' in plot_dict.keys() and iso_vals[i] in plot_dict['skip_iso']:
            print('skipping iso')
            continue
        plt.plot(xtr, ytr[i] - np.log(iso_vals[i]), label=r'$g_{' + str(iso_vals[i]) + '}$')
    if plot_dict['xlims'] is not None:
        plt.xlim(plot_dict['xlims'])
    if plot_dict['ylims'] is not None:
        plt.ylim(plot_dict['ylims'])
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    plt.xlabel('Epoch')
    plt.ylabel('Gain parameters (log scale)')
    if plot_dict['title'] is not None:
        plt.title(plot_dict['title'])
    plt.legend(loc='upper right', bbox_to_anchor=[1, .8], prop={'size': 16}, ncol=2, fancybox=False, shadow=False)
    plt.tight_layout()
    fig.savefig(os.path.join(fpath, 'gain_params_exp_iso.png'))
    fig.savefig(os.path.join(fpath, 'gain_params_exp_iso.pdf'))
    # plt.show()

    fig = plt.figure()
    for i in range(len(iso_vals)):
        if 'skip_iso' in plot_dict.keys() and iso_vals[i] in plot_dict['skip_iso']:
            print('skipping iso')
            continue
        plt.plot(xtr, np.exp(ytr[i]) * iso_vals[i], label=r'$g_{' + str(iso_vals[i]) + '}$')
    if plot_dict['xlims'] is not None:
        plt.xlim(plot_dict['xlims'])
    if plot_dict['ylims'] is not None:
        plt.ylim(plot_dict['ylims'])
    plt.ylim([0, .5])
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    plt.xlabel('Epoch')
    plt.ylabel('Gain parameters (log scale)')
    if plot_dict['title'] is not None:
        plt.title(plot_dict['title'])
    plt.legend(loc='best', bbox_to_anchor=None, prop={'size': 16}, ncol=2, fancybox=False, shadow=False)
    plt.tight_layout()
    fig.savefig(os.path.join(fpath, 'gain_params_exp_iso_.png'))
    fig.savefig(os.path.join(fpath, 'gain_params_exp_iso_.pdf'))
    # plt.show()


def plot_cam_params(data_path, plot_dict):
    subdir = plot_dict['folders'][0]['folder']
    fpath = os.path.join(data_path, subdir)
    df_params = get_gain_params_data(fpath)
    xtr, ytr = get_cam_gain_params(df_params)
    cam_vals = ['iPhone 7', 'Pixel', 'Nexus 6', 'Galaxy S6', 'G4']
    cam_ids = ['IP', 'GP', 'N6', 'S6', 'G4']

    plt.rcParams.update({'font.size': 20})

    fig = plt.figure()
    for i in range(len(cam_vals)):
        plt.plot(xtr, np.exp(ytr[i]), label=cam_vals[i])
    if plot_dict['xlims'] is not None:
        plt.xlim(plot_dict['xlims'])
    if plot_dict['ylims'] is not None:
        plt.ylim(plot_dict['ylims'])
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    plt.xlabel('Epoch')
    plt.ylabel(r'Gain weights $\psi$')
    if plot_dict['title'] is not None:
        plt.title(plot_dict['title'])
    # if plot_dict['xlims'] is not None:
    #     plt.xlim(plot_dict['xlims'])
    # if plot_dict['ylims'] is not None:
    #     plt.ylim(plot_dict['ylims'])
    plt.legend(loc='upper right', bbox_to_anchor=[1, .38], prop={'size': 14}, ncol=2, fancybox=False, shadow=False)
    plt.tight_layout()
    fig.savefig(os.path.join(fpath, 'cam_params.png'))
    fig.savefig(os.path.join(fpath, 'cam_params.pdf'))
    # plt.show()

    # camera params vs beta_1 params
    cam_iso_nlf = load_cam_iso_nlf()
    cam_beta1 = np.ndarray([5])
    cam_params = np.ndarray([5])
    for i, cam in enumerate(cam_ids):
        cam_beta1[i] = cam_iso_nlf.loc[cam + '_%05d' % 800]['beta1']
        cam_params[i] = ytr[i][2500]

    # fig = plt.figure()
    # plt.plot(cam_beta1, cam_params)
    # # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    # plt.xlabel(r'Camera \beta_1')
    # plt.ylabel('Camera parameter')
    # if plot_dict['xlims'] is not None:
    #     plt.xlim(plot_dict['xlims'])
    # if plot_dict['ylims'] is not None:
    #     plt.ylim(plot_dict['ylims'])
    # # plt.legend(loc='best', bbox_to_anchor=None, prop={'size': 12}, ncol=2, fancybox=False, shadow=False)
    # plt.tight_layout()
    # fig.savefig(os.path.join(fpath, 'cam_params_vs_beta1.png'))
    # fig.savefig(os.path.join(fpath, 'cam_params_vs_beta1.pdf'))
    # # plt.show()

    cam_beta1_all, dbeta1 = get_cam_beta1_profiles(cam_iso_nlf)
    fig = plt.figure()
    for i in range(len(cam_beta1_all)):
        one_cam_prof = np.asarray(cam_beta1_all[i])
        plt.plot(one_cam_prof[:, 0], one_cam_prof[:, 1], marker='+', label=cam_vals[i])
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    plt.ylabel(r'Camera $\beta_1$')
    plt.xlabel('Camera ISO')
    plt.xticks([100, 400, 800, 1600, 3200], [100, 400, 800, 1600, 3200], fontsize=14)  # rotation=30
    # if plot_dict['xlims'] is not None:
    #     plt.xlim(plot_dict['xlims'])
    # if plot_dict['ylims'] is not None:
    #     plt.ylim(plot_dict['ylims'])
    plt.legend(loc='best', bbox_to_anchor=None, prop={'size': 14}, ncol=2, fancybox=False, shadow=False)
    plt.tight_layout()
    fig.savefig(os.path.join(fpath, 'cam_beta1_all.png'))
    fig.savefig(os.path.join(fpath, 'cam_beta1_all.pdf'))
    # plt.show()

    # fig = plt.figure()
    # plt.plot(dbeta1, cam_params)
    # # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    # plt.xlabel(r'Estimated camera gain')
    # plt.ylabel('Camera parameter')
    # if plot_dict['xlims'] is not None:
    #     plt.xlim(plot_dict['xlims'])
    # if plot_dict['ylims'] is not None:
    #     plt.ylim(plot_dict['ylims'])
    # # plt.legend(loc='best', bbox_to_anchor=None, prop={'size': 12}, ncol=2, fancybox=False, shadow=False)
    # plt.tight_layout()
    # fig.savefig(os.path.join(fpath, 'cam_params_vs_beta1.png'))
    # fig.savefig(os.path.join(fpath, 'cam_params_vs_beta1.pdf'))
    # # plt.show()
