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

# plt.interactive(False)

# plot_range_x = [-10, 2000]

# date = 'Oct29'
# from matplotlib.ticker import FormatStrFormatter
# from plotting.broken_ax import broken_ax
from broken_ax import broken_ax


def get_train_test_results(path):
    df_train = pd.read_csv(path + '/train.txt', sep='\t')
    df_test = pd.read_csv(path + '/test.txt', sep='\t')
    return df_train, df_test


data_path = 'experiments/sidd/'
# data_path = 'experiments/sidd_keep/'
smooth = 1
sc = 1

dicts = [
    {  # 0
        'title': '',
        'folders': [
            # {'folder': '190122_01_unc_1_4_32_', 'legend': '4xUnc.'},
            # {'folder': '190124_UncSdn_init',         'legend': 'SdnGain+4xUnc.'},
            # {'folder': '190129/SdnGain2Uncx4',         'legend': 'SdnGain2+4xUnc'},
            # {'folder': '190130/UncSdnUncGainUnc_',         'legend': 'UncSdnUncGainUnc_'},
            # {'folder': '190130/UncSdnUncGainUnc_s',         'legend': 'UncSdnUncGainUnc_s'},
            # {'folder': 'UncSdnUncGainUnc_s_TEST',         'legend': 'UncSdnUncGainUnc_s_TEST'}
            # {'folder': 'U', 'legend': 'U'},
            # {'folder': 'U_', 'legend': 'U_'},
            # {'folder': 'U_1', 'legend': 'U_1'},
            {'folder': 'U4', 'legend': 'U4'},
            {'folder': 'U16', 'legend': 'U16'},
            # {'folder': 'SDN', 'legend': 'S'},
            {'folder': 'S', 'legend': 'S'},
            # {'folder': 'SG', 'legend': 'SG'},
            # {'folder': 'SG3', 'legend': 'SG3'},
            # {'folder': 'S1G3_IP', 'legend': 'S1G3_IP'},
            # {'folder': 'G', 'legend': 'G'},
            # {'folder': 'SU', 'legend': 'SU'},
            # {'folder': 'SUU', 'legend': 'SUU'},
            # {'folder': 'SUGU', 'legend': 'SUGU'},
            # {'folder': 'SUG2U', 'legend': 'SUG2U'},
            # {'folder': 'SUG3U', 'legend': 'SUG3U'},
            # {'folder': 'SUG2U_c1e-2', 'legend': 'SUG2U_c1e-2'},
            # {'folder': 'SxUGxU', 'legend': 'SxUGxU'},
            # {'folder': 'SU4GU4', 'legend': 'SU4GU4'},
            # {'folder': 'U4SU4GU4', 'legend': 'U4SU4GU4'},
            # {'folder': 'SU4GU4_1e-5', 'legend': 'SU4GU4_1e-5'},
            # {'folder': 'U4SU4GU4_1e-5', 'legend': 'U4SU4GU4_1e-5'},
            # {'folder': 'U4SU4GxU4_1e-5', 'legend': 'U4SU4GxU4_1e-5'},
            # {'folder': 'U4SU4GxU4_1', 'legend': 'U4SU4GxU4_1'},
            # {'folder': 'U4SU4GxU4_2', 'legend': 'U4SU4GxU4_2'},
            # {'folder': 'SUGU_FC', 'legend': 'SUGU_FC3'},
            # {'folder': 'SUGU_FC5', 'legend': 'SUGU_FC5'},
            # {'folder': 'SG3_IP_c1e-2', 'legend': 'SG3_IP_c1e-2'},
            # {'folder': 'SG2_IP_c1e-2', 'legend': 'SG2_IP_c1e-2'},
            # {'folder': 'SG2_IP_init0', 'legend': 'SG2_IP_init0'},
            {'folder': 'SG2_IP_init-0', 'legend': 'SG2_IP_init-0'},
            # {'folder': 'SG2_IP_init5', 'legend': 'SG2_IP_init5'},
            # {'folder': 'SG2_IP800_c1e-2', 'legend': 'SG2_IP800_c1e-2'},
            # {'folder': 'S2_IP_init-5', 'legend': 'S2_IP_init-5'},
            {'folder': 'S3_IP_init-5', 'legend': 'S3_IP_init-5'},
            {'folder': 'S4G4', 'legend': 'S4G4'},
            {'folder': 'S4G4_IP', 'legend': 'S4G4_IP'},
            {'folder': 'SUx3GUx3', 'legend': 'SUx3GUx3'},
        ],
        'figsize': [7.4 * sc, 7.8 * sc],  # default [6.4, 4.8]
        'ylim1': None,
        'ylim2': None,#-.75e4,
        'ylims_brk': [[-2e4, -1.5e4], [-1.5e4, -1.1e4]],
        'xlabel': 'Epoch',
        'ylabel': 'NLL',
        'yscale': 'symlog',
        'legend_loc': 'upper center',
        'bbx': [.5, 1.7],
        'adjust': None,
        'fig_fn': 'models'
    },
    {  # 1
        'title': 'Gain parameters (c = 1)',
        'folders': [
            {'folder': 'SUG2U', 'legend': 'SUG2U'},
        ],
        'figsize': [6.4, 4.8],  # default [6.4, 4.8]
        'ylim1': None,
        'ylim2': None,#-.75e4,
        'xlabel': 'Epoch',
        'ylabel': 'NLL',
        'yscale': 'symlog',
        'legend_loc': 'best',
        'bbx': None,
        'adjust': None,
        'fig_fn': 'gian_params_1',
        'c': 1
    },
    {  # 2
        'title': 'Gain parameters (c = 1e-2)',
        'folders': [
            {'folder': 'SUG2U_c1e-2', 'legend': 'SUG2U_c1e-2'},
        ],
        'figsize': [6.4, 4.8],  # default [6.4, 4.8]
        'ylim1': None,
        'ylim2': None,#-.75e4,
        'xlabel': 'Epoch',
        'ylabel': 'NLL',
        'yscale': 'symlog',
        'legend_loc': 'best',
        'bbx': None,
        'adjust': None,
        'fig_fn': 'gian_params_2',
        'c': 1e-2
    },
    {  # 3
        'title': 'Gain parameters',
        'folders': [
            {'folder': 'SG3', 'legend': 'SG3'},
        ],
        'figsize': [6.4, 4.8],  # default [6.4, 4.8]
        'ylim1': None,
        'ylim2': None,#-.75e4,
        'xlabel': 'Epoch',
        'ylabel': 'NLL',
        'yscale': 'symlog',
        'legend_loc': 'best',
        'bbx': None,
        'adjust': None,
        'fig_fn': 'gian_params_SG3',
        'c': 1
    },
    {  # 4
        'title': 'Gain parameters',
        'folders': [
            {'folder': 'S1G3', 'legend': 'S1G3'},
        ],
        'figsize': [6.4, 4.8],  # default [6.4, 4.8]
        'ylim1': None,
        'ylim2': None,#-.75e4,
        'xlabel': 'Epoch',
        'ylabel': 'NLL',
        'yscale': 'symlog',
        'legend_loc': 'best',
        'bbx': None,
        'adjust': None,
        'fig_fn': 'gian_params_S1G3',
        'c': 1
    },
    {  # 5
        'title': 'SG3_IP_c1e-2',
        'folders': [
            {'folder': 'SG3_IP_c1e-2', 'legend': 'SG3_IP_c1e-2'},
        ],
        'figsize': [6.4, 4.8],  # default [6.4, 4.8]
        'ylim1': None,
        'ylim2': None,#-.75e4,
        'xlabel': 'Epoch',
        'ylabel': 'NLL',
        'yscale': 'symlog',
        'legend_loc': 'best',
        'bbx': None,
        'adjust': None,
        'fig_fn': 'SG3_IP_c1e-2',
        'c': 1
    },
    {  # 6
        'title': 'Gain parameters SG3_IP_c1e-2',
        'folders': [
            {'folder': 'SG3_IP_c1e-2', 'legend': 'SG3_IP_c1e-2'},
        ],
        'figsize': [6.4, 4.8],  # default [6.4, 4.8]
        'ylim1': None,
        'ylim2': None,#-.75e4,
        'xlabel': 'Epoch',
        'ylabel': 'NLL',
        'yscale': 'symlog',
        'legend_loc': 'best',
        'bbx': None,
        'adjust': None,
        'fig_fn': 'gian_params_SG3_IP_c1e-2',
        'c': 1e-2
    },
    {  # 7
        'title': 'Sampling NLL SG3_IP_c1e-2',
        'folders': [
            {'folder': 'SG3_IP_c1e-2', 'legend': 'SG3_IP_c1e-2'},
        ],
        'figsize': [6.4, 4.8],  # default [6.4, 4.8]
        'ylim1': None,
        'ylim2': None,#-.75e4,
        'xlabel': 'Epoch',
        'ylabel': 'Sampling NLL',
        'yscale': 'symlog',
        'legend_loc': 'best',
        'bbx': None,
        'adjust': None,
        'fig_fn': 'sample_nll_SG3_IP_c1e-2',
        'c': 1e-2
    },
    {  # 8
        'title': 'Gain parameters S3_IP_init-5 \n gain_scale = ...',
        'folders': [
            {'folder': 'S3_IP_init-5', 'legend': 'S3_IP_init-5'},
        ],
        'figsize': [6.4, 4.8],  # default [6.4, 4.8]
        'ylim1': None,
        'ylim2': None,#-.75e4,
        'xlabel': 'Epoch',
        'ylabel': 'Gain parameters',
        'yscale': 'symlog',
        'legend_loc': 'best',
        'bbx': None,
        'adjust': None,
        'fig_fn': 'gian_params_S3_IP_init-5',
        'c': 1e-1
    },
{  # 9
        'title': 'Sampling NLL SG2_IP_c1e-2',
        'folders': [
            {'folder': 'SG2_IP_c1e-2', 'legend': 'SG2_IP_c1e-2'},
        ],
        'figsize': [6.4, 4.8],  # default [6.4, 4.8]
        'ylim1': None,
        'ylim2': None,#-.75e4,
        'xlabel': 'Epoch',
        'ylabel': 'Sampling NLL',
        'yscale': 'symlog',
        'legend_loc': 'best',
        'bbx': None,
        'adjust': None,
        'fig_fn': 'sample_nll_SG2_IP_c1e-2',
        'c': 1e-2
    },
{  # 10
        'title': 'Sampling KLD SG2_IP_init-0',
        'folders': [
            {'folder': 'SG2_IP_init-0', 'legend': 'SG2_IP_init-0'},
        ],
        'figsize': [6.4, 4.8],  # default [6.4, 4.8]
        'ylim1': None,
        'ylim2': None,#-.75e4,
        'xlabel': 'Epoch',
        'ylabel': 'Sampling KLD',
        'yscale': 'symlog',
        'legend_loc': 'best',
        'bbx': None,
        'adjust': None,
        'fig_fn': 'sample_kld_SG2_IP_init-0',
        'c': 1
    },
{  # 11
        'title': 'SDN parameters SG2_IP_init-0 \n sdn_scale = sqrt(sigmoid(b1) * y + sigmoid(b2))',
        'folders': [
            {'folder': 'SG2_IP_init-0', 'legend': 'SG2_IP_init-0'},
        ],
        'figsize': [7.4, 4.8],  # default [6.4, 4.8]
        'ylim1': None,
        'ylim2': None,#-.75e4,
        'xlabel': 'Epoch',
        'ylabel': 'SDN parameters',
        'yscale': 'symlog',
        'legend_loc': 'best',
        'bbx': None,
        'adjust': None,
        'fig_fn': 'sdn_params_SG2_IP_init-0',
        'c': 1
    },
{  # 12
        'title': 'Gain parameters SG2_IP_init-0 \n gain_scale = exp(c * g) * iso',
        'folders': [
            {'folder': 'SG2_IP_init-0', 'legend': 'SG2_IP_init-0'},
        ],
        'figsize': [6.4, 4.8],  # default [6.4, 4.8]
        'ylim1': None,
        'ylim2': None,#-.75e4,
        'xlabel': 'Epoch',
        'ylabel': 'Gain scale [exp(c * g) * iso]',
        'yscale': 'symlog',
        'legend_loc': 'best',
        'bbx': None,
        'adjust': None,
        'fig_fn': 'gian_params_SG2_IP_init-0',
        'c': 1e-1
    },
{  # 13
        'title': 'SDN parameters S3_IP_init-5 \n sdn_scale = ...',
        'folders': [
            {'folder': 'S3_IP_init-5', 'legend': 'S3_IP_init-5'},
        ],
        'figsize': [7.4, 4.8],  # default [6.4, 4.8]
        'ylim1': None,
        'ylim2': None,#-.75e4,
        'xlabel': 'Epoch',
        'ylabel': 'SDN parameters',
        'yscale': 'symlog',
        'legend_loc': 'best',
        'bbx': None,
        'adjust': None,
        'fig_fn': 'sdn_params_S3_IP_init-5',
        'c': 1
    },
{  # 14
        'title': 'Sampling KLD S3_IP_init-5',
        'folders': [
            {'folder': 'S3_IP_init-5', 'legend': 'S3_IP_init-5'},
        ],
        'figsize': [6.4, 4.8],  # default [6.4, 4.8]
        'ylim1': None,
        'ylim2': None,#-.75e4,
        'xlabel': 'Epoch',
        'ylabel': 'Sampling KLD',
        'yscale': 'symlog',
        'legend_loc': 'best',
        'bbx': None,
        'adjust': None,
        'fig_fn': 'sample_kld_S3_IP_init-5',
        'c': 1
    },
{  # 15
        'title': 'Gain parameters S4G4 \n gain_scale = ...',
        'folders': [
            {'folder': 'S4G4', 'legend': 'S4G4'},
        ],
        'figsize': [6.4, 4.8],  # default [6.4, 4.8]
        'ylim1': None,
        'ylim2': None,#-.75e4,
        'xlabel': 'Epoch',
        'ylabel': 'Gain params',
        'yscale': 'symlog',
        'legend_loc': 'best',
        'bbx': None,
        'adjust': None,
        'fig_fn': 'gian_params_S4G4',
        'c': 1e-1
    },
{  # 16
        'title': 'SDN parameters S4G4 \n sdn_scale = ...',
        'folders': [
            {'folder': 'S4G4', 'legend': 'S4G4'},
        ],
        'figsize': [7.4, 4.8],  # default [6.4, 4.8]
        'ylim1': None,
        'ylim2': None,#-.75e4,
        'xlabel': 'Epoch',
        'ylabel': 'SDN parameters',
        'yscale': 'symlog',
        'legend_loc': 'best',
        'bbx': None,
        'adjust': None,
        'fig_fn': 'sdn_params_S4G4',
        'c': 1
    },
{  # 17
        'title': 'Sampling KLD S4G4',
        'folders': [
            {'folder': 'S4G4', 'legend': 'S4G4'},
        ],
        'figsize': [6.4, 4.8],  # default [6.4, 4.8]
        'ylim1': None,
        'ylim2': None,#-.75e4,
        'xlabel': 'Epoch',
        'ylabel': 'Sampling KLD',
        'yscale': 'symlog',
        'legend_loc': 'best',
        'bbx': None,
        'adjust': None,
        'fig_fn': 'sample_kld_S4G4',
        'c': 1
    },
{  # 18
        'title': 'Gain parameters SUx3GUx3 \n gain_scale = ...',
        'folders': [
            {'folder': 'SUx3GUx3', 'legend': 'SUx3GUx3'},
        ],
        'figsize': [6.4, 4.8],  # default [6.4, 4.8]
        'ylim1': None,
        'ylim2': None,#-.75e4,
        'xlabel': 'Epoch',
        'ylabel': 'Gain params',
        'yscale': 'symlog',
        'legend_loc': 'best',
        'bbx': None,
        'adjust': None,
        'fig_fn': 'gian_params_SUx3GUx3',
        'c': 1e-1
    },
{  # 19
        'title': 'SDN parameters SUx3GUx3 \n sdn_scale = ...',
        'folders': [
            {'folder': 'SUx3GUx3', 'legend': 'SUx3GUx3'},
        ],
        'figsize': [7.4, 4.8],  # default [6.4, 4.8]
        'ylim1': None,
        'ylim2': None,#-.75e4,
        'xlabel': 'Epoch',
        'ylabel': 'SDN parameters',
        'yscale': 'symlog',
        'legend_loc': 'best',
        'bbx': None,
        'adjust': None,
        'fig_fn': 'sdn_params_SUx3GUx3',
        'c': 1
    },
{  # 20
        'title': 'Sampling KLD SUx3GUx3',
        'folders': [
            {'folder': 'SUx3GUx3', 'legend': 'SUx3GUx3'},
        ],
        'figsize': [6.4, 4.8],  # default [6.4, 4.8]
        'ylim1': None,
        'ylim2': None,#-.75e4,
        'xlabel': 'Epoch',
        'ylabel': 'Sampling KLD',
        'yscale': 'symlog',
        'legend_loc': 'best',
        'bbx': None,
        'adjust': None,
        'fig_fn': 'sample_kld_SUx3GUx3',
        'c': 1
    },
{  # 21
        'title': 'Gain parameters S4G4_IP \n gain_scale = ...',
        'folders': [
            {'folder': 'S4G4_IP', 'legend': 'S4G4_IP'},
        ],
        'figsize': [6.4, 4.8],  # default [6.4, 4.8]
        'ylim1': None,
        'ylim2': None,#-.75e4,
        'xlabel': 'Epoch',
        'ylabel': 'Gain params',
        'yscale': 'symlog',
        'legend_loc': 'best',
        'bbx': None,
        'adjust': None,
        'fig_fn': 'gian_params_S4G4_IP',
        'c': 1e-1
    },
{  # 22
        'title': 'SDN parameters S4G4_IP \n sdn_scale = ...',
        'folders': [
            {'folder': 'S4G4_IP', 'legend': 'S4G4_IP'},
        ],
        'figsize': [7.4, 4.8],  # default [6.4, 4.8]
        'ylim1': None,
        'ylim2': None,#-.75e4,
        'xlabel': 'Epoch',
        'ylabel': 'SDN parameters',
        'yscale': 'symlog',
        'legend_loc': 'best',
        'bbx': None,
        'adjust': None,
        'fig_fn': 'sdn_params_S4G4_IP',
        'c': 1
    },
{  # 23
        'title': 'Sampling KLD S4G4_IP',
        'folders': [
            {'folder': 'S4G4_IP', 'legend': 'S4G4_IP'},
        ],
        'figsize': [6.4, 4.8],  # default [6.4, 4.8]
        'ylim1': None,
        'ylim2': None,#-.75e4,
        'xlabel': 'Epoch',
        'ylabel': 'Sampling KLD',
        'yscale': 'symlog',
        'legend_loc': 'best',
        'bbx': None,
        'adjust': None,
        'fig_fn': 'sample_kld_S4G4_IP',
        'c': 1
    },
    {
        'title': 'SD of base measure',
        'folders': [
            # {'folder': 'U_1', 'legend': 'U_1'},
            {'folder': 'U4', 'legend': 'U4'},
            # {'folder': 'SDN', 'legend': 'S'},
            {'folder': 'S', 'legend': 'S'},
            {'folder': 'SG', 'legend': 'SG'},
            {'folder': 'G', 'legend': 'G'},
            {'folder': 'SU', 'legend': 'SU'},
            {'folder': 'SUU', 'legend': 'SUU'},
            {'folder': 'SUGU', 'legend': 'SUGU'},
            {'folder': 'SU4GU4', 'legend': 'SU4GU4'},
            {'folder': 'U4SU4GU4', 'legend': 'U4SU4GU4'},
            # {'folder': 'SUGU_FC', 'legend': 'SUGU_FC3'},
            # {'folder': 'SUGU_FC5', 'legend': 'SUGU_FC5'},
        ],
        'figsize': [7.4, 4.8],  # default [6.4, 4.8]
        'ylim1': .6,
        'ylim2': 1.3,
        'yscale': 'symlog',
        'legend_loc': 'upper center',
        'bbx': [.5, 1.3],
        'adjust': None,
        'fig_fn': 'StdDev'
    },
    {
        'title': '',
        'folders': [
            {'folder': 'SG', 'legend': 'SG'},
            {'folder': 'SUGU', 'legend': 'SUGU'},
            {'folder': 'SUGU_FC', 'legend': 'SUGU_FC3'},
            {'folder': 'SUGU_FC5', 'legend': 'SUGU_FC5'},
        ],
        'figsize': [7.4, 4.8],  # default [6.4, 4.8]
        'ylim1': -3.68e4,
        'ylim2': -3.60e4,
        'yscale': 'symlog',
        'legend_loc': 'upper left',
        'bbx': [1.01, 1.01],
        'adjust': None,
        'fig_fn': 'models2'
    },
{
        'title': '',
        'folders': [
            {'folder': 'G', 'legend': 'G'},
        ],
        'figsize': [7.4, 4.8],  # default [6.4, 4.8]
        'ylim1': None,
        'ylim2': None,
        'yscale': 'symlog',
        'legend_loc': 'best',
        'bbx': None,
        'adjust': None,
        'fig_fn': 'G'
    },
    {
        'title': 'Squeeze factor 1 vs 2\nCam: IP, Model: cY, L:1, D:1, W:32',
        'folders': [
            {'folder': '190124_sqz_1vs2/cYG_sqz1',      'legend': 'cYG_sqz1'},
            {'folder': '190124_sqz_1vs2/cYG_sqz2', 'legend': 'cYG_sqz2'}
        ],
        'figsize': [6.4, 4.8],  # default
        'ylim1': None,
        'ylim2': None,#-.5e4,
        'legend_loc': 'best',
        'bbx': None,
        'adjust': None,
        'fig_fn': 'Squeeze1vs2'
    },
    {
        'title': 'Effect of Conv2d1x1\nCamera: IP, Model: Unc. L:1, D:1, W:32',
        'folders': [
            {'folder': '190124_PermVsNoperm/cYG_NoPrm', 'legend': 'NoConv2d1x1'},
            {'folder': '190124_PermVsNoperm/cYG_Prm',   'legend': 'Conv2d1x1'}
        ],
        'figsize': [6.4, 4.8],  # default
        'ylim1': None,
        'ylim2': -0.75e4,
        'legend_loc': 'best',
        'bbx': None,
        'adjust': None,
        'fig_fn': 'Conv2d1x1'
    },
    {
        'title': 'Effect of Depth (# AC layers)\nCamera: IP, Model: L:1, D:?, W:32',
        'folders': [
            {'folder': '190124_depths/1', 'legend': 'Depth = 1'},
            {'folder': '190124_depths/2', 'legend': 'Depth = 2'},
            {'folder': '190124_depths/3', 'legend': 'Depth = 3'},
            {'folder': '190124_depths/4', 'legend': 'Depth = 4'},
            {'folder': '190124_depths/5', 'legend': 'Depth = 5'}
        ],
        'figsize': [7.4, 4.8],
        'ylim1': -1.6e4,
        'ylim2': -1.3e4,
        'legend_loc': 'upper left',
        'bbx': [1.01, 1.01],
        'adjust': (None, None, .8),  # left, bottom, right, top, wspace, hspace
        'fig_fn': 'Depths'
    },
    {
        'title': 'Uncond. vs Uncond.+SDN_layer\nCam: IP, Model: L:1, D:1, W:32',
        'folders': [
            {'folder': '190128_unc', 'legend': 'Unc.'},
            {'folder': '190128_sdn', 'legend': 'SDN layer'},
            {'folder': '190128_unc_sdn', 'legend': 'Unc.+SDN layer'},
            {'folder': '190128_cXY', 'legend': 'cXY'},
            {'folder': '190128_cXY_sdn', 'legend': 'cXY+SDN layer'},
            {'folder': '190128_sdn_cY_unc', 'legend': 'SDN+cY+Unc'}
        ],
        'figsize': [6.4, 4.8],
        'ylim1': None,
        'ylim2': -1e4,
        'legend_loc': 'best',
        'bbx': None,
        'adjust': None,  # left, bottom, right, top, wspace, hspace
        'fig_fn': 'Unc_Sdn_IP'
    }
]

# fig_fns = [
#     'UncSdn'
#     , 'Conv2d1x1'
# ]

new_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

# df_train = [None] * len(dicts)
# df_test = [None] * len(dicts)

plt_max_epc = 2000
fsz1 = 11
fsz2 = 14

f = 1.
fig_tt = [None] * len(dicts)

for k in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  13, 14, 15, 16, 17, 18, 19, 20] + [21, 22, 23]:
    # range(len(dicts)):

    if k == 0:
        plt.rcParams.update({'font.size': fsz1 * sc})
    else:
        plt.rcParams.update({'font.size': fsz1})

    if k == 0:
        pass
    else:
        fig_tt[k] = plt.figure(figsize=dicts[k]['figsize'])  # default: [6.4, 4.8]

    ax1 = None
    ax2 = None

    for i in range(len(dicts[k]['folders'])):
        try:
            if k == 99:
                import pdb
                pdb.set_trace()
            print('reading: ' + dicts[k]['folders'][i]['folder'] + ' | ' + dicts[k]['title'])
            df_train = pd.read_csv(os.path.join(data_path, dicts[k]['folders'][i]['folder'], 'train.txt'), sep='\t')
            df_test = pd.read_csv(os.path.join(data_path, dicts[k]['folders'][i]['folder'], 'test.txt'), sep='\t')
            df_vars = None
            df_params = None
            if dicts[k]['title'].__contains__('Gain parameters'):
                df_vars = pd.read_csv(os.path.join(data_path, dicts[k]['folders'][i]['folder'], 'vars.txt'), sep='\t')
            if dicts[k]['title'].__contains__('Sampling'):
                df_sample = pd.read_csv(os.path.join(data_path, dicts[k]['folders'][i]['folder'], 'sample.txt'),
                                        sep='\t')
            if dicts[k]['title'].__contains__('SDN parameters'):
                df_vars = pd.read_csv(os.path.join(data_path, dicts[k]['folders'][i]['folder'], 'vars.txt'), sep='\t')
                # df_params = pd.read_csv(os.path.join(data_path, dicts[k]['folders'][i]['folder'], 'col_vars.txt'),
                #                         sep='\t')
            Ytr = Yts = Yvs = Ysm = None

            if k == 0:
                pass
            elif dicts[k]['title'] == 'SD of base measure':
                Ytr = df_train['sdz']
                Yts = df_test['sdz']
            elif dicts[k]['title'].__contains__('Sampling NLL'):
                Xs = df_sample['epoch'][::2]
                Ytr = df_sample['NLL_G'][::2]
                Yts = df_sample['NLL_SDN'][::2]
                Ysm = df_sample['NLL'][::2]
            elif dicts[k]['title'].__contains__('Sampling KLD'):
                Xs = df_sample['epoch'][::2]
                Ytr = df_sample['KLD_G'][::2]
                Yts = df_sample['KLD_NLF'][::2]
                Ysm = df_sample['KLD_NF'][::2]
            elif dicts[k]['title'].__contains__('SDN parameters'):
                # Xs = df_params['epoch']
                # try:
                #     Yb1 = df_params['b1']
                #     Yb2 = df_params['b2']
                # except:
                Xs = df_vars['epoch']
                try:
                    Yb1 = df_vars['b1']
                    Yb2 = df_vars['b2']
                except:
                    Yb1 = df_vars['beta1']
                    Yb2 = df_vars['beta2']
            elif dicts[k]['title'].__contains__('Gain parameters'):
                Yvs = []
                vnames = ['g100', 'g400', 'g800', 'g1600']  # , 'g3200']
                vnames2 = ['gain_params0', 'gain_params1', 'gain_params2', 'gain_params3']  # , 'gain_params4']
                iso_vals = [100, 400, 800, 1600]  # , 3200]
                for v, vn in enumerate(vnames):
                    # Yvs.append(df_vars[vn])
                    # Yvs.append(df_vars[vn] * iso_vals[v])
                    if dicts[k]['title'].__contains__('Gain parameters S3_IP_init-5'):
                        Yvs.append(df_vars[vn])
                    elif dicts[k]['title'].__contains__('Gain parameters SG2_IP_init-0'):
                        Yvs.append(np.exp(dicts[k]['c'] * df_vars[vn]) * iso_vals[v])
                    else:
                        Yvs.append(df_vars[vn])
                try:  # read new params, if any
                    Yvs2 = []
                    for v, vn in enumerate(vnames2):
                        Yvs2.append(df_vars[vn])
                    Yvs = Yvs2
                except:
                    pass
                # dicts[k]['ylabel'] += ' [tf.exp(c * g) * iso]'
            else:
                Ytr = df_train['NLL']
                Yts = df_test['NLL']

            # plot

            if k == 0:
                pass
            elif dicts[k]['title'].__contains__('Gain parameters'):
                for v, Yv in enumerate(Yvs):
                    if dicts[k]['title'].__contains__('Gain parameters S3_IP_init-5'):
                        lbl = vnames[v]
                    else:
                        lbl = 'exp(c * ' + vnames[v] +') * ' + vnames[v][1:]
                    ax1 = plt.plot(np.arange(1, len(Yv) + 1), Yv,
                                   label=lbl,  # np.exp(dicts[k]['c'] * Yv)
                                   linestyle='-', color=new_colors[v])
                    if dicts[k]['title'].__contains__('Gain parameters SG2_IP_init-0'):
                        plt.ylim([0, .25e3])
            elif dicts[k]['title'].__contains__('SDN parameters'):
                ax1 = plt.plot(Xs, Yb1, label='sdn/b1.',
                               linestyle='-', color=new_colors[i])
                plt.plot(Xs, Yb2, label='sdn/b2',
                         linestyle='-', color=new_colors[i + 1])
            elif dicts[k]['title'].__contains__('Sampling'):
                ax1 = plt.plot(Xs, Ytr, label=dicts[k]['folders'][i]['legend'] + ' - Gauss.',
                               linestyle='-', color=new_colors[i])
                plt.plot(Xs, Yts, label=dicts[k]['folders'][i]['legend'] + ' - Cam. NLF',
                         linestyle='-', color=new_colors[i + 1])
                plt.plot(Xs, Ysm, label=dicts[k]['folders'][i]['legend'] + ' - NF',
                               linestyle='-', color=new_colors[i + 2])
            else:
                ax1 = plt.plot(df_train['epoch'], Ytr, label=dicts[k]['folders'][i]['legend'] + ' - train',
                         linestyle='--', color=new_colors[i])
                plt.plot(df_test['epoch'], Yts, label=dicts[k]['folders'][i]['legend'] + ' - test',
                         linestyle='-', color=new_colors[i])
        except Exception as exc:
            print('error: ' + dicts[k]['folders'][i]['folder'])
            print(str(exc))
            import pdb
            pdb.set_trace()

    if k == 0:

        # ylims = [[-3.7e4, -3.6e4], [-1.32e4, -1.1e4]]
        # ylims = [[-3.9e4, -3.6e4], [-1.5e4, -1.1e4]]
        fig_tt[k], ax1, ax2 = broken_ax(data_path, dicts[k], ylims=dicts[k]['ylims_brk'], u=6, r1=3, max_epc=10000)

    if k == 0:
        gidx = 4
        gauss_nll = np.load(os.path.join(data_path, dicts[k]['folders'][gidx]['folder'], 'nll_bpd_gauss.npy'))[0]
        sdn_nll = np.load(os.path.join(data_path, dicts[k]['folders'][gidx]['folder'], 'nll_bpd_sdn.npy'))[0]
        gauss_nll /= f
        Y = [gauss_nll] * plt_max_epc
        ax1.plot(Y, label='Gauss.', linestyle='-.', color=new_colors[8])
        sdn_nll /= f
        Y = [sdn_nll] * plt_max_epc
        ax1.plot(Y, label='Cam. NLF', linestyle='-.', color=new_colors[9])

    if k == 0:
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        # plt.xlim((0, 2000))
        ax1.legend(loc=dicts[k]['legend_loc'], bbox_to_anchor=dicts[k]['bbx'], prop={'size': 8 * sc}, ncol=3,
                   fancybox=True, shadow=True)
    else:
        # plt.xlim((1, plt_max_epc))
        plt.ylim(bottom=dicts[k]['ylim1'], top=dicts[k]['ylim2'])
        # ax.set_yscale(dicts[k]['yscale'], lintreshy=[-3.5e4, 5e3])
        # ax.set_yscale(dicts[k]['yscale'])
        # plt.legend(loc=dicts[k]['legend_loc'], bbox_to_anchor=dicts[k]['bbx'], prop={'size': 8})
        # if k == 3:
        #     plt.subplots_adjust(right=.8)
        # plt.legend(prop={'size': 8})
        plt.legend(loc=dicts[k]['legend_loc'], bbox_to_anchor=dicts[k]['bbx'], prop={'size': 8}, ncol=1, fancybox=True,
                   shadow=True)
        plt.title(dicts[k]['title'], fontdict={'size': fsz2})
        plt.xlabel(dicts[k]['xlabel'])#, fontdict={'size': fsz2})
        plt.ylabel(dicts[k]['ylabel'])#, fontdict={'size': fsz2})
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

    plt.tight_layout()

    fig_tt[k].savefig(os.path.join(data_path, 'figs', dicts[k]['fig_fn'] + '.png'))
    # plt.interactive(False)
    # fig_tt[k].show()
    fig_tt[k].show()
plt.show()
