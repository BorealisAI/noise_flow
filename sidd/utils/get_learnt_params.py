# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import csv
import os
from os.path import exists

import numpy as np
import pandas as pd


def extract_save_learnt_params(model_dir, epoch=2000):
    data = pd.read_csv(os.path.join(model_dir, 'vars.txt'), sep='\t')
    params_epoch_df = data[data.epoch == epoch]
    params_epoch_df.to_csv(os.path.join(model_dir, 'learnt_params_epoch_%d.txt' % epoch), sep='\t')
    # import pdb
    # pdb.set_trace()
    return params_epoch_df


def get_learnt_params(model_dir, epoch=2000, npcam=3):
    # for S5G4 models
    fpath = os.path.join(model_dir, 'learnt_params_epoch_%d.txt' % epoch)
    if not exists(fpath):
        params_epoch_df = extract_save_learnt_params(model_dir, epoch)
    else:
        params_epoch_df = pd.read_csv(fpath, sep='\t')
    # cam params as ? x 5 array
    cam_params = np.ndarray([npcam, 5])
    for p in range(cam_params.shape[0]):
        for c in range(cam_params.shape[1]):
            cam_params[p][c] = params_epoch_df['cam_params%d%d' % (p, c)].values[0]
    # gain params as [5] array
    gain_params = np.ndarray([5])
    for g in range(5):
        gain_params[g] = params_epoch_df['gain_params%d' % g].values[0]
    # beta1, beta2
    beta1 = params_epoch_df['beta1'].values[0]
    beta2 = params_epoch_df['beta2'].values[0]
    return beta1, beta2, gain_params, cam_params


# model_dir1 = '~/skynet/_Code/fourier_flows/experiments/sidd/S6G4/'
model_dir1 = '~/_Code/fourier_flows/experiments/sidd/S6G4/'
params_epoch1 = extract_save_learnt_params(model_dir1, epoch=2000)

# print(params_epoch1)
# print(beta1)
# print(beta2)
# print(gain_params)
# print(cam_params)

# print(params_epoch1.iat[0, 0])
# print(params_epoch1.at[208, 'gain_params3'])
# print(params_epoch1['gain_params3'].values[0])
# print(type(params_epoch1['gain_params3']))
# print(params_epoch1['gain_params3'].shape)
# print(params_epoch1.values)
# print(type(params_epoch1.values))
# print(params_epoch1.values.shape)
