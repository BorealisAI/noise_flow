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


data_path = 'experiments/sidd/'

new_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

model_names = ['Gaussian', 'Camera NLF', 'Noise FLow', 'Real']

res_dirs = ['SxUGxU', 'SUGU', 'SG', 'SUG2U', 'SUG2U_c1e-2', 'SU4GU4_1e-5', 'U16', 'SG3', 'S1G3', 'S1G3_IP']

epochs = [0, 1738, 182, 0, 0, 1431, 1883, 0, 0, 0]

temps = np.arange(.1, 1.1, .1)
kls = np.ndarray([len(temps), 4])

for res_dir, epoch in zip(res_dirs, epochs):
    for i in range(len(temps)):
        temp = temps[i]
        sub_dir = os.path.join(data_path, res_dir, 'samples_epoch_%04d' % epoch, 'samples_%.1f' % temp)
        kl_fn = os.path.join(sub_dir, 'kldiv_fwd_avg.txt')
        kls[i, :] = np.loadtxt(kl_fn)

    print(kls)

    fig = plt.figure()  # default: [6.4, 4.8]
    for i in range(len(model_names)):
        plt.plot(temps, kls[:, i], label=model_names[i])
    plt.xlabel('temperature')
    plt.ylabel('KL divergence')
    plt.legend()
    # plt.show()
    fig.savefig(os.path.join(data_path, res_dir, 'samples_epoch_%04d' % epoch, 'kldiv_fwd_vs_temp.png'))
    plt.close(fig)
