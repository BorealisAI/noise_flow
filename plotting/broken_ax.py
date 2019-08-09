# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


"""
Broken axis example, where the y-axis will have a portion cut out.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import pandas as pd


def broken_ax(data_path, plt_dict, xlims=None, ylims=None, u=4, r1=3, max_epc=2000):
    new_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf',
                  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf'
                  ]

    f = plt.figure(figsize=plt_dict['figsize'])

    # If we were to simply plot pts, we'd lose most of the interesting
    # details due to the outliers. So let's 'break' or 'cut-out' the y-axis
    # into two portions - use the top (ax) for the outliers, and the bottom
    # (ax2) for the details of the majority of our data

    # f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

    # ax2.set_xlabel('XLABEL')
    # ax2.set_ylabel('YLABEL')

    # u = 4
    v = 1
    # r1 = 3
    r2 = u - r1
    rat = u + 1 / r2  # u / r2
    print(rat)
    gridspec.GridSpec(u, v)
    ax = plt.subplot2grid((u, v), (0, 0), colspan=1, rowspan=r1)
    ax2 = plt.subplot2grid((u, v), (r1, 0), colspan=1, rowspan=r2)

    # plot the same data on both axes
    for c in range(len(plt_dict['folders'])):
        # ls = '--' if c % 2 == 0 else '-'
        # ci = c // 2
        # ax.plot(multi_pts[c][0], multi_pts[c][1], linestyle=ls, color=new_colors[ci])
        # ax2.plot(multi_pts[c][0], multi_pts[c][1], linestyle=ls, color=new_colors[ci])

        df_train = pd.read_csv(os.path.join(data_path, plt_dict['folders'][c]['folder'], 'train.txt'), sep='\t')
        df_test = pd.read_csv(os.path.join(data_path, plt_dict['folders'][c]['folder'], 'test.txt'), sep='\t')
        Xtr = df_train['epoch']
        Xts = df_test['epoch']
        Ytr = df_train['NLL']
        Yts = df_test['NLL']
        if len(Xtr) > max_epc:
            print('len=%s'%len(Xtr))
            Xtr = Xtr[:max_epc]
            print('len=%s'%len(Xtr))
        if len(Ytr) > max_epc:
            print('len=%s'%len(Ytr))
            Ytr = Ytr[:max_epc]
            print('len=%s'%len(Ytr))
        if len(Xts) > max_epc:
            print('len=%s'%len(Xts))
            Xts = Xts[:max_epc]
            print('len=%s'%len(Xts))
        if len(Yts) > max_epc:
            print('len=%s'%len(Yts))
            Yts = Yts[:max_epc]
            print('len=%s'%len(Yts))
        ax.plot(Xtr, Ytr, label=plt_dict['folders'][c]['legend'] + ' - train',
                linestyle='--', color=new_colors[c])
        ax.plot(Xts, Yts, label=plt_dict['folders'][c]['legend'] + ' - test',
                linestyle='-', color=new_colors[c])
        ax2.plot(Xtr, Ytr, label=plt_dict['folders'][c]['legend'] + ' - train',
                 linestyle='--', color=new_colors[c])
        ax2.plot(Xts, Yts, label=plt_dict['folders'][c]['legend'] + ' - test',
                 linestyle='-', color=new_colors[c])

    # zoom-in / limit the view to different portions of the data
    # ax.set_ylim(.78, 1.)  # outliers only
    # ax2.set_ylim(0, .22)  # most of the data
    if ylims is not None:
        ax2.set_ylim(ylims[0])
        ax.set_ylim(ylims[1])
    ax2.set_xlim([0, max_epc])
    ax.set_xlim([0, max_epc])

    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    # This looks pretty good, and was fairly painless, but you can get that
    # cut-out diagonal lines look with just a bit more work. The important
    # thing to know here is that in axes coordinates, which are always
    # between 0-1, spine endpoints are at these locations (0,0), (0,1),
    # (1,0), and (1,1).  Thus, we just need to put the diagonals in the
    # appropriate corners of each of our axes, and so long as we use the
    # right transform and disable clipping.

    du = .015  # .015  # how big to make the diagonal lines in axes coordinates
    dv = 0  # .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-du, +du), (-dv, +dv), **kwargs)        # top-left diagonal
    ax.plot((1 - du, 1 + du), (-dv, +dv), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-du, +du), (1 - rat * dv, 1 + rat * dv), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - du, 1 + du), (1 - rat * dv, 1 + rat * dv), **kwargs)  # bottom-right diagonal

    # What's cool about this is that now if we vary the distance between
    # ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
    # the diagonal lines will move accordingly, and stay right at the tips
    # of the spines they are 'breaking'

    # plt.show()
    return f, ax, ax2


# # 30 points between [0, 0.2) originally made using np.random.rand(30)*.2
# pts = np.array([
#     0.015, 0.166, 0.133, 0.159, 0.041, 0.024, 0.195, 0.039, 0.161, 0.018,
#     0.143, 0.056, 0.125, 0.096, 0.094, 0.051, 0.043, 0.021, 0.138, 0.075,
#     0.109, 0.195, 0.050, 0.074, 0.079, 0.155, 0.020, 0.010, 0.061, 0.008])
# # Now let's make two outlier points which are far away from everything.
# pts[[3, 14]] += .8
#
# multi_pts = [None] * 2
# multi_pts[0] = pts
# multi_pts[1] = pts + .05
#
# broken_ax(multi_pts, ylims=[[0, .22], [.87, 1]])
