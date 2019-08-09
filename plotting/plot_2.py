# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from plot_nll import plot_nll

from plot_gain_params import plot_gain_params, plot_cam_params

from plot_kld import plot_kld

from plot_nll import plot_nll_multi

from plot_kld import plot_kld_multi

from plot_sdn_params import plot_sdn_params

data_path = './experiments/sidd/'

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf',
          '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf'
          ]

# S4G4 ------------------------------------------------------
model = 'S4G4'
plot_dict = {
    'title': None,
    'folders': [
        {'folder': model, 'legend': model},
    ],
    'figsize': [6.4, 4.8],  # default [6.4, 4.8]
    'xlims': None, 'ylims': [-1.5e4, 0], 'ylims_brk': None,
    'xlabel': 'Epoch', 'ylabel': 'NLL', 'yscale': None,
    'legend_loc': 'best', 'bbx': [.5, 1.7], 'adjust': None,
    'fig_fn': None, 'gain_param_name': 'gain_params', 'c': 1, 'fsz': 18
}
plot_nll(data_path, plot_dict)
plot_dict['ylims'] = None
# plot_gain_params(data_path, plot_dict)
# plot_sdn_params(data_path, plot_dict)
# plot_kld(data_path, plot_dict)

# S4G4 ------------------------------------------------------
model = 'S4G4_IP'
plot_dict = {
    'title': None,
    'folders': [
        {'folder': model, 'legend': model},
    ],
    'figsize': [6.4, 4.8],  # default [6.4, 4.8]
    'xlims': [0, 3000], 'ylims': [-1.5e4, 0], 'ylims_brk': None,
    'xlabel': 'Epoch', 'ylabel': 'NLL', 'yscale': None,
    'legend_loc': 'best', 'bbx': [.5, 1.7], 'adjust': None,
    'fig_fn': 'S4G4_IP', 'gain_param_name': 'gain_params', 'c': 1, 'fsz': 20
}
plot_nll(data_path, plot_dict)
plot_dict['ylims'] = None
plot_dict['skip_iso'] = [3200]
plot_gain_params(data_path, plot_dict)
plot_sdn_params(data_path, plot_dict)
# plot_kld(data_path, plot_dict)

# S5G4 ------------------------------------------------------
model = 'S5G4'
plot_dict = {
    'title': None,
    'folders': [
        {'folder': model, 'legend': model},
    ],
    'figsize': [6.4, 4.8],  # default [6.4, 4.8]
    'xlims': None, 'ylims': [-1.5e4, 0], 'ylims_brk': None,
    'xlabel': 'Epoch', 'ylabel': 'NLL', 'yscale': None,
    'legend_loc': 'best', 'bbx': [.5, 1.7], 'adjust': None,
    'fig_fn': None, 'gain_param_name': 'gain_param', 'c': 1, 'fsz': 18
}
plot_nll(data_path, plot_dict)
plot_dict['ylims'] = None
# plot_gain_params(data_path, plot_dict)
# plot_sdn_params(data_path, plot_dict)
# plot_kld(data_path, plot_dict)

# camera params ------------------
models = [
    ['S5G4', 'S5G4']
]
plot_dict = {
    'title': None, 'train_or_test': 'both',
    'folders': [],
    'figsize': [6.4, 4.8],  # default [6.4, 4.8]
    'xlims': [0, 3000], 'ylims': None, 'ylims_brk': None,
    'xlabel': 'Epoch', 'ylabel': 'Camera parameters', 'yscale': None,
    'legend_loc': 'bottom center', 'bbx': [.5, 1.], 'adjust': None,
    'fig_fn': 'cam_params', 'figscale': 1, 'fsz': 18
}
for i in range(len(models)):
    plot_dict['folders'].append({'folder': models[i][0], 'legend': models[i][1]})
plot_cam_params(data_path, plot_dict)

models = [
    ['S6G4', 'S6G4']
]
plot_dict['models'] = []
plot_dict['fig_fn'] = 'cam_params_6'
for i in range(len(models)):
    plot_dict['folders'].append({'folder': models[i][0], 'legend': models[i][1]})
plot_cam_params(data_path, plot_dict)

# SUx3GUx3 ------------------------------------------------------
# model = 'SUx3GUx3'
# # NLL
# plot_dict = {
#     'title': '',
#     'folders': [
#         {'folder': model, 'legend': model},
#     ],
#     'figsize': [6.4, 4.8],  # default [6.4, 4.8]
#     'xlims': None, 'ylims': [-1.8e4, 0], 'ylims_brk': None,
#     'xlabel': 'Epoch', 'ylabel': 'NLL', 'yscale': None,
#     'legend_loc': 'best', 'bbx': [.5, 1.7], 'adjust': None,
#     'fig_fn': None, 'gain_param_name': 'g'
# }
# plot_nll(data_path, plot_dict)
# # gain params
# plot_dict['ylims'] = None
# # plot_gain_params(data_path, plot_dict)
# # sampling KLD
# plot_kld(data_path, plot_dict)

# ------------------------------------------------------

model = 'S5Ux4G4Ux4_'
# NLL
plot_dict = {
    'title': '',
    'folders': [
        {'folder': model, 'legend': model},
    ],
    'figsize': [6.4, 4.8],  # default [6.4, 4.8]
    'xlims': None, 'ylims': [-1.8e4, 0], 'ylims_brk': None,
    'xlabel': 'Epoch', 'ylabel': 'NLL', 'yscale': None,
    'legend_loc': 'best', 'bbx': [.5, 1.7], 'adjust': None,
    'fig_fn': None, 'gain_param_name': 'gain_param', 'c': 1, 'fsz': 18
}
plot_nll(data_path, plot_dict)
plot_dict['ylims'] = None
plot_gain_params(data_path, plot_dict)
plot_sdn_params(data_path, plot_dict)
plot_kld(data_path, plot_dict)

model = 'S5Ux4G4Ux4_i'
# NLL
plot_dict = {
    'title': '',
    'folders': [
        {'folder': model, 'legend': model},
    ],
    'figsize': [6.4, 4.8],  # default [6.4, 4.8]
    'xlims': None, 'ylims': [-1.8e4, 0], 'ylims_brk': None,
    'xlabel': 'Epoch', 'ylabel': 'NLL', 'yscale': None,
    'legend_loc': 'best', 'bbx': [.5, 1.7], 'adjust': None,
    'fig_fn': None, 'gain_param_name': 'gain_param', 'c': 1, 'fsz': 18
}
plot_nll(data_path, plot_dict)
plot_dict['ylims'] = None
plot_gain_params(data_path, plot_dict)
plot_sdn_params(data_path, plot_dict)
plot_kld(data_path, plot_dict)

# multi NLL & KLD ------------------------------------------------------
models = [
    'U8',  # <---
    'U16',  # <---
    'S4G4',
    'S5G4',
    'S6G4',  # <---
    # 'S6G4_sidd',
    # 'S5G4Ux4',
    # 'S4Ux4G4Ux4',
    # 'S5Ux4G4Ux4_',
    # 'S5Ux4G4Ux4_w32',
    'S5Ux4G4Ux4_i',
    # 'S5Ux8G4Ux8',
    'S5Ux8G4Ux8_i',
    'S5G4Ux16_i',
    'S5G4Ux32_i',  # <---
    # 'S5Ux4G4Ux4_w32',
    'S5Ux1G4Ux1_i',
    'S5Ux1G4Ux1_i_w32',
    'S5Ux1G4Ux1_i_w128',
]
# NLL --------------------------------------------------------
plot_dict = {
    'title': 'Training/testing NLL', 'train_or_test': 'both',
    'folders': [
        # {'folder': model1, 'legend': model1},
        # {'folder': model2, 'legend': model2},
    ],
    'figsize': [6.4 * 2, 4.8 * 2],  # default [6.4, 4.8]
    'xlims': [0, 2000], 'ylims': [-1.5e4, -1e4], 'ylims_brk': None,
    'xlabel': 'Epoch', 'ylabel': 'NLL', 'yscale': None,
    'legend_loc': 'best', 'bbx': [.5, 1.7], 'adjust': None,
    'fig_fn': '', 'figscale': 1, 'fsz': 18
}
for i in range(len(models)):
    plot_dict['folders'].append({'folder': models[i], 'legend': models[i]})
# plot_nll_multi(data_path, plot_dict, colors)

# NLL 2 --------------------------------------------------
plot_dict = {
    'title': 'Training/testing NLL', 'train_or_test': 'both',
    'folders': [],
    'figsize': [6.4 * 2, 4.8 * 2],  # default [6.4, 4.8]
    'xlims': [0, 2000], 'ylims': [-1.47e4, -1.38e4], 'ylims_brk': None,
    'xlabel': 'Epoch', 'ylabel': 'NLL', 'yscale': None,
    'legend_loc': 'bottom center', 'bbx': [.5, 1.], 'adjust': None,
    'fig_fn': '__', 'figscale': 1, 'fsz': 18
}
for i in range(len(models)):
    plot_dict['folders'].append({'folder': models[i], 'legend': models[i]})
# plot_nll_multi(data_path, plot_dict, colors)

# NLL 3 -- wide ----------------------------------------------------
plot_dict = {
    'title': 'Training/testing NLL', 'train_or_test': 'both',
    'folders': [],
    'figsize': [6.4, 4.8],  # default [6.4, 4.8]
    'xlims': None, 'ylims': None, 'ylims_brk': None,
    'xlabel': 'Epoch', 'ylabel': 'NLL', 'yscale': None,
    'legend_loc': 'bottom center', 'bbx': [.5, 1.], 'adjust': None,
    'fig_fn': 'w', 'figscale': 2, 'fsz': 18
}
for i in range(len(models)):
    plot_dict['folders'].append({'folder': models[i], 'legend': models[i]})
# plot_nll_multi(data_path, plot_dict, colors)

# NLL 4 -- NFvsBase -----------
models = [
    ['S5Ux4G4Ux4_', 'Noise Flow']
]
plot_dict = {
    'title': None, 'train_or_test': 'both',
    'folders': [],
    'figsize': [6.4, 4.8],  # default [6.4, 4.8]
    'xlims': [0, 200], 'ylims': [-3.8, -2], 'ylims_brk': None,
    'xlabel': 'Epoch', 'ylabel': 'NLL (per dimension)', 'yscale': None,
    'legend_loc': 'bottom center', 'bbx': [.5, 1.], 'adjust': None,
    'fig_fn': 'NFvsBase', 'figscale': 1, 'fsz': 20
}
for i in range(len(models)):
    plot_dict['folders'].append({'folder': models[i][0], 'legend': models[i][1]})
# plot_nll_multi(data_path, plot_dict, colors)
# plot_dict['ylims'] = [-.05, .5]
# plot_dict['ylabel'] = r'Marginal D_{KL}'
# plot_kld_multi(data_path, plot_dict, colors)


# NLL  -- ablation -- with vs without camera params -----------
models = [
    # ['U8', 'U8'],  # <---
    # ['U16', 'U16'],  # <---
    ['S4G4', 'S-G'],
    # ['S', 'S'],
    # ['S4G4', 'S4-G'],
    ['S5G4', 'S-G-CAM'],
    # ['S5Ux4G4Ux4_', 'S5Ux4G4Ux4_'],
    # ['S5Ux1G4Ux1_i', 'S-Ax1-G-Ax1'],
    # ['S5Ux4G4Ux4_i', 'S-Ax4-G-Ax4'],
]
plot_dict = {
    'title': None, 'train_or_test': 'test',
    'folders': [],
    'figsize': [6.4, 4.8],  # default [6.4, 4.8]
    'xlims': [0, 2000], 'ylims': [-1.45e4, -1.25e4], 'ylims_brk': None,
    'xlabel': 'Epoch', 'ylabel': 'NLL', 'yscale': None,
    'legend_loc': 'bottom center', 'bbx': [.5, 1.], 'adjust': None,
    'fig_fn': 'ablation_cam_params', 'figscale': 1, 'fsz': 18,
    'skip_base': True, 'skip_postfix': True, 'movavg': False, 'clr_offset': [1, 2]
}
for i in range(len(models)):
    plot_dict['folders'].append({'folder': models[i][0], 'legend': models[i][1]})
# plot_nll_multi(data_path, plot_dict, colors)
# plot_kld_multi(data_path, plot_dict, colors)


# NLL  -- ablation -----------  affine coupling effect
models = [
    # ['U8', 'U8'],  # <---
    # ['U16', 'U16'],  # <---
    # ['S4G4', 'S4G4'],
    # ['S', 'S'],
    ['S4G4', 'S4-G'],
    ['S5G4', 'S-G-CAM'],
    # ['S5Ux4G4Ux4_', 'S5Ux4G4Ux4_'],
    ['S5Ux1G4Ux1_i', 'S-Ax1-G-Ax1-CAM'],
    ['S5Ux4G4Ux4_i', 'S-Ax4-G-Ax4-CAM'],
]
plot_dict = {
    'title': None, 'train_or_test': 'test',
    'folders': [],
    'figsize': [6.4, 4.8],  # default [6.4, 4.8]
    'xlims': [0, 2000], 'ylims': [-1.444e4, -1.430e4], 'ylims_brk': None,
    'xlabel': 'Epoch', 'ylabel': 'NLL', 'yscale': None,
    'legend_loc': 'bottom center', 'bbx': [.5, 1.], 'adjust': None,
    'fig_fn': 'ablation', 'figscale': 1, 'fsz': 18,
    'skip_base': True, 'skip_postfix': True, 'movavg': True, 'clr_offset': [2, 3, 0],
    'ystep': 40
}
for i in range(len(models)):
    plot_dict['folders'].append({'folder': models[i][0], 'legend': models[i][1]})
# plot_nll_multi(data_path, plot_dict, colors)
plot_kld_multi(data_path, plot_dict, colors)


# KLD ------------------
models = [
    'U8',  # <---
    'U16',  # <---
    'S4G4',
    'S5G4',
    'S6G4',  # <---
    # 'S6G4_sidd',
    # 'S5G4Ux4',
    # 'S4Ux4G4Ux4',
    'S5Ux4G4Ux4_',
    # 'S5Ux4G4Ux4_w32',
    'S5Ux4G4Ux4_i',
    # 'S5Ux8G4Ux8',
    'S5Ux8G4Ux8_i',
    'S5G4Ux16_i',
    'S5G4Ux32_i',  # <---
    # 'S5Ux4G4Ux4_w32',
    'S5Ux1G4Ux1_i',
    'S5Ux1G4Ux1_i_w32',
    'S5Ux1G4Ux1_i_w128',
]
plot_dict = {
    'title': None, 'train_or_test': 'both',
    'folders': [],
    'figsize': [6.4, 4.8],  # default [6.4, 4.8]
    'xlims': [0, 2000], 'ylims': [-1.5e4, -.5e4], 'ylims_brk': None,
    'xlabel': 'Epoch', 'ylabel': 'NLL', 'yscale': None,
    'legend_loc': 'bottom center', 'bbx': [.5, 1.], 'adjust': None,
    'fig_fn': 'NFvsBaseX', 'figscale': 1, 'fsz': 18
}
plot_dict['title'] = None
plot_dict['ylabel'] = 'Marginal KL divergence'
plot_dict['ylims'] = [0, 0.05]
for i in range(len(models)):
    plot_dict['folders'].append({'folder': models[i], 'legend': models[i]})
# plot_kld_multi(data_path, plot_dict, colors)
