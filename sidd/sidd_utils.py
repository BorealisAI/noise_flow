# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

""" SIDD: Smartphone Image Denoising Dataset
# Code for:
# - preparing specific portions of training, testing, and metadata filenames
"""
import copy
import glob
import logging
import multiprocessing
import os
import queue
import random
import sys
import time

import gc
from os.path import exists
from shutil import copyfile
from threading import Thread

import h5py
import imageio
import numpy as np
from os import path

from numpy import save, load
from scipy.io import loadmat, savemat
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import gridspec


# from image_pipeline.ImagePipeline import ImagePipeline


def set_paths___del(hps):
    if hps.server == 'skynet':
        hps.sidd_path = '/shared-data/SIDD/'
        # hps.sidd_path = '/scratch/SIDD_Medium_Raw_Dir/'
        hps.sidd_bench_path = '/home/abdo/_Data/SIDD_Benchmark_Data'
        hps.sidd_bench_gt_path = '/home/abdo/_Data/SIDD_Benchmark_GT'
        hps.tr_mb_dir = '/shared-data/SIDD_MB/sidd_mbs_train_' + str(hps.n_batch_train)
        # hps.tr_mb_dir = '/scratch/sidd_mbs_train_' + str(hps.n_batch_train)
        hps.ts_mb_dir = '/shared-data/SIDD_MB/sidd_mbs_test_' + str(hps.n_batch_test)
        # hps.ts_mb_dir = '/scratch/sidd_mbs_test_' + str(hps.n_batch_test)
    else:  # eecs
        # hps.sidd_path = '/home/kamel/_Data/_New_Dataset/'
        hps.sidd_path = '/local/ssd/kamel/SIDD_Medium_Raw_Dir/'
        hps.sidd_bench_path = '/home/kamel/_Data/SIDD_Benchmark_Data'
        hps.sidd_bench_gt_path = '/home/kamel/_Data/SIDD_Benchmark_GT'
        hps.tr_mb_dir = '/local/ssd/kamel/sidd_mbs_train_' + str(hps.n_batch_train)
        hps.ts_mb_dir = '/local/ssd/kamel/sidd_mbs_test' + str(hps.n_batch_test)
    if hps.camera is not None:
        hps.tr_mb_dir += '_' + str(hps.camera)
        hps.ts_mb_dir += '_' + str(hps.camera)
    if hps.iso is not None:
        hps.tr_mb_dir += '_' + str(hps.iso)
        hps.ts_mb_dir += '_' + str(hps.iso)


def calc_train_test_stats(hps):
    hps.n_train_per_scene = hps.end_tr_im_idx - hps.start_tr_im_idx
    hps.n_test_per_scene = hps.end_ts_im_idx - hps.start_ts_im_idx
    hps.n_train = hps.n_tr_inst * hps.n_train_per_scene * hps.n_patches_per_image
    hps.n_test = hps.n_ts_inst * hps.n_test_per_scene * hps.n_patches_per_image
    hps.n_tr_bat_per_seq = int(np.ceil(hps.n_tr_inst * hps.n_patches_per_image / hps.n_batch_train))
    hps.n_ts_bat_per_seq = int(np.ceil(hps.n_ts_inst * hps.n_patches_per_image / hps.n_batch_test))


def print_train_test_stats(hps):
    logging.info('n_tr_inst = %d' % hps.n_tr_inst)
    logging.info('n_ts_inst = %d' % hps.n_ts_inst)
    logging.info('n_train_per_scene = %d' % hps.n_train_per_scene)
    logging.info('n_test_per_scene  = %d' % hps.n_test_per_scene)
    logging.info('n_patches_per_image = %d' % hps.n_patches_per_image)
    logging.info('n_train = %d' % hps.n_train)
    logging.info('n_test = %d' % hps.n_test)
    logging.info('n_batch_train = %d' % hps.n_batch_train)
    logging.info('n_batch_test = %d' % hps.n_batch_test)
    logging.info('train_its = %d' % hps.train_its)
    logging.info('test_its = %d' % hps.test_its)
    logging.info('nll_gauss = %f' % hps.nll_gauss)
    logging.info('nll_sdn = %f' % hps.nll_sdn)


def sidd_train_filenames(sidd_path, all_idxs, first_im_idx, last_im_idx, max_train_scenes=200):
    """Returns training filenames: list of tuples: (input noisy, ground truth, per-pixel variance, metadata), all .MAT
    # the number of returned tuples is multiples of 160, i.e., (last_idx - first_idx) * 160 tuples
    """
    # test indices: make sure not to use them for training, these are for testing only
    # test_idxs = [
    #     9, 21, 24, 26, 31, 37, 41, 46,
    #     49, 53, 56, 58, 61, 67, 71, 74,
    #     79, 82, 85, 93, 95, 100, 103, 109,
    #     112, 119, 124, 128, 131, 141, 143, 148,
    #     153, 158, 162, 171, 174, 176, 183, 187]
    # TODO: use all 160 scene instances
    # if not train_idxs:
    #     n_inst = max_train_scenes
    #     train_idxs = range(1, n_inst + 1)
    #     train_idxs = [i for i in train_idxs if i not in test_idxs]

    # example: 0001_001_S6_00100_00060_3200_L
    # training scenes: 1, 2, 3, 4, 5, 7
    # testing scenes: 6, 8, 9, 10
    # all_idxs = np.arange(1, 201)
    all_idxs = np.arange(16, 201)  # skip some of scene #1
    train_fns = []
    train_idxs = []
    # tr_sc = [1, 3, 4, 5, 6, 7]
    # tr_sc = [2, 9, 10]
    tr_sc = [2]
    for id in all_idxs:
        id_str = '%04d' % id
        subdir = path.split(glob.glob(path.join(sidd_path, id_str + '*'))[0])[-1]
        # skip ISO levels not found in the testing scenes: 50, 320, 500, 640, 1000, 1250, 2000, 6400, 10000
        if int(subdir[12:17]) in [50, 320, 500, 640, 1000, 1250, 2000, 10000]:  # 6400
            continue
        # skip test scenes
        if not int(subdir[5:8]) in tr_sc[:max_train_scenes]:  # 1, 3, 4, 5, 6, 7
            continue
        # skip high exposure scene instances
        if id in [3, 6]:
            continue
        n_files = len(glob.glob(path.join(sidd_path, subdir, id_str + '_GT_RAW', '*.MAT')))
        last_idx_limit = last_im_idx if last_im_idx <= n_files else n_files
        for i in range(first_im_idx - (last_im_idx - last_idx_limit) + 1, last_idx_limit + 1):
            # TODO: remove the next if statement later
            if i == 10:  # to avoid testing images, if training on the same scenes used for testing
                continue
            # each tuple: (input noisy, ground truth, per-pixel variance, metadata), all .MAT
            a_tuple = tuple(
                (
                    path.join(sidd_path, subdir, id_str + '_NOISY_RAW', id_str + '_NOISY_RAW_%03d.MAT' % i),
                    path.join(sidd_path, subdir, id_str + '_GT_RAW', id_str + '_GT_RAW_%03d.MAT' % i),
                    path.join(sidd_path, subdir, id_str + '_VARIANCE_RAW', id_str + '_VARIANCE_RAW_%03d.MAT' % i),
                    path.join(sidd_path, subdir, id_str + '_METADATA_RAW', id_str + '_METADATA_RAW_%03d.MAT' % i)
                )
            )
            train_fns.append(a_tuple)
        train_idxs.append(id)
    # print('# train scenes instances = %d' % len(train_idxs))
    # print('train scenes instances = %s' % str(train_idxs))
    # garbage collection
    all_idxs = train_idxs = id = id_str = subdir = n_files = last_idx_limit = last_im_idx = first_im_idx = i = \
        a_tuple = None
    gc.collect()
    return train_fns


def sidd_test_filenames(sidd_path, sidd_bench_path='', sidd_bench_gt_path='', all_idxs='',
                        first_im_idx=0, last_im_idx=1,
                        max_test_scenes=40):
    """Returns testing filenames: list of tuples: (input noisy, ground truth, per-pixel variance, metadata), all .MAT
    # the number of returned tuples is 40
    """
    # if not test_idxs:
    #     test_idxs = [
    #         9, 21, 24, 26, 31, 37, 41, 46,
    #         49, 53, 56, 58, 61, 67, 71, 74,
    #         79, 82, 85, 93, 95, 100, 103, 109,
    #         112, 119, 124, 128, 131, 141, 143, 148,
    #         153, 158, 162, 171, 174, 176, 183, 187]
    #     test_idxs = test_idxs[:max_test_scenes + 1]

    # example: 0001_001_S6_00100_00060_3200_L
    all_idxs = np.arange(16, 201)
    test_idxs = []
    test_fns = []

    # TODO: use all 40 images for testing
    # n_inst = 2
    # for debugging, use a few images for testing
    # test_idxs = [37]  # scene 37 is from Google Pixel
    # ts_sc = [2, 9, 10]
    ts_sc = [2]
    # ts_sc = [1, 3, 4, 5, 6, 7]
    for id in all_idxs:
        id_str = '%04d' % id
        subdir = path.split(glob.glob(path.join(sidd_path, id_str + '*'))[0])[-1]
        if int(subdir[12:17]) in [50, 320, 500, 640, 1000, 1250, 2000, 10000]:  # 6400
            continue
        # skip train scenes
        if not int(subdir[5:8]) in ts_sc[:max_test_scenes]:  # 2, 8x, 9, 10
            continue
        n_files = len(glob.glob(path.join(sidd_path, subdir, id_str + '_GT_RAW', '*.MAT')))
        last_idx_limit = last_im_idx if last_im_idx <= n_files else n_files
        for i in range(first_im_idx - (last_im_idx - last_idx_limit) + 1, last_idx_limit + 1):
            # each tuple: (input noisy, ground truth, per-pixel variance, metadata), all .MAT
            a_tuple = tuple(
                (
                    # path.join(sidd_path, subdir, id_str + '_NOISY_RAW_%03d.MAT' % test_im_idx),
                    # path.join(sidd_path, subdir, id_str + '_GT_RAW_%03d.MAT' % test_im_idx),
                    # path.join(sidd_path, subdir, id_str + '_VARIANCE_RAW_%03d.MAT' % test_im_idx),
                    # path.join(sidd_path, subdir, id_str + '_METADATA_RAW_%03d.MAT' % test_im_idx)

                    path.join(sidd_path, subdir, id_str + '_NOISY_RAW', id_str + '_NOISY_RAW_%03d.MAT' % i),
                    path.join(sidd_path, subdir, id_str + '_GT_RAW', id_str + '_GT_RAW_%03d.MAT' % i),
                    path.join(sidd_path, subdir, id_str + '_VARIANCE_RAW', id_str + '_VARIANCE_RAW_%03d.MAT' % i),
                    path.join(sidd_path, subdir, id_str + '_METADATA_RAW', id_str + '_METADATA_RAW_%03d.MAT' % i)
                )
            )
            test_fns.append(a_tuple)
        test_idxs.append(id)
    # print('# test scenes instances = %d' % len(test_idxs))
    # print('test scenes instances = %s' % str(test_idxs))
    return test_fns


def dequeue_dict(que):
    while True:
        time.sleep(random.random())
        if not que.empty():
            return que.get()


def load_one_tuple_images(filepath_tuple):
    in_path = filepath_tuple[0]  # index 0: input noisy image path
    gt_path = filepath_tuple[1]  # index 1: ground truth image path
    var_path = filepath_tuple[2]  # index 2: per-pixel noise variance
    meta_path = filepath_tuple[3]  # index 3: metadata path

    # raw = loadmat(in_path)  # (use this for .mat files without -v7.3 format)
    with h5py.File(in_path, 'r') as f:  # (use this for .mat files with -v7.3 format)
        raw = f[list(f.keys())[0]]  # use the first and only key
        # input_image = np.transpose(raw)  # TODO: transpose?
        input_image = np.expand_dims(pack_raw(raw), axis=0)
        input_image = np.nan_to_num(input_image)
        input_image = np.clip(input_image, 0.0, 1.0)

    with h5py.File(gt_path, 'r') as f:
        gt_raw = f[list(f.keys())[0]]  # use the first and only key
        # gt_image = np.transpose(gt_raw)  # TODO: transpose?
        gt_image = np.expand_dims(pack_raw(gt_raw), axis=0)
        gt_image = np.nan_to_num(gt_image)
        gt_image = np.clip(gt_image, 0.0, 1.0)

    # with h5py.File(var_path, 'r') as f:
    #     var_raw = f[list(f.keys())[0]]  # use the first and only key
    #     var_image = np.expand_dims(pack_raw(var_raw), axis=0)
    #     np.nan_to_num(var_image)
    var_image = []

    metadata = load_metadata(meta_path)

    nlf0, nlf1 = get_nlf(metadata)

    fparts = in_path.split('/')
    sdir = fparts[-3]
    if len(sdir) != 30:
        sdir = fparts[-2]  # if subdirectory does not exist
    iso = float(sdir[12:17])
    # max_iso = 3200.0
    # iso = iso / max_iso  # - 0.5  # TODO: is this okay?
    cam = float(['IP', 'GP', 'S6', 'N6', 'G4'].index(sdir[9:11]))

    # use noise layer instead of noise image TODO: just to be aware of this crucial step
    input_image = input_image - gt_image

    # fix NLF
    # nlf0 = sys.float_info.epsilon if nlf0 <= 0 else nlf0
    nlf0 = 1e-6 if nlf0 <= 0 else nlf0
    # nlf1 = sys.float_info.epsilon if nlf1 <= 0 else nlf1
    nlf1 = 1e-6 if nlf1 <= 0 else nlf1

    (one, h, w, c) = input_image.shape

    # repeat NLF TODO: get rid of np.tile()
    # add four dimension to left to be 1 x H x W x C
    # for i in range(4):
    #     nlf0 = np.expand_dims(nlf0, axis=0)
    #     nlf1 = np.expand_dims(nlf1, axis=0)
    # nlf0 = np.tile(nlf0, [1, h, w, c])
    # nlf1 = np.tile(nlf1, [1, h, w, c])

    return input_image, gt_image, var_image, nlf0, nlf1, iso, cam, metadata


def sample_patches(input_image, gt_image, var_image, nlf,
                   patch_height, patch_width, n_patches_per_image, n_target_ch, sampling='uniform'):
    H = input_image.shape[1]
    W = input_image.shape[2]
    if sampling == 'uniform':
        ii, jj, max_patches_per_image = sample_indices_uniform(H, W, patch_height, patch_width)
        if n_patches_per_image > max_patches_per_image:
            n_patches_per_image = max_patches_per_image
    else:  # 'random'
        ii, jj = sample_indices_random(H, W, patch_height, patch_width, n_patches_per_image)

    input_patches = np.zeros((n_patches_per_image, patch_height, patch_width, n_target_ch),
                             dtype=float)  # input noisy images
    gt_patches = np.zeros((n_patches_per_image, patch_height, patch_width, n_target_ch),
                          dtype=float)  # ground truth clean images
    # var_patches = np.zeros((n_patches, patch_height, patch_width, n_target_ch), dtype=float)  # per-pixel variances
    var_patches = None
    nlfs_patches = [None] * n_patches_per_image  # noise level functions

    offset = 0
    if sampling == 'uniform':
        offset = int((max_patches_per_image - n_patches_per_image) / 2)  # middle patches
    for p_idx in np.arange(n_patches_per_image):
        i = ii[p_idx + offset]
        j = jj[p_idx + offset]
        if n_target_ch == 1:
            for ch in range(4):
                input_patches[p_idx] = np.expand_dims(
                    input_image[:, i:i + patch_height, j:j + patch_width, ch], axis=3)
                gt_patches[p_idx] = np.expand_dims(
                    gt_image[:, i:i + patch_height, j:j + patch_width, ch], axis=3)
                # var_patches[p_idx] = np.expand_dims(
                #     var_image[:, yy:yy + patch_height, xx:xx + patch_width, ch], axis=3)
                nlfs_patches[p_idx] = nlf
        else:  # n_target_channels == 4
            input_patches[p_idx] = input_image[:, i:i + patch_height, j:j + patch_width, :]
            gt_patches[p_idx] = gt_image[:, i:i + patch_height, j:j + patch_width, :]
            # var_patches[p_idx] = var_image[:, yy:yy + patch_height, xx:xx + patch_width, :]
            nlfs_patches[p_idx] = nlf
        # safe mean calculation
        # patch_mean += (input_patches[patch_idx] - patch_mean) / (patch_idx + 1)
    return input_patches, gt_patches, var_patches, nlfs_patches


def sample_one_patch(input_image, gt_image, var_image, patch_height, patch_width):
    H = input_image.shape[1]
    W = input_image.shape[2]

    i = np.random.randint(0, H - patch_height + 1)
    j = np.random.randint(0, W - patch_width + 1)

    input_patch = input_image[:, i:i + patch_height, j:j + patch_width, :]
    gt_patch = gt_image[:, i:i + patch_height, j:j + patch_width, :]
    # var_patch = var_image[:, yy:yy + patch_height, xx:xx + patch_width, :]
    # nlf_patch = nlf

    return input_patch, gt_patch, None


def sample_one_patch_res(input_image, gt_image, var_image, patch_height, patch_width, res):
    H = input_image.shape[1]
    W = input_image.shape[2]

    i = np.random.randint(0, H - patch_height + 1)
    j = np.random.randint(0, W - patch_width + 1)

    input_patch = input_image[:, i:i + patch_height, j:j + patch_width, :]
    gt_patch = gt_image[:, i:i + patch_height, j:j + patch_width, :]
    # var_patch = var_image[:, yy:yy + patch_height, xx:xx + patch_width, :]

    res = (input_patch, gt_patch, None)


def sample_one_patch_args(args):
    tr_dict, patch_height, patch_width, idx = args
    H = tr_dict['tr_in_ims'][idx].shape[1]
    W = tr_dict['tr_in_ims'][idx].shape[2]

    i = np.random.randint(0, H - patch_height + 1)
    j = np.random.randint(0, W - patch_width + 1)

    input_patch = tr_dict['tr_in_ims'][idx][:, i:i + patch_height, j:j + patch_width, :]
    gt_patch = tr_dict['tr_gt_ims'][idx][:, i:i + patch_height, j:j + patch_width, :]
    # var_patch = var_image[:, yy:yy + patch_height, xx:xx + patch_width, :]
    # nlf_patch = nlf

    return input_patch, gt_patch, None


def thread_mini_batch_sampler_tr_que(tr_mb_que, n_patches, tr_dict, hps, mb_pool=None):
    thr = Thread(target=sample_mini_batch_dict_tr_que, args=(tr_mb_que, n_patches, tr_dict, hps, mb_pool))
    thr.start()
    return thr


def thread_mini_batch_sampler_ts_que(ts_mb_que, n_patches, ts_dict, hps, mb_pool=None):
    thr = Thread(target=sample_mini_batch_dict_ts_que, args=(ts_mb_que, n_patches, ts_dict, hps, mb_pool))
    thr.start()
    return thr


def sample_mini_batch_dict_tr_que___rand(mb_que, n_patches, tr_dict, hps, mb_pool, n_target_ch=4):
    while True:
        time.sleep(random.random())
        if not mb_que.full():
            t0 = time.time()
            x_batch = np.zeros((n_patches, hps.patch_height, hps.patch_height, n_target_ch), dtype=float)
            y_batch = np.zeros((n_patches, hps.patch_height, hps.patch_height, n_target_ch), dtype=float)
            nlf_batch = [None] * n_patches
            # print('sample_mini_batch_dict: from image %d to ' % hps.cur_im_idx, end='')
            for i in range(n_patches):
                (input_patch, gt_patch, var_patch) = sample_one_patch(
                    tr_dict['tr_in_ims'][hps.cur_tr_im_idx],
                    tr_dict['tr_gt_ims'][hps.cur_tr_im_idx],
                    [],
                    hps.patch_height, hps.patch_height)
                x_batch[i, :, :, :] = input_patch
                y_batch[i, :, :, :] = gt_patch
                # v_batch
                nlf_batch[i] = tr_dict['tr_nl_ims'][hps.cur_tr_im_idx]
                # round robin over images in dictionary
                hps.cur_tr_im_idx += 1
                hps.cur_tr_im_idx = hps.cur_tr_im_idx % len(tr_dict['tr_in_ims'])
            # print('Sample train batch # %d  in  %.2f  sec.' % (hps.cur_im_idx, time.time() - t0))
            mb_dict = {'_x': x_batch, '_y': y_batch, '_v': [], '_nlf': nlf_batch}
            mb_que.put(mb_dict)


def sample_mini_batch_dict_tr_que(tr_mb_que, n_patches, tr_dict, hps, mb_pool, n_target_ch=4):
    H = tr_dict['tr_in_ims'][hps.cur_tr_im_idx].shape[1]
    W = tr_dict['tr_in_ims'][hps.cur_tr_im_idx].shape[2]
    hps.cur_tr_uni_idxs = sample_indices_uniform(H, W, hps.patch_height, hps.patch_height, shuf=True)
    while True:
        time.sleep(random.random())
        if not tr_mb_que.full():
            t0 = time.time()
            x_batch = np.zeros((n_patches, hps.patch_height, hps.patch_height, n_target_ch), dtype=float)
            y_batch = np.zeros((n_patches, hps.patch_height, hps.patch_height, n_target_ch), dtype=float)
            nlf_batch = [None] * n_patches
            # print('sample_mini_batch_dict: from image %d to ' % hps.cur_im_idx, end='')
            for k in range(n_patches):
                if hps.tr_pat_idx >= hps.n_patches_per_image:
                    hps.tr_pat_idx = 0
                    hps.cur_tr_im_idx = (hps.cur_tr_im_idx + 1) % len(tr_dict['tr_in_ims'])
                    hps.cur_tr_uni_idxs = sample_indices_uniform(H, W, hps.patch_height, hps.patch_height, shuf=True)
                # ii, jj, n_p = sample_indices_uniform(H, W, hps.patch_height, hps.patch_height)
                i = hps.cur_tr_uni_idxs[0][hps.tr_pat_idx]
                j = hps.cur_tr_uni_idxs[1][hps.tr_pat_idx]
                input_patch = tr_dict['tr_in_ims'][hps.cur_tr_im_idx][:, i:i + hps.patch_height, j:j + hps.patch_height,
                              :]
                gt_patch = tr_dict['tr_gt_ims'][hps.cur_tr_im_idx][:, i:i + hps.patch_height, j:j + hps.patch_height, :]
                # var_patch = var_image[:, yy:yy + patch_height, xx:xx + patch_width, :]
                # nlf_patch = nlf
                x_batch[k, :, :, :] = input_patch
                y_batch[k, :, :, :] = gt_patch
                # v_batch
                nlf_batch[k] = tr_dict['tr_nl_ims'][hps.cur_tr_im_idx]
            # print('Sample test batch # %d in  %.2f  sec.' % (hps.cur_tr_im_idx, time.time() - t0))
            mb_dict = {'_x': x_batch, '_y': y_batch, '_v': [], '_nlf': nlf_batch}
            tr_mb_que.put(mb_dict)


def sample_mini_batch_dict_ts_que___rand(ts_mb_que, n_patches, ts_dict, hps, mb_pool, n_target_ch=4):  # RANDOM
    while True:
        time.sleep(random.random())
        if not ts_mb_que.full():
            t0 = time.time()
            x_batch = np.zeros((n_patches, hps.patch_height, hps.patch_height, n_target_ch), dtype=float)
            y_batch = np.zeros((n_patches, hps.patch_height, hps.patch_height, n_target_ch), dtype=float)
            nlf_batch = [None] * n_patches
            # print('sample_mini_batch_dict: from image %d to ' % hps.cur_im_idx, end='')
            for k in range(n_patches):
                (input_patch, gt_patch, var_patch) = sample_one_patch(
                    ts_dict['ts_in_ims'][hps.cur_ts_im_idx],
                    ts_dict['ts_gt_ims'][hps.cur_ts_im_idx],
                    [],
                    hps.patch_height, hps.patch_height)
                x_batch[k, :, :, :] = input_patch
                y_batch[k, :, :, :] = gt_patch
                # v_batch
                nlf_batch[k] = ts_dict['ts_nl_ims'][hps.cur_ts_im_idx]
                # round robin over images in dictionary
                hps.cur_ts_im_idx += 1
                hps.cur_ts_im_idx = hps.cur_ts_im_idx % len(ts_dict['ts_in_ims'])
            # print('Sample test batch # %d in  %.2f  sec.' % (hps.cur_ts_im_idx, time.time() - t0))
            mb_dict = {'_x': x_batch, '_y': y_batch, '_v': [], '_nlf': nlf_batch}
            ts_mb_que.put(mb_dict)


def sample_mini_batch_dict_ts_que(ts_mb_que, n_patches, ts_dict, hps, mb_pool, n_target_ch=4):
    H = ts_dict['ts_in_ims'][hps.cur_ts_im_idx].shape[1]
    W = ts_dict['ts_in_ims'][hps.cur_ts_im_idx].shape[2]
    hps.cur_ts_uni_idxs = sample_indices_uniform(H, W, hps.patch_height, hps.patch_height, shuf=True)
    while True:
        time.sleep(random.random())
        if not ts_mb_que.full():
            t0 = time.time()
            x_batch = np.zeros((n_patches, hps.patch_height, hps.patch_height, n_target_ch), dtype=float)
            y_batch = np.zeros((n_patches, hps.patch_height, hps.patch_height, n_target_ch), dtype=float)
            nlf_batch = [None] * n_patches
            # print('sample_mini_batch_dict: from image %d to ' % hps.cur_im_idx, end='')
            for k in range(n_patches):
                if hps.ts_pat_idx >= hps.n_patches_per_image:
                    hps.ts_pat_idx = 0
                    hps.cur_ts_im_idx = (hps.cur_ts_im_idx + 1) % len(ts_dict['ts_in_ims'])
                    # hps.cur_ts_im_idx += 1
                    # if hps.cur_ts_im_idx >= hps.end_ts_im_idx:
                    #     hps.cur_ts_im_idx = hps.start_ts_im_idx
                    hps.cur_ts_uni_idxs = sample_indices_uniform(H, W, hps.patch_height, hps.patch_height, shuf=True)
                # ii, jj, n_p = sample_indices_uniform(H, W, hps.patch_height, hps.patch_height)
                i = hps.cur_ts_uni_idxs[0][hps.ts_pat_idx]
                j = hps.cur_ts_uni_idxs[1][hps.ts_pat_idx]
                input_patch = ts_dict['ts_in_ims'][hps.cur_ts_im_idx][:, i:i + hps.patch_height, j:j + hps.patch_height,
                              :]
                gt_patch = ts_dict['ts_gt_ims'][hps.cur_ts_im_idx][:, i:i + hps.patch_height, j:j + hps.patch_height, :]
                # var_patch = var_image[:, yy:yy + patch_height, xx:xx + patch_width, :]
                # nlf_patch = nlf
                x_batch[k, :, :, :] = input_patch
                y_batch[k, :, :, :] = gt_patch
                # v_batch
                nlf_batch[k] = ts_dict['ts_nl_ims'][hps.cur_ts_im_idx]
            # print('Sample test batch # %d in  %.2f  sec.' % (hps.cur_ts_im_idx, time.time() - t0))
            mb_dict = {'_x': x_batch, '_y': y_batch, '_v': [], '_nlf': nlf_batch}
            ts_mb_que.put(mb_dict)


def sample_mini_batch_dict_tr(n_patches, tr_dict, hps, mb_pool=None, n_target_ch=4):
    t0 = time.time()
    x_batch = np.zeros((n_patches, hps.patch_height, hps.patch_height, n_target_ch), dtype=float)
    y_batch = np.zeros((n_patches, hps.patch_height, hps.patch_height, n_target_ch), dtype=float)
    nlf_batch = [None] * n_patches
    print('sample_mini_batch_dict: from image %d to ' % hps.cur_tr_im_idx, end='')
    # res = mb_pool.map(sample_one_patch_args, zip([tr_dict] * n_patches, [hps.patch_height] * n_patches,
    #                                              [hps.patch_height] * n_patches, range(n_patches)))
    for i in range(n_patches):
        (input_patch, gt_patch, var_patch) = sample_one_patch(
            tr_dict['tr_in_ims'][hps.cur_tr_im_idx],
            tr_dict['tr_gt_ims'][hps.cur_tr_im_idx],
            [],
            hps.patch_height, hps.patch_height)
        x_batch[i, :, :, :] = input_patch
        y_batch[i, :, :, :] = gt_patch
        # v_batch
        nlf_batch[i] = tr_dict['tr_nl_ims'][hps.cur_tr_im_idx]
        # round robin over images in dictionary
        hps.cur_tr_im_idx += 1
        # hps.cur_tr_im_idx = hps.cur_tr_im_idx % len(tr_dict['tr_in_ims'])
    print('%3d  in  %.2f  sec.' % (hps.cur_tr_im_idx, time.time() - t0))
    return x_batch, y_batch, [], nlf_batch


def sample_mini_batch_dict_tr_threads(n_patches, tr_dict, hps, mb_pool=None, n_target_ch=4):
    t0 = time.time()
    x_batch = np.zeros((n_patches, hps.patch_height, hps.patch_height, n_target_ch), dtype=float)
    y_batch = np.zeros((n_patches, hps.patch_height, hps.patch_height, n_target_ch), dtype=float)
    nlf_batch = [None] * n_patches
    print('sample_mini_batch_dict: from image %d to ' % hps.cur_tr_im_idx, end='')
    nt = 4
    thr = [None] * nt
    res = [queue.Queue(1)] * nt
    for i in range(0, n_patches, nt):
        for t in range(nt):
            thr[t] = Thread(target=sample_one_patch_res, args=(tr_dict['tr_in_ims'][hps.cur_tr_im_idx + t],
                                                               tr_dict['tr_gt_ims'][hps.cur_tr_im_idx + t],
                                                               [], hps.patch_height, hps.patch_height, res[t]))
            thr[t].start()
        for t in range(nt):
            thr[t].join()
            r = res[t].get()
            x_batch[i + t, :, :, :] = r[0]
            y_batch[i + t, :, :, :] = r[1]
            # v_batch
            nlf_batch[i + t] = tr_dict['tr_nl_ims'][hps.cur_tr_im_idx]

        # round robin over images in dictionary
        hps.cur_tr_im_idx += nt
        hps.cur_tr_im_idx = hps.cur_tr_im_idx % len(tr_dict['tr_in_ims'])
    print('%3d  in  %.2f  sec.' % (hps.cur_tr_im_idx, time.time() - t0))
    return x_batch, y_batch, [], nlf_batch


def get_save_patch_stats(hps, train_its, tr_que, tr_mb_que):
    postfix = '_nseq_%d_nsce_%d_npat_%d' % (hps.n_train_per_scene, hps.n_train_scenes, hps.n_patches_per_image)
    if not exists('sidd/in_mean%s.npy' % postfix):
        t1 = time.time()
        in_mean = np.zeros((1, hps.patch_height, hps.patch_height, 4))
        gt_mean = np.zeros((1, hps.patch_height, hps.patch_height, 4))
        in_std = np.zeros((1, hps.patch_height, hps.patch_height, 4))
        gt_std = np.zeros((1, hps.patch_height, hps.patch_height, 4))
        n = 0
        t2 = time.time()
        for it in range(train_its):
            # print('*** iteration: %5d / %5d' % (it, train_its), flush=True)
            if it > 0 and (it % hps.n_tr_bat_per_seq == 0):  # reload another sequence of images
                tr_dict = dequeue_dict(tr_que)  # blocking
                gc.collect()
                tim2 = time.time() - t2
                print('*** it: %5d / %5d\t time = %.2f sec.' % (it, train_its, tim2), flush=True)
                print('*** approx. time remaining = %.2f sec.' % (tim2 * (train_its - it) / hps.n_tr_bat_per_seq))
                t2 = time.time()
            # _x, _y, v_batch, n_batch = sample_mini_batch_dict_tr(hps.n_batch_train, tr_dict, hps)
            # _x, _y, v_batch, n_batch = sample_mini_batch_dict_tr_threads(hps.n_batch_train, tr_dict, hps)
            tr_mb_dict = dequeue_dict(tr_mb_que)
            gc.collect()
            _x = tr_mb_dict['_x']
            _y = tr_mb_dict['_y']
            v_batch = tr_mb_dict['_v']
            n_batch = tr_mb_dict['_nlf']
            n += 1  # TODO: check if this is a bug
            # pre_in_mean = in_mean
            # pre_gt_mean = gt_mean
            in_mean = in_mean + (_x[0, :, :, :] - in_mean) / n
            gt_mean = gt_mean + (_y[0, :, :, :] - gt_mean) / n
            for p in range(1, hps.n_batch_train):
                n += 1
                # this is sample variance, take square root after the loop
                in_std = in_std + \
                         ((_x[p, :, :, :] - in_mean) * (_x[p, :, :, :] - in_mean) / n) - \
                         ((in_std * in_std) / (n - 1))
                gt_std = gt_std + \
                         ((_y[p, :, :, :] - gt_mean) * (_y[p, :, :, :] - gt_mean) / n) - \
                         ((gt_std * gt_std) / (n - 1))
                # pre_in_mean = in_mean
                # pre_gt_mean = gt_mean
                in_mean = in_mean + (_x[p, :, :, :] - in_mean) / n
                gt_mean = gt_mean + (_y[p, :, :, :] - gt_mean) / n
                # this is variance, take square root after the loop
                # in_std = in_std + (((_x[p, :, :, :] - pre_in_mean) * (_x[p, :, :, :] - in_mean)) - in_std) / n
                # gt_std = gt_std + (((_y[p, :, :, :] - pre_gt_mean) * (_y[p, :, :, :] - gt_mean)) - gt_std) / n

        # scalar mean and scalar sample variance
        k = hps.n_train
        g = hps.patch_height * hps.patch_height * 4
        sc_in_mean = np.mean(in_mean)
        sc_gt_mean = np.mean(gt_mean)
        t_sum = np.sum(in_std) + np.var(in_mean) * (k * (g - 1)) / (k - 1)
        sc_in_std = t_sum * (k - 1) / (k * g - 1)
        t_sum = np.sum(gt_std) + np.var(gt_mean) * (k * (g - 1)) / (k - 1)
        sc_gt_std = t_sum * (k - 1) / (k * g - 1)
        in_std = np.sqrt(in_std)
        gt_std = np.sqrt(gt_std)
        sc_in_std = np.sqrt(sc_in_std)
        sc_gt_std = np.sqrt(sc_gt_std)

        save('sidd/in_mean%s.npy' % postfix, in_mean)
        save('sidd/gt_mean%s.npy' % postfix, gt_mean)
        save('sidd/in_std%s.npy' % postfix, in_std)
        save('sidd/gt_std%s.npy' % postfix, gt_std)
        save('sidd/sc_in_mean%s.npy' % postfix, sc_in_mean)
        save('sidd/sc_gt_mean%s.npy' % postfix, sc_gt_mean)
        save('sidd/sc_in_std%s.npy' % postfix, sc_in_std)
        save('sidd/sc_gt_std%s.npy' % postfix, sc_gt_std)
        print('training mean and std. dev.: time = %f sec.' % (time.time() - t1))
    else:
        in_mean = load('sidd/in_mean%s.npy' % postfix)
        gt_mean = load('sidd/gt_mean%s.npy' % postfix)
        in_std = load('sidd/in_std%s.npy' % postfix)
        gt_std = load('sidd/gt_std%s.npy' % postfix, )
        sc_in_mean = load('sidd/sc_in_mean%s.npy' % postfix)
        sc_gt_mean = load('sidd/sc_gt_mean%s.npy' % postfix)
        sc_in_std = load('sidd/sc_in_std%s.npy' % postfix)
        sc_gt_std = load('sidd/sc_gt_std%s.npy' % postfix)
    return dict({'in_mean': in_mean, 'gt_mean': gt_mean, 'in_std': in_std, 'gt_std': gt_std,
                 'sc_in_mean': sc_in_mean, 'sc_gt_mean': sc_gt_mean, 'sc_in_std': sc_in_std, 'sc_gt_std': sc_gt_std})


def get_save_gauss_nll_bpd(hps, test_its, ts_que, ts_mb_que, pat_stats):
    postfix = '_nseq_%d_nsce_%d_npat_%d' % (hps.n_train_per_scene, hps.n_train_scenes, hps.n_patches_per_image)
    if not exists('sidd/gauss_nll%s.npy' % postfix):
        t1 = time.time()
        sum_sqr = 0.0
        sum_sqr_lst = [sum_sqr]
        n = 0
        t2 = time.time()
        for it in range(test_its):
            # print('*** iteration: %5d / %5d' % (it, train_its), flush=True)
            if it > 0 and (it % hps.n_ts_bat_per_seq == 0):  # reload another sequence of images
                ts_dict = dequeue_dict(ts_que)  # blocking
                # gc.collect()
                tim2 = time.time() - t2
                print('*** it: %5d / %5d\t time = %.2f sec.' % (it, test_its, tim2), flush=True)
                print('*** approx. time remaining = %.2f sec.' % (tim2 * (test_its - it) / hps.n_ts_bat_per_seq))
                t2 = time.time()
            ts_mb_dict = dequeue_dict(ts_mb_que)
            # gc.collect()
            _x = ts_mb_dict['_x']
            # _y = ts_mb_dict['_y']
            # v_batch = ts_mb_dict['_v']
            # n_batch = ts_mb_dict['_nlf']
            n += 1
            # n += np.prod(_x.shape)
            sum_sqr += np.mean((_x - pat_stats['sc_in_mean']) ** 2)
            sum_sqr_lst.append(sum_sqr)

        sig_sqr = pat_stats['sc_in_std'] ** 2
        gauss_nll = (n / 2.0) * np.log(2 * np.pi * sig_sqr) + \
                    (1.0 / (2 * sig_sqr)) * sum_sqr
        gauss_bpd = bpd(gauss_nll, hps.n_bins, hps.n_dims)
        save('sidd/gauss_nll%s.npy' % postfix, gauss_nll)
        save('sidd/gauss_bpd%s.npy' % postfix, gauss_bpd)
        save('sidd/gauss_sum_sqr%s.npy' % postfix, sum_sqr)
        save('sidd/gauss_sum_sqr_lst%s.npy' % postfix, sum_sqr_lst)
        print('gauss_nll and gauss_bpd: time = %f sec.' % (time.time() - t1))
    else:
        gauss_nll = load('sidd/gauss_nll%s.npy' % postfix)
        gauss_bpd = load('sidd/gauss_bpd%s.npy' % postfix)
    return gauss_nll, gauss_bpd


def load_patch_stats_tmp():
    postfix = '_nseq_%d_nsce_%d_npat_%d' % (100, 118, 180)
    in_mean = load('sidd/in_mean%s.npy' % postfix)
    gt_mean = load('sidd/gt_mean%s.npy' % postfix)
    in_std = load('sidd/in_std%s.npy' % postfix)
    gt_std = load('sidd/gt_std%s.npy' % postfix, )
    sc_in_mean = load('sidd/sc_in_mean%s.npy' % postfix)
    sc_gt_mean = load('sidd/sc_gt_mean%s.npy' % postfix)
    sc_in_std = load('sidd/sc_in_std%s.npy' % postfix)
    sc_gt_std = load('sidd/sc_gt_std%s.npy' % postfix)
    return dict({'in_mean': in_mean, 'gt_mean': gt_mean, 'in_std': in_std, 'gt_std': gt_std,
                 'sc_in_mean': sc_in_mean, 'sc_gt_mean': sc_gt_mean, 'sc_in_std': sc_in_std, 'sc_gt_std': sc_gt_std})


def load_gauss_nll_bpd_tmp():
    postfix = '_nseq_%d_nsce_%d_npat_%d' % (100, 118, 180)
    gauss_nll = load('sidd/gauss_nll%s.npy' % postfix)
    gauss_bpd = load('sidd/gauss_bpd%s.npy' % postfix)
    gauss_sum_sqr = load('sidd/gauss_sum_sqr%s.npy' % postfix)
    gauss_sum_sqr_lst = load('sidd/gauss_sum_sqr_lst%s.npy' % postfix)
    return gauss_nll, gauss_bpd


def load_metadata(meta_path):
    """Loads metadata from file."""
    meta = loadmat(meta_path)
    # meta = meta[list(meta.keys())[3]]  # 3rd key: 'metadata'
    meta = meta['metadata']  # key: 'metadata'
    return meta[0, 0]


def get_nlf(metadata):
    nlf = metadata['UnknownTags'][7, 0][2][0][0:2]
    # print('nlf shape = %s' % str(nlf.shape))
    return nlf


def pack_raw(raw_im):
    """Packs Bayer image to 4 channels (h, w) --> (h/2, w/2, 4)."""
    # pack Bayer image to 4 channels
    im = np.expand_dims(raw_im, axis=2)
    img_shape = im.shape
    # print('img_shape: ' + str(img_shape))
    h = img_shape[0]
    w = img_shape[1]
    out = np.concatenate((im[0:h:2, 0:w:2, :],
                          im[0:h:2, 1:w:2, :],
                          im[1:h:2, 1:w:2, :],
                          im[1:h:2, 0:w:2, :]), axis=2)

    del raw_im
    gc.collect()

    return out


def unpack_raw(raw4ch):
    """Unpacks 4 channels to Bayer image (h/2, w/2, 4) --> (h, w)."""
    img_shape = raw4ch.shape
    h = img_shape[0]
    w = img_shape[1]
    # d = img_shape[2]
    bayer = np.zeros([h * 2, w * 2], dtype=np.float32)
    # bayer = raw4ch
    # bayer.reshape((h * 2, w * 2))
    bayer[0::2, 0::2] = raw4ch[:, :, 0]
    bayer[0::2, 1::2] = raw4ch[:, :, 1]
    bayer[1::2, 1::2] = raw4ch[:, :, 2]
    bayer[1::2, 0::2] = raw4ch[:, :, 3]
    return bayer


def synthesize_noise(model, params, shape, clean_image):
    """Returns a synthetic noise layer of size ('shape') sampled from a noise model ('model') with parameters ('params')
     optionally conditioned on a clean image ('clean_image').
     If model is 'gauss', params = {mean, stddev}.
     If model is 'nlf', params = {beta1, beta2}, i.e., slope and intersection of noise level function (NLF).
    """
    if model == 'nlf':
        local_stddev = params['beta1'] * clean_image + params['beta2']
        return np.random.normal(0.0, 1.0, shape) * local_stddev
    else:  # gauss
        return np.random.normal(params['mean'], params['stddev'], shape)


def sidd_preprocess(x, min_, max_, mean_):
    """Normalize and mean-shift."""
    # if type(x) != np.float32 or type(x) != 'float' or type(x) != tf.float32:
    #     x = np.asarray(x).astype('float32')
    if min_ is None or max_ is None:
        min_ = np.min(x, axis=0)
        max_ = np.max(x, axis=0)
    x = x - min_
    x = x / (max_ - min_)
    if mean_ is None:
        mean_ = np.mean(x, axis=0)
    x = x - mean_
    return x, min_, max_, mean_


def sidd_postprocess(x, min_, max_, mean_):
    """Undo normalize and mean-shift."""
    # if type(x) != np.float32 or type(x) != 'float' or type(x) != tf.float32:
    #     x = np.asarray(x).astype('float32')
    if mean_ is None:
        mean_ = tf.reduce_mean(x, axis=0)
    x = x + mean_
    if min_ is None or max_ is None:
        min_ = tf.reduce_min(x, axis=0)
        max_ = tf.reduce_max(x, axis=0)
    x = x * (max_ - min_)
    x = x + min_
    return x


def sidd_preprocess_standardize(x, mean_, stddev_):
    """Mean-shift & normalize."""
    # if type(x) != np.float32 or type(x) != 'float' or type(x) != tf.float32:
    #     x = np.asarray(x).astype('float32')
    if stddev_ is None:
        stddev_ = np.std(x, axis=0)
    if mean_ is None:
        mean_ = np.mean(x, axis=0)
    x = (x - mean_) / stddev_
    return x, mean_, stddev_


def sidd_postprocess_undo_standardize(x, mean_, stddev_):
    """Undo mean-shift & normalize."""
    # if type(x) != np.float32 or type(x) != 'float' or type(x) != tf.float32:
    #     x = np.asarray(x).astype('float32')
    x = x * stddev_ + mean_
    return x


def sample_indices_uniform(h, w, ph, pw, shuf=False, n_pat_per_im=None):
    """Uniformly sample patch indices from (0, 0) up to (h, w) with patch height and width (ph, pw) """
    ii = []
    jj = []
    n_p = 0
    for i in np.arange(0, h - ph + 1, ph):
        for j in np.arange(0, w - pw + 1, pw):
            ii.append(i)
            jj.append(j)
            n_p += 1
            if (n_pat_per_im is not None) and (n_p == n_pat_per_im):
                break
        if (n_pat_per_im is not None) and (n_p == n_pat_per_im):
            break
    if shuf:
        ii, jj = shuffle(ii, jj)
    return ii, jj, n_p


def sample_indices_random(h, w, ph, pw, n_p):
    """Randomly sample n_p patch indices from (0, 0) up to (h, w) with patch height and width (ph, pw) """
    ii = []
    jj = []
    for k in np.arange(0, n_p):
        i = np.random.randint(0, h - ph + 1)
        j = np.random.randint(0, w - pw + 1)
        ii.append(i)
        jj.append(j)
    return ii, jj


def nll_normal(mu, vr, data):
    """Negative log likelihood of data given mean (mu) and variance (vr) of a univariate normal distribution
    NLL is computed pixel-wise (i.e., each pixel is a data point"""
    n = np.prod(data.shape)
    nll = (n / 2) * np.log(2 * np.pi * vr) + np.sum((data - mu) ** 2) / (2 * vr)
    return nll


def kld_nrm(data1, data2):
    """KL divergence between two sets of data, assuming normal distributions"""
    n = np.prod(data1.shape)
    v1 = np.var(data1)
    v2 = np.var(data2)
    kld = 0.5 * np.log(v2 / v1) + ((v1 + (data1 - data2) ** 2) / (2 * v2)) - 0.5
    kld = np.sum(kld)
    return kld


def bpd(nll, n_bins, n_dims):
    quantiz_factor = np.log(n_bins)  # divided by n_dim already
    return (nll / n_dims + np.float64(quantiz_factor)) / (np.log(2.))  # NLL is not divided by n_dim


def sidd_filenames_que_inst(sidd_path, train_or_test='train', first_im_idx=0, last_im_idx=1, cam=None, iso=None):
    """Returns filenames: list of tuples: (input noisy, ground truth, per-pixel variance, metadata), all .MAT
    """
    if train_or_test == 'train':
        inst_idxs = [4, 11, 13, 17, 18, 20, 22, 23, 25, 27, 28, 29, 30, 34, 35, 39, 40, 42, 43, 44, 45, 47, 81, 86, 88,
                     90, 101, 102, 104, 105, 110, 111, 115, 116, 125, 126, 127, 129, 132, 135,
                     138, 140, 175, 177, 178, 179, 180, 181, 185, 186, 189, 192, 193, 194, 196, 197]
        # removed: 114, 134, 184, 136, 190, 188, 117, 137, 191
    else:
        inst_idxs = [54, 55, 57, 59, 60, 62, 63, 66, 150, 151, 152, 154, 155, 159, 160, 161, 163, 164, 165, 166, 198,
                     199]
    # example: 0001_001_S6_00100_00060_3200_L
    fns = queue.Queue(0)
    idxs = []
    cnt_inst = 0
    for id in inst_idxs:
        id_str = '%04d' % id
        subdir = path.split(glob.glob(path.join(sidd_path, id_str + '*'))[0])[-1]
        if (cam is not None) and (subdir[9:11] != cam):
            continue
        if (iso is not None) and (iso != 0) and (int(subdir[12:17]) != iso):
            continue
        n_files = len(glob.glob(path.join(sidd_path, subdir, id_str + '_GT_RAW', '*.MAT')))
        for i in range(first_im_idx, last_im_idx):
            if 'SIDD_Medium' in sidd_path:
                a_tuple = tuple(
                    (
                        path.join(sidd_path, subdir, id_str + '_NOISY_RAW_%03d.MAT' % i),
                        path.join(sidd_path, subdir, id_str + '_GT_RAW_%03d.MAT' % i),
                        path.join(sidd_path, subdir, id_str + '_VARIANCE_RAW_%03d.MAT' % i),
                        path.join(sidd_path, subdir, id_str + '_METADATA_RAW_%03d.MAT' % i)
                    )
                )
            else:
                a_tuple = tuple(
                    (
                        path.join(sidd_path, subdir, id_str + '_NOISY_RAW', id_str + '_NOISY_RAW_%03d.MAT' % i),
                        path.join(sidd_path, subdir, id_str + '_GT_RAW', id_str + '_GT_RAW_%03d.MAT' % i),
                        path.join(sidd_path, subdir, id_str + '_VARIANCE_RAW', id_str + '_VARIANCE_RAW_%03d.MAT' % i),
                        path.join(sidd_path, subdir, id_str + '_METADATA_RAW', id_str + '_METADATA_RAW_%03d.MAT' % i)
                    )
                )
            fns.put(a_tuple)
        idxs.append(id)
        cnt_inst += 1
    return fns, cnt_inst


def save_minibatch(mb, dir, id):
    np.save(os.path.join(dir, 'mb_%08d_meta.npy' % id), mb['metadata'])
    mb['metadata'] = None
    np.save(os.path.join(dir, 'mb_%08d.npy' % id), mb)


def load_minibatch(dir1, id1):
    mb = np.load(os.path.join(dir1, 'mb_%08d.npy' % id1)).item()
    meta = np.load(os.path.join(dir1, 'mb_%08d_meta.npy' % id1))
    mb['metadata'] = meta
    return mb


def load_minibatch_que(dir1, id1, que):
    mb = np.load(os.path.join(dir1, 'mb_%08d.npy' % id1)).item()
    meta = np.load(os.path.join(dir1, 'mb_%08d_meta.npy' % id1))
    mb['metadata'] = meta
    que.put(mb)


def load_minibatches_que(mb_dir, ids, que, requeue=False, calc_stds=False):
    if calc_stds:
        cam_ids = ['GP', 'IP', 'S6', 'N6', 'G4']
        iso_ids = ['00100', '00400', '00800', '01600', '03200']
        n_cam = 5
        n_iso = 5
        stds = np.ndarray([n_cam, n_iso])
        std_cnts = np.ndarray([n_cam, n_iso])
        stds[:] = 0.0

    mb = None
    meta = None
    first = True
    while first or requeue:
        first = False
        for i in ids:
            # print('loading mb %d' % i)
            mb = np.load(os.path.join(mb_dir, 'mb_%08d.npy' % i)).item()
            meta = np.load(os.path.join(mb_dir, 'mb_%08d_meta.npy' % i))
            mb['metadata'] = meta
            mb['id'] = i
            que.put(mb)

            if calc_stds:
                cam_idx = cam_ids.index(mb['fn'][9:11])
                iso_idx = iso_ids.index(mb['fn'][12:17])
                stds[cam_idx, iso_idx] += \
                    np.sqrt(np.mean((mb['_x'] / (np.sqrt(mb['_y']) + sys.float_info.epsilon)) ** 2))
                std_cnts[cam_idx, iso_idx] += 1
        if calc_stds:
            stds /= std_cnts
            stds = np.sqrt(stds)
            np.savetxt('stds.txt', stds)


def save_visual_minibatch(mb, hps):
    np.save(os.path.join('experiments', hps.problem, 'vis_mb_meta.npy'), mb['metadata'])
    mb['metadata'] = None
    np.save(os.path.join('experiments', hps.problem, 'vis_mb.npy'), mb)
    with open(os.path.join('experiments', hps.problem, 'vis_mb.txt'), 'w') as text_file:
        text_file.write("fn=%s" % mb['fn'])


def calc_kldiv_mb(mb, x_samples, vis_dir, sc_sd):
    subdir = str(mb['fn']).split('|')[0]
    subdir = os.path.join(vis_dir, subdir)
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    step = 5  # 30
    n_models = 4
    klds_que = queue.Queue()
    klds_avg = np.ndarray([n_models])
    klds_avg[:] = 0.0
    cnt = 0
    # thr = []
    for i in range(0, mb['_x'].shape[0], step):
        kldiv_patch_set(i, mb, x_samples, sc_sd, subdir, klds_que)
        cnt += 1
    for k in range(cnt):
        klds_avg += klds_que.get()
    return klds_avg / cnt


def kldiv_patch_set(i, mb, x_samples, sc_sd, subdir, klds_que):
    y = unpack_raw(mb['_y'][i, :, :, :])
    nlf_sd = np.sqrt(mb['nlf0'] * y + mb['nlf1'])  # Camera NLF
    ng = np.random.normal(0, sc_sd, y.shape)  # Gaussian
    ns = unpack_raw(x_samples[i, :, :, :])  # NF-sampled
    nl = nlf_sd * np.random.normal(0, 1, y.shape)  # Camera NLF
    n = unpack_raw(mb['_x'][i, :, :, :])  # Real
    xs = np.clip(y + ns, 0.0, 1.0)
    xg = np.clip(y + ng, 0.0, 1.0)
    xl = np.clip(y + nl, 0.0, 1.0)
    x = np.clip(y + n, 0.0, 1.0)
    pid = mb['pid'][i]
    noise_pats_raw = (ng, nl, ns, n)

    # savemat(os.path.join(subdir, 'meta.mat'), {'x': meta})

    save_mat = True
    # save mat files
    if save_mat:
        savemat(os.path.join(subdir, '%s_%04d.mat' % ('y', pid)), {'x': y})
        savemat(os.path.join(subdir, '%s_%04d.mat' % ('ng', pid)), {'x': ng})
        savemat(os.path.join(subdir, '%s_%04d.mat' % ('nl', pid)), {'x': nl})
        savemat(os.path.join(subdir, '%s_%04d.mat' % ('ns', pid)), {'x': ns})
        savemat(os.path.join(subdir, '%s_%04d.mat' % ('n', pid)), {'x': n})
        savemat(os.path.join(subdir, '%s_%04d.mat' % ('xg', pid)), {'x': xg})
        savemat(os.path.join(subdir, '%s_%04d.mat' % ('xl', pid)), {'x': xl})
        savemat(os.path.join(subdir, '%s_%04d.mat' % ('xs', pid)), {'x': xs})
        savemat(os.path.join(subdir, '%s_%04d.mat' % ('x', pid)), {'x': x})
    # histograms
    bw = 0.2 / 64
    bin_edges = np.concatenate(([-1000.0], np.arange(-0.1, 0.1 + 1e-9, bw), [1000.0]), axis=0)
    hists = [None] * len(noise_pats_raw)
    klds = np.ndarray([len(noise_pats_raw)])
    klds[:] = 0.0
    for h in reversed(range(len(noise_pats_raw))):
        hists[h], bin_centers = get_histogram(noise_pats_raw[h], bin_edges=bin_edges)
        # noinspection PyTypeChecker
        klds[h] = kl_div_forward(hists[-1], hists[h])

    savemat(os.path.join(subdir, '%s_%04d.mat' % ('kl_ng', pid)), {'x': klds[0]})
    savemat(os.path.join(subdir, '%s_%04d.mat' % ('kl_nl', pid)), {'x': klds[1]})
    savemat(os.path.join(subdir, '%s_%04d.mat' % ('kl_ns', pid)), {'x': klds[2]})

    klds_que.put(klds)


def plot_multi_patch(noise_pats, noisy_pats, clean_pat, subdir, pid, fn, hists, bin_centers, klds):
    titles = ['Gaussian', 'Camera NLF', 'Noise Flow', 'Real']
    # model_names = ['gauss', 'nlf', 'nf', 'real']
    f = plt.figure(figsize=[10, 4])
    f.suptitle(fn)
    u = 2
    n_models = len(noise_pats)
    v = n_models + 1
    gridspec.GridSpec(u, v)
    for s in range(u):
        for t in range(v):
            ax = plt.subplot2grid((u, v), (s, t), colspan=1, rowspan=1, fig=f)
            # plt.locator_params(axis='x', nbins=5)
            # plt.locator_params(axis='y', nbins=5)
            if s == 0 and t == v - 1:
                # pass
                for j in range(n_models):
                    ax.plot(bin_centers, hists[j], label='%s KL=%.3f' % (titles[j], klds[j]))
                ax.set_title('Noise Histograms')
                ax.legend(prop={'size': 6})
            elif s == u - 1 and t == v - 1:
                ax.imshow(clean_pat)
                ax.set_title('Clean')
                ax.axis('off')
            else:
                if s == 0:
                    ax.imshow(noise_pats[t])
                    ax.set_title(titles[t])
                    if t == 0:
                        ax.set_ylabel('Noise')
                else:
                    ax.imshow(noisy_pats[t])
                    if t == 0:
                        ax.set_ylabel('Noisy')
                ax.axis('off')
    f.tight_layout()
    plt.show()
    f.savefig(os.path.join(subdir, 'sample_%04d.png' % pid))
    plt.close(f)


def process_save_one_patch_que(args):
    que = args[-1]
    args = args[:-1]
    que.put(process_save_one_patch(args))


def process_save_one_patch(args):
    (pipe, patch, name, metadata, dir1, pat_id) = args
    srgb = pipe.process(patch, metadata, 'normal', 'gamma')
    srgb = np.uint8(srgb * 255.0)
    imageio.imwrite(os.path.join(dir1, '%s_%04d.png' % (name, pat_id)), srgb)
    return srgb


def load_visual_minibatch(hps):
    try:
        mb = np.load(os.path.join(hps.logdir, '..', 'vis_mb.npy')).item()
        # mb = np.load(os.path.join('experiments', hps.problem, 'vis_mb.npy')).item()
        meta = np.load(os.path.join(hps.logdir, '..', 'vis_mb_meta.npy'))
        # meta = np.load(os.path.join('experiments', hps.problem, 'vis_mb_meta.npy'))
        mb['metadata'] = meta
        return mb
    except:
        print('error loading visualization minibatch, filename: %s' % os.path.join(hps.logdir, '..', 'vis_mb.npy'))
        return None


def restore_epoch_model(ckpt_dir, sess, saver, epoch):
    model_checkpoint_path = os.path.join(ckpt_dir, 'model.ckpt-%d' % epoch)
    print('loading ' + model_checkpoint_path)
    # noinspection PyBroadException
    try:
        saver.restore(sess, model_checkpoint_path)
        return True
    except Exception as ex:
        print('failed to load model: %s' % model_checkpoint_path)
        print(str(ex))
        return False


def restore_best_model(ckpt_dir, sess, saver):
    model_checkpoint_path = os.path.join(ckpt_dir, 'model.ckpt-best')
    print('loading ' + model_checkpoint_path)
    # noinspection PyBroadException
    try:
        saver.restore(sess, model_checkpoint_path)
    except:
        print('failed to load model: %s' % model_checkpoint_path)


def restore_last_model(ckpt_dir, sess, saver):
    last_epoch = 0
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt:
        print('loading ' + ckpt.model_checkpoint_path)
        try:
            last_epoch = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            saver.restore(sess, ckpt.model_checkpoint_path)
        except:
            print('failed to load last model, starting from epoch 1')
    return last_epoch


def sample_sidd_tf(sess, flow_model, is_training, temp, y, nlf0, nlf1, iso, cam, is_cond):
    if is_cond:
        x_sample = flow_model.sample(y, temp, y, nlf0, nlf1, iso, cam)
    else:
        x_sample = flow_model.sample(y, temp)
    return x_sample


def sample_sidd(sess, flow_model, is_training, temp, x, y, nlf0, nlf1, iso, cam, is_cond):
    z = tf.zeros(x.shape)
    # obj = tf.zeros_like(z, dtype='float32')[:, 0, 0, 0]  # for inverse
    if is_cond:
        x = flow_model.sample(z, temp, y, nlf0, nlf1, iso, cam)
    else:
        x = flow_model.sample(z, temp)
    x_val = sess.run(x, feed_dict={is_training: False})
    return x_val


def copy_stats(hps):
    fns = ['pat_stats.npy', 'nll_bpd_gauss.npy', 'nll_bpd_sdn.npy']
    for fn in fns:
        src = os.path.join('experiments', hps.problem, fn)
        dst = os.path.join(hps.logdir, fn)
        copyfile(src, dst)


def im_normalize(im):
    nmin = np.min(im)
    nmax = np.max(im)
    return (im - nmin) / (nmax - nmin)


def im_shift(im):
    return im - np.min(im)


def kl_div_forward(p, q):
    idx = ~(np.isnan(p) | np.isinf(p) | np.isnan(q) | np.isinf(q))
    p = p[idx]
    q = q[idx]
    idx = (p > 0) & (q > 0)
    p = p[idx]
    q = q[idx]
    return np.sum(p * np.log(p / q))


def kl_div_inverse(p, q):
    idx = ~(np.isnan(p) | np.isinf(p) | np.isnan(q) | np.isinf(q))
    p = p[idx]
    q = q[idx]
    idx = (p > 0) & (q > 0)
    p = p[idx]
    q = q[idx]
    return np.sum(q * np.log(q / p))


def kl_div_sym(p, q):
    return (kl_div_forward(p, q) + kl_div_inverse(p, q)) / 2.0


def kl_div_3(p, q):
    kl_fwd = kl_div_forward(p, q)
    kl_inv = kl_div_inverse(p, q)
    kl_sym = (kl_inv + kl_fwd) / 2.0
    return kl_fwd, kl_inv, kl_sym


def kl_div_forward_data(p_data, q_data, left_edge=0.0, right_edge=1.0, n_bins=1000):
    """ Forward KL divergence between two sets of data points p and q"""
    p, _ = get_histogram(p_data, left_edge, right_edge, n_bins)
    q, _ = get_histogram(q_data, left_edge, right_edge, n_bins)
    return kl_div_forward(p, q)


def kl_div_inverse_data(p_data, q_data, left_edge=0.0, right_edge=1.0, n_bins=1000):
    """ Forward KL divergence between two sets of data points p and q"""
    p, _ = get_histogram(p_data, left_edge, right_edge, n_bins)
    q, _ = get_histogram(q_data, left_edge, right_edge, n_bins)
    return kl_div_inverse(p, q)


def kl_div_3_data(p_data, q_data, bin_edges=None, left_edge=0.0, right_edge=1.0, n_bins=1000):
    """Returns forward, inverse, and symmetric KL divergence between two sets of data points p and q"""
    if bin_edges is None:
        data_range = right_edge - left_edge
        bin_width = data_range / n_bins
        bin_edges = np.arange(left_edge, right_edge + bin_width, bin_width)
    p, _ = get_histogram(p_data, bin_edges, left_edge, right_edge, n_bins)
    q, _ = get_histogram(q_data, bin_edges, left_edge, right_edge, n_bins)
    idx = (p > 0) & (q > 0)
    p = p[idx]
    q = q[idx]
    logp = np.log(p)
    logq = np.log(q)
    kl_fwd = np.sum(p * (logp - logq))
    kl_inv = np.sum(q * (logq - logp))
    kl_sym = (kl_fwd + kl_inv) / 2.0
    return kl_fwd, kl_inv, kl_sym


def get_histogram(data, bin_edges=None, left_edge=0.0, right_edge=1.0, n_bins=1000):
    data_range = right_edge - left_edge
    bin_width = data_range / n_bins
    if bin_edges is None:
        bin_edges = np.arange(left_edge, right_edge + bin_width, bin_width)
    bin_centers = bin_edges[:-1] + (bin_width / 2.0)
    n = np.prod(data.shape)
    hist, _ = np.histogram(data, bin_edges)
    return hist / n, bin_centers


def add_gaussian_noise(im, sd):
    return im + np.random.normal(0, sd, im.shape)


def calc_kld_avgs():
    root = '/home/abdo/skynet/_Code/fourier_flows/experiments/sidd/190130/UncSdnUncGainUnc_s/'
    ids = [1, 2, 3, 5, 7]
    epcs = [1000]
    pat_ids = range(0, 322, 10)
    n = 4

    kld_avgs = [0] * n

    for id in ids:
        for epc in epcs:
            cnt = 0
            for p in pat_ids:
                fn = os.path.join(root, 'samples%d' % id, 'epoch_%04d' % epc, 'klds_%04d.npy' % p)
                print(fn)
                kld_avgs += np.load(fn)
                cnt += 1
            kld_avgs /= cnt
            fn = os.path.join(root, 'samples%d' % id, 'epoch_%04d' % epc, 'kld_avgs.txt')
            np.savetxt(fn, kld_avgs)


def get_avg_klds(temp_ids=[1, 2, 3, 5, 7]):
    root = '/home/abdo/skynet/_Code/fourier_flows/experiments/sidd/190130/UncSdnUncGainUnc_s/'
    # temp_ids = [1, 2, 3, 5, 7]
    epcs = [1000]
    # pat_ids = range(0, 322, 10)
    n = 4

    kld_avgs = np.ndarray([len(temp_ids), n])

    for i, id in enumerate(temp_ids):
        for epc in epcs:
            fn = os.path.join(root, 'samples%d' % id, 'epoch_%04d' % epc, 'kld_avgs.txt')
            klds = np.loadtxt(fn)
            # kld_avgs.append(klds)
            kld_avgs[i, :] = klds
    return kld_avgs


def plot_klds_vs_temp(klds, temps):
    root = '/home/abdo/skynet/_Code/fourier_flows/experiments/sidd/190130/UncSdnUncGainUnc_s/'
    titles = ['Gaussian', 'Camera NLF', 'Noise Flow', 'Real']
    temps = np.asarray(temps)
    fig = plt.figure()
    for i in range(4):
        plt.plot(temps / 10.0, klds[:, i], label=titles[i])
    plt.xlabel('Sampling temperature')
    plt.ylabel('Average KL divergence KL(sampled, real)')
    plt.legend()
    fig.savefig(os.path.join(root, 'kld_vs_temp.png'))


def copy_one_sample(sample_id, last_epoch):
    epochs = np.concatenate((np.asarray(range(1, 11)), np.asarray(range(20, last_epoch + 1, 10))))
    root = '/home/abdo/skynet/_Code/fourier_flows/experiments/sidd/UNCOND/'
    id = ''
    tgt_dir = os.path.join(root, 'one_sample_%04d' % sample_id)
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir, exist_ok=True)
    for epc in epochs:
        fn = os.path.join(root, 'samples%s' % id, 'epoch_%04d' % epc, 'sample_%04d.png' % sample_id)
        fn2 = os.path.join(tgt_dir, 'sample_%04d_epoch_%04d.png' % (sample_id, epc))
        copyfile(fn, fn2)


def create_gif(im_dir):
    root = '/home/abdo/skynet/_Code/fourier_flows/experiments/sidd/UNCOND/'
    sample_id = 50
    im_dir = os.path.join(root, 'one_sample_%04d' % sample_id)
    filenames = glob.glob(path.join(im_dir, '*.png'))
    filenames = sorted(filenames)
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(os.path.join(im_dir, 'sample_%04d.gif' % sample_id), images, duration=1)


def divide_parts(n, n_parts):
    """divide a number into a list of parts"""
    (div, rem) = divmod(n, n_parts)
    divs = [div] * n_parts
    if rem != 0:
        for r in range(rem):
            divs[r] += 1
    return divs


def divide_array_parts(n, n_parts):
    """divide an array into parts"""
    divs = divide_parts(n, n_parts)
    arr = np.asarray(range(n))
    parts = [None] * n_parts
    idx = 0
    for i in range(n_parts):
        parts[i] = arr[idx: idx + divs[i]]
        idx += divs[i]
    return parts
