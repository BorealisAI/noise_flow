# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import glob
import logging
import os
import queue
import time
from threading import Thread

# import cv2
import h5py
import numpy as np
import pandas as pd


patch_size, stride = 32, 64  # patch size  = [32, 32, 4]
aug_times = 1
scales = [1]  # [1, 0.9, 0.8, 0.7]
batch_size = 128


def data_aug(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def pack_raw(raw_im):
    """Packs Bayer image to 4 channels (h, w) --> (h/2, w/2, 4)."""
    # pack Bayer image to 4 channels
    im = np.expand_dims(raw_im, axis=2)
    img_shape = im.shape
    h = img_shape[0]
    w = img_shape[1]
    out = np.concatenate((im[0:h:2, 0:w:2, :],
                          im[0:h:2, 1:w:2, :],
                          im[1:h:2, 1:w:2, :],
                          im[1:h:2, 0:w:2, :]), axis=2)
    return out


def unpack_raw(raw4ch):
    """Unpacks 4 channels to Bayer image (h/2, w/2, 4) --> (h, w)."""
    img_shape = raw4ch.shape
    h = img_shape[0]
    w = img_shape[1]
    bayer = np.zeros([h * 2, w * 2], dtype=np.float32)
    bayer[0::2, 0::2] = raw4ch[:, :, 0]
    bayer[0::2, 1::2] = raw4ch[:, :, 1]
    bayer[1::2, 1::2] = raw4ch[:, :, 2]
    bayer[1::2, 0::2] = raw4ch[:, :, 3]
    return bayer


def load_raw_image_packed(im_file):
    """Loads and returns a normalized packed raw-RGB image from .mat file (im_file) with dimensions (1, ?, ?, 4)"""
    with h5py.File(im_file, 'r') as f:  # (use this for .mat files with -v7.3 format)
        raw = f[list(f.keys())[0]]  # use the first and only key
        # input_image = np.transpose(raw)  # transpose?
        raw = np.expand_dims(pack_raw(raw), axis=0)
        raw = np.nan_to_num(raw)
        raw = np.clip(raw, 0.0, 1.0)
    return raw


def gen_patches_thread(file_name, out_que):
    img = load_raw_image_packed(file_name)
    dim1, h, w, c = img.shape
    patches = None
    cam_iso = file_name[-41:-33]
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            x = img[0, i:i + patch_size, j:j + patch_size, :]  # first dim will be removed
            # data aug
            for k in range(0, aug_times):
                # x_aug = data_aug(x, mode=np.random.randint(0, 8))
                if patches is None:
                    patches = x[np.newaxis, :, :, :]  # restore first dim
                else:
                    patches = np.concatenate((patches, x[np.newaxis, :, :, :]), axis=0)  # patches.append(x_aug)
    cam_iso_patches = [cam_iso] * len(patches)
    out_que.put((patches, cam_iso_patches))


def gen_patches(file_name):
    # read image
    img = load_raw_image_packed(file_name)
    dim1, h, w, c = img.shape
    patches = None
    # extract patches
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            x = img[0, i:i + patch_size, j:j + patch_size, :]  # first dim will be removed
            # data aug
            for k in range(0, aug_times):
                # x_aug = data_aug(x, mode=np.random.randint(0, 8))
                if patches is None:
                    patches = x[np.newaxis, :, :, :]  # restore first dim
                else:
                    patches = np.concatenate((patches, x[np.newaxis, :, :, :]), axis=0)  # patches.append(x_aug)
    return patches


def load_data(data_dir='/shared-data/SIDD_Medium_Raw/Data/', verbose=False):
    file_list = glob.glob(os.path.join(data_dir, '*/*GT_RAW_010.MAT'))  # get name list of all GT .mat files
    # initialize
    data1 = None
    cam_iso_info = []
    # generate patches
    tt = time.time()
    for i in range(len(file_list)):
        cam_iso = file_list[i][-41:-33]
        patch = gen_patches(file_list[i])
        cam_iso_info.append([cam_iso] * len(patch))
        if data1 is None:
            data1 = patch
        else:
            data1 = np.concatenate((data1, patch), axis=0)
        if verbose:
            print(str(i + 1) + '/' + str(len(file_list)) + ' is done ^_^')
    assert len(cam_iso_info) == len(data1)
    tt = time.time() - tt
    discard_n = len(data1) - len(data1) // batch_size * batch_size
    data1 = np.delete(data1, range(discard_n), axis=0)
    cam_iso_info = cam_iso_info[discard_n:]
    assert len(cam_iso_info) == len(data1)
    print('^_^-training data finished-^_^ time = %s sec' % str(tt))
    return data1, cam_iso_info


# filter train set
def load_cam_iso_nlf():
    cin = pd.read_csv('cam_iso_nlf.txt')
    cin = cin.drop_duplicates()
    cin = cin.set_index('cam_iso', drop=False)
    return cin


def load_data_threads(data_dir, max_images=0, verbose=False):
    # data_dir='/shared-data/SIDD_Medium_Raw/Data/'
    file_list = glob.glob(os.path.join(data_dir, '**', '*GT_RAW_010.MAT'))  # get name list of GT .mat files
    if max_images != 0:
        file_list = file_list[:max_images]
    print('# images pre-filter = %d' % len(file_list))

    cam_iso_nlf = load_cam_iso_nlf()
    cam_iso_vals = cam_iso_nlf['cam_iso']
    file_list_copy = []
    for f in file_list:
        if f[-41:-33] in cam_iso_vals:
            file_list_copy.append(f)
    file_list = file_list_copy
    print('# images post-filter = %d' % len(file_list))

    # initialize
    data1 = None
    cam_iso_info = []
    # data_que = queue.Queue()
    # generate patches
    tt = time.time()
    threads = [None] * len(file_list)
    ques = [None] * len(file_list)
    for i in range(len(file_list)):
        # patch = gen_patches(file_list[i])
        ques[i] = queue.Queue(1)
        threads[i] = Thread(target=gen_patches_thread, args=(file_list[i], ques[i]))
        threads[i].start()
    for i in range(len(file_list)):
        threads[i].join()
    # assert data_que.qsize() == len(file_list)
    for i in range(len(file_list)):
        # patches = data_que.get()
        (patches, cam_iso_patches) = ques[i].get()
        if data1 is None:
            data1 = patches
        else:
            data1 = np.concatenate((data1, patches), axis=0)
        cam_iso_info = cam_iso_info + cam_iso_patches
        if verbose:
            print(str(i + 1) + '/' + str(len(file_list)) + ' is done ^_^')
    discard_n = len(data1) - len(data1) // batch_size * batch_size
    data1 = np.delete(data1, range(discard_n), axis=0)
    cam_iso_info = cam_iso_info[discard_n:]
    assert len(cam_iso_info) == len(data1)
    tt = time.time() - tt
    print('loading data finished, time = %s sec' % str(tt))
    return data1, cam_iso_info


def load_data_threads_with_noisy(data_dir, verbose=False):
    # data_dir='/shared-data/SIDD_Medium_Raw/Data/'
    file_list = glob.glob(os.path.join(data_dir, '**', '*GT_RAW_010.MAT'))  # get name list of all GT .mat files
    # get name list of all noisy .mat files
    file_list_noisy = glob.glob(os.path.join(data_dir, '**', '*NOISY_RAW_010.MAT'))
    for i in range(len(file_list)):
        assert file_list[i][-19:-15] == file_list_noisy[i][-22:-18]  # same scene ID
    print('# images pre-filter = %d' % len(file_list))

    cam_iso_nlf = load_cam_iso_nlf()
    cam_iso_vals = cam_iso_nlf['cam_iso']
    file_list_copy = []
    file_list_noisy_copy = []
    for i, f in enumerate(file_list):
        if f[-41:-33] in cam_iso_vals:
            file_list_copy.append(f)
            file_list_noisy_copy.append(file_list_noisy[i])
    file_list = file_list_copy
    file_list_noisy = file_list_noisy_copy
    print('# images post-filter = %d' % len(file_list))

    # initialize
    data1 = None
    data1_noisy = None
    cam_iso_info = []
    # data_que = queue.Queue()
    # generate patches
    tt = time.time()
    threads = [None] * len(file_list)
    threads_noisy = [None] * len(file_list)
    ques = [None] * len(file_list)
    ques_noisy = [None] * len(file_list)
    for i in range(len(file_list)):
        # patch = gen_patches(file_list[i])
        ques[i] = queue.Queue(1)
        threads[i] = Thread(target=gen_patches_thread, args=(file_list[i], ques[i]))
        threads[i].start()

        ques_noisy[i] = queue.Queue(1)
        threads_noisy[i] = Thread(target=gen_patches_thread, args=(file_list_noisy[i], ques_noisy[i]))
        threads_noisy[i].start()

    for i in range(len(file_list)):
        threads[i].join()
        threads_noisy[i].join()

    # assert data_que.qsize() == len(file_list)
    for i in range(len(file_list)):
        # patches = data_que.get()
        (patches, cam_iso_patches) = ques[i].get()
        (patches_noisy, _) = ques_noisy[i].get()
        if data1 is None:
            data1 = patches
            data1_noisy = patches_noisy
        else:
            data1 = np.concatenate((data1, patches), axis=0)
            data1_noisy = np.concatenate((data1_noisy, patches_noisy), axis=0)

        cam_iso_info = cam_iso_info + cam_iso_patches
        if verbose:
            print(str(i + 1) + '/' + str(len(file_list)) + ' is done ^_^')
    discard_n = len(data1) - len(data1) // batch_size * batch_size
    data1 = np.delete(data1, range(discard_n), axis=0)
    data1_noisy = np.delete(data1_noisy, range(discard_n), axis=0)
    cam_iso_info = cam_iso_info[discard_n:]
    assert len(cam_iso_info) == len(data1)
    assert len(cam_iso_info) == len(data1_noisy)
    tt = time.time() - tt
    print('data loading finished, time = %s sec' % str(tt))
    return data1, cam_iso_info, data1_noisy
