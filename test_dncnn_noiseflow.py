# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import datetime
import os
import time
import urllib.request

import numpy as np
from keras.models import load_model, model_from_json
from numpy import savetxt
from scipy.io import loadmat, savemat
from skimage.io import imsave

from sidd.data_loader import pack_raw, unpack_raw

from skimage.measure import compare_ssim as ssim


def get_best_model(model_dir):
    max_file = os.path.join(model_dir, 'max_epc_psnr.txt')
    max_epc_psnr = np.loadtxt(max_file)
    max_epc = max_epc_psnr[0]
    model_path = os.path.join(model_dir, 'model_%03d.hdf5' % max_epc)
    model = load_model(model_path, compile=False)
    return model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--set_dir', default='data', type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default=['SIDD'], type=list, help='names of test datasets')
    parser.add_argument('--model_name', type=str, help='model name (e.g., DnCNN_Gauss)')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of results')
    parser.add_argument('--save_result', action='store_true', help='whether to save the denoised images')
    parser.add_argument('--min_epc', default=1, type=int, help='start epoch for testing')
    parser.add_argument('--max_epc', default=2000, type=int, help='end epoch for testing')
    parser.add_argument('--epc_step', default=1, type=int, help='epoch step for testing')

    return parser.parse_args()


def output_psnr_mse(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr


def mean_psnr_raw(ref_mat, res_mat):
    n_im, n_blk, h, w = ref_mat.shape
    mean_psnr = 0
    psnrs = np.ndarray([n_im, n_blk]);
    for i in range(n_im):
        for b in range(n_blk):
            ref_block = ref_mat[i, b, :, :]
            res_block = res_mat[i, b, :, :]
            psnr = output_psnr_mse(ref_block, res_block)
            mean_psnr += psnr
            psnrs[i, b] = psnr
    return mean_psnr / (n_im * n_blk), psnrs


def mean_ssim_raw(ref_mat, res_mat):
    n_im, n_blk, h, w = ref_mat.shape
    mean_ssim = 0
    for i in range(n_im):
        for b in range(n_blk):
            ref_block = ref_mat[i, b, :, :]
            res_block = res_mat[i, b, :, :]
            ref_block = np.reshape(ref_block, (h, w))
            res_block = np.reshape(res_block, (h, w))
            ssim1 = ssim(ref_block, res_block, gaussian_weights=True, use_sample_covariance=False)
            # ssim1 = ssim(ref_block, res_block, max_val=1.0)
            mean_ssim += ssim1
    return mean_ssim / (n_im * n_blk)


def to_tensor(img):
    if img.ndim == 2:
        return img[np.newaxis, ..., np.newaxis]
    elif img.ndim == 3:
        return np.moveaxis(img, 2, 0)[..., np.newaxis]


def from_tensor(img):
    return np.squeeze(np.moveaxis(img[..., 0], 0, -1))


def log(*args1, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S "), *args1, **kwargs)


def save_result(result, path):
    path = path if path.find('.') != -1 else path + '.png'
    ext1 = os.path.splitext(path)[-1]
    if ext1 in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        imsave(path, np.clip(result, 0, 1))


def show(x1, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x1, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def download_url_to_file(url, file):
    file_data = urllib.request.urlopen(url)
    data_to_write = file_data.read()
    with open(file, 'wb') as f:
        f.write(data_to_write)


def get_testing_data():
    noisy_mat_path = os.path.join(args.set_dir, 'ValidationNoisyBlocksRaw.mat')
    ref_mat_path = os.path.join(args.set_dir, 'ValidationGtBlocksRaw.mat')

    # download?
    if not os.path.exists(noisy_mat_path):
        noisy_mat_url = 'ftp://sidd_user:sidd_2018@130.63.97.225/SIDD_Blocks/ValidationNoisyBlocksRaw.mat'
        print('downloading ' + noisy_mat_url)
        print('to ' + ref_mat_path)
        download_url_to_file(noisy_mat_url, noisy_mat_path)
    if not os.path.exists(ref_mat_path):
        ref_mat_url = 'ftp://sidd_user:sidd_2018@130.63.97.225/SIDD_Blocks/ValidationGtBlocksRaw.mat'
        print('downloading ' + ref_mat_url)
        print('to ' + ref_mat_path)
        download_url_to_file(ref_mat_url, ref_mat_path)

    noisy_mat1 = loadmat(noisy_mat_path)['ValidationNoisyBlocksRaw']
    ref_mat1 = loadmat(ref_mat_path)['ValidationGtBlocksRaw']
    exc_iso = [1, 3, 5, 7, 10, 11, 13, 14, 15, 18, 19, 20, 23, 24, 25, 28, 31, 33, 35, 38]
    noisy_mat1 = np.delete(noisy_mat1, exc_iso, axis=0)
    ref_mat1 = np.delete(ref_mat1, exc_iso, axis=0)

    return noisy_mat1, ref_mat1


if __name__ == '__main__':
    tt = time.time()
    args = parse_args()
    args.model_dir = os.path.join('models', args.model_name)
    print('args.model_dir = %s' % args.model_dir)
    args.epochs = np.asarray(range(args.min_epc, args.max_epc + 1, args.epc_step))  # include max_epc
    print('args.save_result = ' + str(args.save_result))

    epc_psnr = np.ndarray([len(args.epochs), 2])

    noisy_mat, ref_mat = get_testing_data()

    n_im, n_pt, pt_h, pt_w = noisy_mat.shape
    print('*** n_im = %d' % n_im)
    print('loaded noisy and reference images')

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    subdir = os.path.join(args.result_dir, args.model_name)
    if not os.path.exists(subdir):
        os.mkdir(subdir)



    for k, epc in enumerate(args.epochs):

        args.model_ckpt = 'model_%03d.hdf5' % epc
        mod_pth = os.path.join(args.model_dir, args.model_ckpt)
        if not os.path.exists(mod_pth):
            # load json and create model
            json_file = open(os.path.join(args.model_dir, 'model.json'), 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights(os.path.join(args.model_dir, 'model.h5'))
            log('load trained model on Train400 dataset by kai')
        else:
            log('load trained model %s' % mod_pth)
            model = load_model(mod_pth, compile=False)
            
        for set_cur in args.set_names:

            out_dir = os.path.join(args.result_dir, args.model_name, set_cur)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)

            res_file = open(os.path.join(out_dir, 'epc_psnr.txt'), 'w', buffering=1)
            res_file.write('epoch,psnr\n')

            res_mat = np.zeros(noisy_mat.shape)

            for i in range(n_im):
                for p in range(n_pt):
                    noisy_patch = np.squeeze(noisy_mat[i, p, :, :])
                    noisy_patch = pack_raw(noisy_patch)
                    noisy_patch = noisy_patch[np.newaxis, :, :, :]
                    denoised_patch = model.predict(noisy_patch)  # inference
                    denoised_patch = unpack_raw(np.squeeze(denoised_patch))
                    res_mat[i, p, :, :] = denoised_patch

            mean_psnr, psnrs = mean_psnr_raw(ref_mat, res_mat)
            mean_ssim = mean_ssim_raw(ref_mat, res_mat)
            epc_psnr[k, :] = [epc, mean_psnr]

            if args.save_result:
                # insert epoch number into file name in order to save each result separately
                save_file = os.path.join(args.result_dir, args.model_name, 'results.mat')
                log('saving results: ' + save_file)
                savemat(save_file, {'results': res_mat})
            savetxt(os.path.join(out_dir, 'psnr.txt'), [mean_psnr])
            savetxt(os.path.join(out_dir, 'ssim.txt'), [mean_ssim])
            psnrs_file = os.path.join(out_dir, 'psnrs.mat')
            savemat(psnrs_file, {'psnrs': psnrs})
            res_file.write('%s,%s\n' % (str(epc), str(mean_psnr)))

            log('Datset: {0:10s} \t  PSNR = {1:2.2f}dB'.format(set_cur, mean_psnr))

            res_file.close()

    # plot___

    tt = time.time() - tt
    print('total time = %s' % str(tt))
