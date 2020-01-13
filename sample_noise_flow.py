import glob
import logging
import os

import cv2
from scipy.io import savemat

from borealisflows.NoiseFlowWrapper import NoiseFlowWrapper
from mylogger import add_logging_level
import sidd.data_loader as loader
import pandas as pd
import numpy as np

from sidd.data_loader import check_download_sidd
from sidd.pipeline import process_sidd_image
from sidd.raw_utils import read_metadata
from sidd.sidd_utils import unpack_raw

data_dir = 'data'
sidd_path = os.path.join(data_dir, 'SIDD_Medium_Raw/Data')
nf_model_path = 'models/NoiseFlow'

samples_dir = os.path.join(data_dir, 'samples')
os.makedirs(samples_dir, exist_ok=True)


def main():

    # Download SIDD_Medium_Raw?
    check_download_sidd()

    # set up a custom logger
    add_logging_level('TRACE', 100)
    logging.getLogger(__name__).setLevel("TRACE")
    logging.basicConfig(level=logging.TRACE)

    # Prepare NoiseFlow
    noise_flow = NoiseFlowWrapper(nf_model_path)

    # sample noise and add it to clean images
    patch_size = 32
    batch_size = 1  # using batches is faster
    for sc_id in [10, 52, 64]:  # scene IDs

        # load images
        noisy = loader.load_raw_image_packed(glob.glob(os.path.join(sidd_path, '%04d_*' % sc_id, '*NOISY_RAW_010.MAT'))[0])
        clean = loader.load_raw_image_packed(glob.glob(os.path.join(sidd_path, '%04d_*' % sc_id, '*GT_RAW_010.MAT'))[0])
        metadata, bayer_2by2, wb, cst2, iso, cam = read_metadata(
            glob.glob(os.path.join(sidd_path, '%04d_*' % sc_id, '*METADATA_RAW_010.MAT'))[0])

        if iso not in [100, 400, 800, 1600, 3200]:
            continue

        n_pat = 10
        for p in range(n_pat):

            # crop patches
            v = np.random.randint(0, clean.shape[1] - patch_size)
            u = np.random.randint(0, clean.shape[2] - patch_size)

            clean_patch = clean[0, v:v + patch_size, u:u + patch_size, :]
            noisy_patch = noisy[0, v:v + patch_size, u:u + patch_size, :]

            clean_patch = np.expand_dims(clean_patch, 0)

            # sample noise
            noise_patch_syn = noise_flow.sample_noise_nf(clean_patch, 0.0, 0.0, iso, cam)

            noise_patch_syn = np.squeeze(noise_patch_syn)[1:-1, 1:-1, :]
            clean_patch = np.squeeze(clean_patch)[1:-1, 1:-1, :]
            noisy_patch_syn = unpack_raw(np.clip(clean_patch + noise_patch_syn, 0.0, 1.0))
            clean_patch = unpack_raw(clean_patch)
            noisy_patch = unpack_raw(np.squeeze(noisy_patch)[1:-1, 1:-1, :])

            # process
            clean_patch_srgb = process_sidd_image(clean_patch, bayer_2by2, wb, cst2)
            noisy_patch_srgb = process_sidd_image(noisy_patch, bayer_2by2, wb, cst2)
            noisy_patch_syn_srgb = process_sidd_image(noisy_patch_syn, bayer_2by2, wb, cst2)
            conc_im = np.concatenate([clean_patch_srgb, noisy_patch_srgb, noisy_patch_syn_srgb], axis=1)
            conc_height, conc_width, _ = conc_im.shape

            # save as .png
            scale = 16
            cv2.resize(conc_im, (conc_width * scale, conc_height * scale), interpolation=cv2.INTER_NEAREST)
            save_fn = os.path.join(samples_dir, '%02d_%02d_%04d.png' % (sc_id, p, iso))
            cv2.imwrite(save_fn, conc_im)

            # save as .mat
            save_mat_fn = os.path.join(samples_dir, '%02d_%02d_%04d.mat' % (sc_id, p, iso))
            savemat(save_mat_fn, {'clean': clean_patch, 'noisy': noisy_patch, 'noisy_syn': noisy_patch_syn,
                                  'metadata': metadata})


def load_cam_iso_nlf():
    cin = pd.read_csv('cam_iso_nlf.txt')
    cin = cin.drop_duplicates()
    cin = cin.set_index('cam_iso', drop=False)
    return cin


if __name__ == '__main__':
    main()


