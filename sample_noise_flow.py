import logging
from borealisflows.NoiseFlowWrapper import NoiseFlowWrapper
from mylogger import add_logging_level
import sidd.data_loader as loader
import pandas as pd
import numpy as np
import random


def main():

    # TODO: change these parameters if needed
    nf_model_path = 'models/NoiseFlow'
    sidd_path = '/shared-data/SIDD_Medium_Raw/Data'

    # set up a custom logger
    add_logging_level('TRACE', 100)
    logging.getLogger(__name__).setLevel("TRACE")
    logging.basicConfig(level=logging.TRACE)

    # Prepare NoiseFlow
    noise_flow = NoiseFlowWrapper(nf_model_path)

    # load data
    clean_images, cam_iso_info = loader.load_data_threads(sidd_path, max_images=1)

    # camera IDs and ISO levels related to the SIDD dataset
    cam_iso_nlf = load_cam_iso_nlf()
    n_cam_iso = cam_iso_nlf['cam_iso'].count()
    iso_vals = [100.0, 400.0, 800.0, 1600.0, 3200.0]
    cam_ids = [0, 1, 3, 3, 4]  # IP, GP, S6, N6, G4
    cam_vals = ['IP', 'GP', 'S6', 'N6', 'G4']

    # sample noise and add it to clean images
    batch_size = 128  # using batches is faster
    indices = list(range(clean_images.shape[0]))
    for i in range(0, len(indices), batch_size):

        # clean images
        batch_clean = clean_images[indices[i:i + batch_size]]
        batch_info = cam_iso_info[i:i + batch_size]

        # ISO and camera ID
        cam_iso_idx = random.randint(0, n_cam_iso - 1)
        row = cam_iso_nlf.iloc[cam_iso_idx]
        cam = cam_vals.index(row['cam_iso'][:2])
        iso = float(row['cam_iso'][3:])

        # sample noise
        batch_noise = noise_flow.sample_noise_nf(batch_clean, 0.0, 0.0, iso, cam)

        # add to clean images (do not forget to clip)
        batch_noisy = batch_clean + batch_noise
        batch_noisy = np.clip(batch_noisy, 0.0, 1.0)

        # TODO: save or feed to another module
        # ...


def load_cam_iso_nlf():
    cin = pd.read_csv('cam_iso_nlf.txt')
    cin = cin.drop_duplicates()
    cin = cin.set_index('cam_iso', drop=False)
    return cin


if __name__ == '__main__':
    main()


