import cv2
import numpy as np

from sidd.img_utils import swap_channels
from sidd.raw_utils import (demosaic_CV2, flip_bayer,
                            stack_rggb_channels)


def process_sidd_image(image, bayer_pattern, wb, cst, *, save_file_rgb=None):
    """Simple processing pipeline"""

    image = flip_bayer(image, bayer_pattern)

    image = stack_rggb_channels(image)

    rgb2xyz = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )

    rgb2cam = np.matmul(cst, rgb2xyz)

    cam2rgb = np.linalg.inv(rgb2cam)

    cam2rgb = cam2rgb / np.sum(cam2rgb, axis=-1, keepdims=True)
    image_srgb = process(image, 1 / wb[0][0], 1 / wb[0][1], 1 / wb[0][2], cam2rgb)

    image_srgb = image_srgb * 255.0
    image_srgb = image_srgb.astype(np.uint8)

    image_srgb = swap_channels(image_srgb)

    if save_file_rgb:
        # Save
        cv2.imwrite(save_file_rgb, image_srgb)

    return image_srgb


def apply_gains(bayer_image, red_gains, green_gains, blue_gains):
    gains = np.stack([red_gains, green_gains, green_gains, blue_gains], axis=-1)
    gains = gains[np.newaxis, np.newaxis, :]
    return bayer_image * gains


def demosaic_simple(rggb_channels_stack):
    channels_rgb = rggb_channels_stack[:, :, :3]
    channels_rgb[:, :, 0] = channels_rgb[:, :, 0]
    channels_rgb[:, :, 1] = np.mean(rggb_channels_stack[:, :, 1:3], axis=2)
    channels_rgb[:, :, 2] = rggb_channels_stack[:, :, 3]
    return channels_rgb


def apply_ccm(image, ccm):
    images = image[:, :, np.newaxis, :]
    ccms = ccm[np.newaxis, np.newaxis, :, :]
    return np.sum(images * ccms, axis=-1)


def gamma_compression(images, gamma=2.2):
    return np.maximum(images, 1e-8) ** (1.0 / gamma)


def process(bayer_images, red_gains, green_gains, blue_gains, cam2rgbs):
    bayer_images = apply_gains(bayer_images, red_gains, green_gains, blue_gains)
    bayer_images = np.clip(bayer_images, 0.0, 1.0)
    images = demosaic_CV2(bayer_images)
    images = apply_ccm(images, cam2rgbs)
    images = np.clip(images, 0.0, 1.0)
    images = gamma_compression(images)
    return images
