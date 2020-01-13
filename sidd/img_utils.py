import random

import cv2
import numpy as np


def psnr(img1, img2):
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def is_large_enough(image, height, width):
    """Checks if `image` is at least as large as `height` by `width`."""
    shape = image.shape
    image_height = shape[0]
    image_width = shape[1]
    return image_height >= height and image_width >= width


def augment(image, height, width):
    """Randomly flips and crops `images` to `height` by `width`."""
    y = random.randint(0, image.shape[0] - height)
    x = random.randint(0, image.shape[1] - width)
    image = image[y:y + height, x:x + width]
    if random.choice([True, False]):
        image = np.fliplr(image)
    if random.choice([True, False]):
        image = np.flipud(image)
    return image


def swap_dims(image):
    """CxHxW --> HxWxC"""
    c, h, w = image.shape
    image1 = np.zeros((h, w, c))
    for i in range(c):
        # image1[:, :, i] = image[i, :, :]
        image1[:, :, c - i - 1] = image[i, :, :]
    return image1


def swap_channels(image):
    """Swap the order of channels: RGB --> BGR"""
    h, w, c = image.shape
    image1 = np.zeros(image.shape)
    for i in range(c):
        image1[:, :, i] = image[:, :, c - i - 1]
    return image1


def pad_image(input_image, multiple):
    """Pad image so that height and width are multiples of `multiple`"""
    padh = 0
    if (input_image.shape[0] % multiple) != 0:
        padh = multiple - (input_image.shape[0] % multiple)
    padw = 0
    if (input_image.shape[1] % multiple) != 0:
        padw = multiple - (input_image.shape[1] % multiple)

    if padh != 0:
        input_image = cv2.copyMakeBorder(
            input_image,
            top=0,
            bottom=padh,
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
    if padw != 0:
        input_image = cv2.copyMakeBorder(
            input_image,
            top=0,
            bottom=0,
            left=0,
            right=padw,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
    return input_image, padh, padw


def unpad_image(im, padh, padw):
    if padh != 0:
        im = im[:-padh, :, :]
    if padw != 0:
        im = im[:, :-padw, :]
    return im
