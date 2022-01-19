# -*- coding: utf-8 -*-

import sys
import imageio
import numpy as np
from PIL import Image

try:
    from skimage.measure import compare_psnr

    def eval_fn(target, reco):
        return compare_psnr(target, reco, 255)


except ImportError:
    from skimage.metrics import peak_signal_noise_ratio

    def eval_fn(target, reco):
        return peak_signal_noise_ratio(target, reco, data_range=255)


class stdout_logger(object):
    """Redirect print statements both to console and file

    Source: https://stackoverflow.com/a/14906787

    Licensed under the Academic Free License version 3.0
    """

    def __init__(self, txt_file):
        self.terminal = sys.stdout
        self.log = open(txt_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def get_image(image_file, rescale=None):
    """Read image from file, optionally rescale and return as numpy array

    :param image_file: Full path to image
    :type image_file: str
    :param rescale: If provided, the image height and width will be rescaled
                    by this factor
    :type rescale: float
    :return: Image array
    :type return: np.ndarray

    Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
    Licensed under the Academic Free License version 3.0
    """
    img = imageio.imread(image_file)
    isrgb = np.ndim(img) == 3 and img.shape[2] == 3
    isgrey = np.ndim(img) == 2
    assert isrgb or isgrey, "Expect img image to be either RGB or grey"
    if rescale is not None:
        orig_shape = img.shape
        target_shape = [int(orig_shape[1] * rescale), int(orig_shape[0] * rescale)]
        img = (
            np.asarray(
                [
                    np.asarray(
                        Image.fromarray(img[:, :, ch]).resize(target_shape, resample=Image.NEAREST),
                        dtype=np.float64,
                    )
                    for ch in range(3)
                ]
            ).transpose(1, 2, 0)
            if isrgb
            else np.asarray(
                Image.fromarray(img).resize(target_shape, resample=Image.NEAREST), dtype=np.float64
            )
        )
        print("Resized input image from {}->{}".format(orig_shape, np.asarray(img).shape))
        return img
    else:
        return np.asarray(img, dtype=np.float64)
