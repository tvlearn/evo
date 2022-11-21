# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import os
import PIL
import h5py
import imageio
import argparse
import tifffile
import numpy as np


def _get_args():
    p = argparse.ArgumentParser(
        description="Read image from file, add Gaussian noise and write original and noisy image "
        "into H5 file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "image_file",
        type=str,
        help="Clean image file path",
    )

    p.add_argument(
        "sigma",
        type=int,
        help="Standard deviation of the additive white Gaussian noise",
    )

    p.add_argument(
        "rescale",
        type=float,
        help="If specified, the size of the clean image will be rescaled by this factor "
        "(only for demonstration purposes to minimize computational effort)",
    )

    p.add_argument(
        "--output_directory",
        type=str,
        help="Directory to write H5 file to (same as image file directory if not specified)",
        default=None,
    )

    return p.parse_args()


def _rescale(img, factor):
    """Rescale image in size

    :param img: Original size image
    :type img: np.ndarray
    :param factor: Factor by which to scale image height and width
    :type factor: float
    :return: Rescaled size image
    :type return: np.ndarray

    Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
    Licensed under the Academic Free License version 3.0
    """
    ndim = np.ndim(img)
    assert ndim == 2 or ndim == 3, "Expect input with dimensions either "
    "(height, width), (height, width, no_channels)"
    no_channels = 1 if ndim == 2 else img.shape[-1]

    orig_shape = img.shape
    target_shape = [int(orig_shape[1] * factor), int(orig_shape[0] * factor)]

    img = (
        np.asarray(
            [
                np.asarray(
                    PIL.Image.fromarray(img[:, :, ch]).resize(
                        target_shape, resample=PIL.Image.Resampling(0)
                    ),
                    dtype=img.dtype,
                )
                for ch in range(no_channels)
            ]
        ).transpose(1, 2, 0)
        if no_channels > 1
        else np.asarray(
            PIL.Image.fromarray(img).resize(target_shape, resample=PIL.Image.Resampling(0)),
            dtype=img.dtype,
        )
    )
    print("Resized input image from {}->{}".format(orig_shape, np.asarray(img).shape))
    return img


def _degrade(img, sigma):
    """Degrade image by adding white Gaussian noise

    :param img: Original image
    :type img: np.ndarray
    :param sigma: Standard deviation of additive white Gaussian noise
    :type sigma: int
    :return: Degraded image
    :type return: np.ndarray

    Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
    Licensed under the Academic Free License version 3.0
    """
    degraded = np.random.normal(img, scale=sigma)
    print("Added white Gaussian noise with σ={}".format(sigma))
    return degraded


def make_target_and_degraded(image_file, sigma_noise, factor_rescale=None):
    _imread = tifffile.imread if os.path.splitext(image_file)[1] == ".tiff" else imageio.v2.imread
    target = _imread(image_file).astype(np.float64)
    if factor_rescale:
        target = _rescale(target, factor_rescale)
    noisy = _degrade(target, sigma_noise)
    return target, noisy


if __name__ == "__main__":
    args = _get_args()

    target, noisy = make_target_and_degraded(args.image_file, args.sigma, args.rescale)

    # infer name of .h5 file to write out
    filepath, filename = os.path.split(args.image_file)
    output_directory = filepath if args.output_directory is None else args.output_directory
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    file_extension = os.path.splitext(filename)[-1]
    rescale_suff = "-rescale{:.2f}".format(args.rescale).replace(".", "_") if args.rescale else ""
    h5_file = os.path.join(
        output_directory,
        filename.replace(file_extension, "-sigma{}{}.h5".format(args.sigma, rescale_suff)),
    )

    # write out as h5 file
    with h5py.File(h5_file, "w") as f:
        f.create_dataset(name="target", data=target)
        f.create_dataset(name="noisy", data=noisy)
    print("Wrote {}".format(h5_file))
