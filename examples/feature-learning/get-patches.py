# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import numpy as np
import argparse
from imageio.v2 import imread
from tvutil.prepost import extract_random_patches


def _get_args():
    parser = argparse.ArgumentParser(
        description="Extract random patches from image and write to npz file"
    )
    parser.add_argument("image_file", type=str, help="Full image file path")
    parser.add_argument(
        "patch_size",
        type=int,
        nargs=2,
        help="Patch size, provided as (patch_height, patch_width) tuple",
    )
    parser.add_argument("no_patches", type=int, help="Number of patches to extract")
    parser.add_argument("npz_file", type=str, help="Full .npz file path")
    return parser.parse_args()


if __name__ == "__main__":
    args = _get_args()

    # read data and check dimensions
    img = imread(args.image_file)
    assert np.ndim(img) == 2 or np.ndim(img) == 3, (
        "Image dimension mismatch, expected "
        "(image_height, image_width) or (image_height, image_width, no_channels)"
    )
    print("Read {} of shape {}".format(args.image_file, img.shape))
    no_channels = img.shape[-1] if np.ndim(img) == 3 else 1

    # extract patches
    patches = extract_random_patches(
        images=img[None, :, :, None] if no_channels == 1 else img[None, :, :, :],
        patch_size=args.patch_size,
        no_patches=args.no_patches,
    )
    print("Extracted {} patches of size {}".format(args.no_patches, args.patch_size))

    # write out as npz file
    np.savez(
        file=args.npz_file,
        data=patches.astype(np.float32),
        patch_height=args.patch_size[0],
        patch_width=args.patch_size[1],
        no_channels=no_channels,
    )
    print("Wrote {}".format(args.npz_file))
