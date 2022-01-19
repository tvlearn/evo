# -*- coding: utf-8 -*-

import sys
import numpy as np


def generate_bars_dict(H, neg_bars=False):
    """Generate a ground-truth dictionary W suitable for a std. bars test

    Creates H bases vectors with horizontal and vertival bars on a R*R pixel grid,
    (wth R = H // 2).  The function thus returns a matrix storing H dictionaries of
    size D=R*R.

    :param H: Number of latent variables
    :type  H: int
    :param neg_bars: Should half the bars have a negative value (-1)?
    :type  neg_bars: bool
    :rtype: ndarray (D x H)

    Source: https://github.com/ml-uol/prosper/blob/master/prosper/utils/barstest.py::
    generate_bars_dict

    For LICENSING and COPYRIGHT for this function see prosper's license at:
    https://github.com/ml-uol/prosper/blob/master/LICENSE.txt
    """
    R = H // 2
    D = R ** 2
    W_gt = np.zeros((R, R, H))
    for i in range(R):
        W_gt[i, :, i] = 1.0
        W_gt[:, i, R + i] = 1.0

    if neg_bars:
        sign = 1 - 2 * np.random.randint(2, size=(H,))
        W_gt = sign[None, None, :] * W_gt
    return W_gt.reshape((D, H))


def merge_dict(*dicts):
    """Merge dictionaries

    :param dicts: Dictionaries to merge
    :type dicts: Tuple of dicts
    :return: Dictionary containing fields of all input dictionaries
    :type return: dict

    Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
    Licensed under the Academic Free License version 3.0
    """
    merged = dicts[0].copy()
    for d in dicts:
        merged.update(d)  # Python 2 friendly
    return merged


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
