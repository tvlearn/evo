# -*- coding: utf-8 -*-

import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


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


def viz_lower_bound(png_file, values):
    """Visualize lower bound across epochs

    :param png_file: Full path of png file to write out
    :type png_file: str
    :param values: Lower bound values
    :type values: list or one-dim. nd.array

    Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
    Licensed under the Academic Free License version 3.0
    """

    label_kwargs = {"fontsize": 16}
    plt.figure()
    plt.plot(values)
    plt.xlabel("Epoch", **label_kwargs)
    plt.ylabel(r"$\mathcal{F}(\mathcal{K},\Theta)$", **label_kwargs)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(png_file)
    plt.close()
    print("Wrote {}".format(png_file))
