# -*- coding: utf-8 -*-

import sys

try:
    from skimage.measure import compare_psnr

    def compute_psnr(target, reco):
        return compare_psnr(target, reco, 255)

except ImportError:
    from skimage.metrics import peak_signal_noise_ratio

    def compute_psnr(target, reco):
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


def print_dict(d):
    for k in sorted(d, key=lambda s: s.lower()):
        print("{: <25} : {}".format(k, d[k]))
