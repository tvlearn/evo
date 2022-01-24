# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from __future__ import division, print_function

import os
import re
import glob
import numpy as np
from utils import eval_fn
import matplotlib
from matplotlib.ticker import MaxNLocator

matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa
from tvutil.viz import make_grid_with_black_boxes_and_white_background, scale  # noqa


class Visualizer(object):
    def __init__(
        self,
        viz_every,
        output_directory,
        clean_image,
        incomplete_image,
        patch_size=None,
        ncol_gfs=5,
        sort_gfs=True,
        topk_gfs=None,
        cmap=None,
        figsize=[9, 3],
        positions={
            "clean": [0.001, 0.01, 0.20, 0.85],
            "incomplete": [0.218, 0.01, 0.20, 0.85],
            "rec": [0.433, 0.01, 0.20, 0.85],
            "gfs": [0.715, 0.37, 0.28, 0.56],
            "pies": [0.730, 0.18, 0.26, 0.24],
        },
        labelsize=10,
        gif_framerate=None,
    ):

        self._viz_every = viz_every
        self._output_directory = output_directory
        self._gif_framerate = gif_framerate
        if gif_framerate is not None and viz_every > 1:
            print("Choose --viz_every=1 for best gif results")
        self._clean_image = clean_image
        self._incomplete_image = incomplete_image
        self._patch_size = patch_size
        self._ncol_gfs = ncol_gfs
        self._sort_gfs = sort_gfs
        if sort_gfs:
            print(
                "Displayed GFs will be ordered according to prior activation (from highest "
                "to lowest)"
            )
        self._topk_gfs = topk_gfs
        self._isrgb = np.ndim(clean_image) == 3
        self._cmap = (plt.cm.jet if self._isrgb else plt.cm.gray) if cmap is None else cmap
        self._positions = positions
        self._labelsize = labelsize

        self._fig = plt.figure(figsize=figsize)
        self._axes = {k: self._fig.add_axes(v, xmargin=0, ymargin=0) for k, v in positions.items()}
        self._handles = {k: None for k in positions}
        self._viz_clean()
        self._viz_incomplete()

    def _viz_clean(self):
        assert "clean" in self._axes
        ax = self._axes["clean"]
        clean = scale(self._clean_image, [0.0, 1.0]) if self._isrgb else self._clean_image
        self._handles["clean"] = ax.imshow(clean)
        ax.axis("off")
        self._handles["clean"].set_cmap(self._cmap)
        ax.set_title("Clean\n")

    def _viz_incomplete(self):
        assert "incomplete" in self._axes
        ax = self._axes["incomplete"]
        incomplete = self._incomplete_image.copy()
        if self._isrgb:
            wasnan = np.isnan(incomplete)
            incomplete[wasnan] = np.nanmin(incomplete)
            incomplete = scale(incomplete, [0.0, 1.0])
            incomplete[wasnan] = np.nan
        self._handles["incomplete"] = ax.imshow(incomplete)
        ax.axis("off")
        self._handles["incomplete"].set_cmap(self._cmap)
        ratio = int(float(np.isnan(incomplete).sum()) / self._clean_image.size * 100)
        ax.set_title("Incomplete\n{} % missing".format(ratio))

    def _viz_rec(self, epoch, rec):
        assert "rec" in self._axes
        psnr = eval_fn(self._clean_image, rec)
        rec = scale(rec, [0.0, 1.0]) if self._isrgb else rec
        ax = self._axes["rec"]
        if self._handles["rec"] is None:
            self._handles["rec"] = ax.imshow(rec)
            ax.axis("off")
        else:
            self._handles["rec"].set_data(rec)
        self._handles["rec"].set_cmap(self._cmap)
        self._handles["rec"].set_clim(vmin=np.min(rec), vmax=np.max(rec))
        ax.set_title("Reco @ {}\nPSNR={:.2f}".format(epoch, psnr))

    def _viz_weights(self, epoch, gfs, suffix=""):
        assert "gfs" in self._axes
        ax = self._axes["gfs"]
        D, H = gfs.shape
        no_channels = 3 if self._isrgb else 1
        patch_height, patch_width = (
            (int(np.sqrt(D / no_channels)), int(np.sqrt(D / no_channels)))
            if self._patch_size is None
            else self._patch_size
        )
        grid, cmap, vmin, vmax, scale_suff = make_grid_with_black_boxes_and_white_background(
            images=gfs.T.reshape(H, no_channels, patch_height, patch_width),
            nrow=int(np.ceil(H / self._ncol_gfs)),
            surrounding=2,
            padding=4,
            repeat=10,
            global_clim=False,
            sym_clim=False,
            cmap=self._cmap,
            eps=0.02,
        )

        gfs = grid.transpose(1, 2, 0) if self._isrgb else np.squeeze(grid)
        if self._handles["gfs"] is None:
            self._handles["gfs"] = ax.imshow(gfs, interpolation="none")
            ax.axis("off")
        else:
            self._handles["gfs"].set_data(gfs)
        self._handles["gfs"].set_cmap(cmap)
        self._handles["gfs"].set_clim(vmin=vmin, vmax=vmax)
        ax.set_title("GFs @ {}".format(epoch) + ("\n" + suffix) if suffix else "")

    def _viz_pies(self, epoch, pies, suffix=""):
        assert "pies" in self._axes
        ax = self._axes["pies"]
        xdata = np.arange(1, len(pies) + 1)
        ydata = pies
        if self._handles["pies"] is None:
            (self._handles["pies"],) = ax.plot(
                xdata,
                ydata,
                "b",
                linestyle="none",
                marker=".",
                markersize=4,
            )
            ax.set_ylabel(
                r"$\pi_h$ @ {}".format(epoch) + (" " + suffix) if suffix else "",
                fontsize=self._labelsize,
            )
            ax.set_xlabel(r"$h$", fontsize=self._labelsize)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(axis="x", labelrotation=30)
            self._handles["pies_summed"] = ax.text(
                0.81,
                0.76,
                r"$\sum_h \pi_h$ = " + "{:.2f}".format(pies.sum()),
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
        else:
            self._handles["pies"].set_xdata(xdata)
            self._handles["pies"].set_ydata(ydata)
            ax.set_ylabel(r"$\pi_h$ @ {}".format(epoch) + (" " + suffix) if suffix else "")
            ax.relim()
            ax.autoscale_view()
            self._handles["pies_summed"].set_text(
                r"$\sum_h \pi_h$ = " + "{:.2f}".format(pies.sum())
            )

    def viz_epoch(self, epoch, theta, rec):
        inds_sort = (
            np.argsort(theta["pies"])[::-1] if self._sort_gfs else np.arange(len(theta["pies"]))
        )
        inds_sort_gfs = inds_sort[: self._topk_gfs] if self._topk_gfs is not None else inds_sort
        if rec is not None:
            self._viz_rec(epoch, rec)
        suffix_gfs = (
            (
                "sorted, top {}".format(self._topk_gfs)
                if self._sort_gfs
                else "top {}".format(self._topk_gfs)
            )
            if self._topk_gfs
            else ("sorted" if self._sort_gfs else "")
        )
        self._viz_weights(epoch, theta["W"].copy()[:, inds_sort_gfs], suffix_gfs)
        self._viz_pies(epoch, theta["pies"][inds_sort], "(sorted)" if self._sort_gfs else "")

    def process_epoch(self, epoch, theta, rec):
        if epoch == 1 or epoch % self._viz_every == 0:
            assert rec is not None
            self.viz_epoch(epoch, theta, rec)
            self.save_epoch(epoch)

    def save_epoch(self, epoch):
        output_directory = self._output_directory
        png_file = "{}/training{}.png".format(
            output_directory,
            "_epoch{:04d}".format(epoch) if self._gif_framerate is not None else "",
        )
        plt.savefig(png_file)
        print("\tWrote " + png_file)

    def _write_gif(self, framerate):
        output_directory = self._output_directory
        gif_file = "{}/training.gif".format(output_directory)
        print("Creating {} ...".format(gif_file), end="")
        # work-around for correct color display from https://stackoverflow.com/a/58832086
        os.system(
            "ffmpeg -y -framerate {} -i {}/training_epoch%*.png -vf palettegen \
            {}/palette.png".format(
                framerate, output_directory, output_directory
            )
        )
        os.system(
            "ffmpeg -y -framerate {} -i {}/training_epoch%*.png -i {}/palette.png -lavfi \
            paletteuse {}/training.gif".format(
                framerate, output_directory, output_directory, output_directory
            )
        )
        print("Done")

        png_files = glob.glob("{}/training_epoch*.png".format(output_directory))
        png_files.sort(
            key=lambda var: [
                int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)
            ]
        )
        last_epoch_str = "_epoch{}".format(png_files[-1].split("_epoch")[1].replace(".png", ""))
        for f in png_files:
            if last_epoch_str in f:
                old = f
                new = f.replace(last_epoch_str, "_last_epoch")
                os.rename(old, new)
                print("Renamed {}->{}".format(old, new))
            else:  # keep png of last epoch
                os.remove(f)
                print("Removed {}".format(f))
        os.remove("{}/palette.png".format(output_directory))

    def finalize(self):
        plt.close()
        if self._gif_framerate is not None:
            self._write_gif(framerate=self._gif_framerate)
