# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from __future__ import division, print_function

import os
import re
import glob
import argparse
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa
from tvutil.viz import make_grid_with_black_boxes_and_white_background, scale  # noqa


def _get_args():
    p = argparse.ArgumentParser(
        description="Visualize results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("results_file", type=str, help="Path to H5 file with training output")
    p.add_argument(
        "--output_directory",
        type=str,
        help="Directory to write PNG file to (defaults " "to results file directory)",
    )
    return p.parse_args()


class Visualizer(object):
    def __init__(
        self,
        output_directory,
        noisy_image,
        patch_size,
        sort_gfs=True,
        topk_gfs=None,
        cmap=None,
        figsize=[9, 3],
        positions={
            "target": [0.001, 0.01, 0.20, 0.85],
            "noisy": [0.218, 0.01, 0.20, 0.85],
            "rec": [0.433, 0.01, 0.20, 0.85],
            "gfs": [0.715, 0.37, 0.28, 0.56],
            "priors": [0.731, 0.17, 0.26, 0.19],
        },
        labelsize=10,
        target_image=None,
        eval_method=None,
        eval_name=None,
        gif_framerate=None,
    ):
        self._output_directory = output_directory
        self._noisy_image = noisy_image
        self._patch_size = patch_size
        self._nrow_gfs = 10
        self._sort_gfs = sort_gfs
        if sort_gfs:
            print(
                "Displayed GFs will be ordered according to prior activation (from highest "
                "to lowest)"
            )
        self._topk_gfs = topk_gfs
        self._isrgb = np.ndim(noisy_image) == 3
        self._cmap = (plt.cm.jet if self._isrgb else plt.cm.gray) if cmap is None else cmap
        self._positions = positions
        self._labelsize = labelsize
        self._target_image = target_image
        self._eval_name = eval_method[0]
        self._eval_fn = eval_method[1]
        self._gif_framerate = gif_framerate

        self._fig = plt.figure(figsize=figsize)
        self._axes = {k: self._fig.add_axes(v, xmargin=0, ymargin=0) for k, v in positions.items()}
        self._handles = {k: None for k in positions}
        self._viz_noisy()
        self._viz_target()

    def _viz_target(self):
        assert "target" in self._axes
        ax = self._axes["target"]
        ax.axis("off")
        title = "Target"
        if self._target_image is not None:
            target = scale(self._target_image, [0.0, 1.0]) if self._isrgb else self._target_image
            self._handles["target"] = ax.imshow(target)
            self._handles["target"].set_cmap(self._cmap)
            title += "\n"
        else:
            _img = np.ones_like(self._noisy_image)
            _img[:] = np.nan
            self._handles["target"] = ax.imshow(_img)
            ax.text(
                0.5,
                0.5,
                "n/a",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
        ax.set_title(title)

    def _viz_noisy(self):
        assert "noisy" in self._axes
        ax = self._axes["noisy"]
        noisy = self._noisy_image
        noisy = scale(noisy, [0.0, 1.0]) if self._isrgb else noisy
        self._handles["noisy"] = ax.imshow(noisy)
        ax.axis("off")
        self._handles["noisy"].set_cmap(self._cmap)
        title = "Noisy"
        if self._target_image is not None:
            eval_result = self._eval_fn(self._target_image, noisy)
            title += "\n{}={:.2f}".format(self._eval_name.upper(), eval_result)
        ax.set_title(title)

    def _viz_rec(self, epoch, rec):
        assert "rec" in self._axes
        ax = self._axes["rec"]
        rec = scale(rec, [0.0, 1.0]) if self._isrgb else rec
        if self._handles["rec"] is None:
            self._handles["rec"] = ax.imshow(rec)
            ax.axis("off")
        else:
            self._handles["rec"].set_data(rec)
        self._handles["rec"].set_cmap(self._cmap)
        self._handles["rec"].set_clim(vmin=np.min(rec), vmax=np.max(rec))
        title = "Reco @ {}".format(epoch)
        if self._target_image is not None:
            eval_result = self._eval_fn(self._target_image, rec)
            title += "\n{}={:.2f}".format(self._eval_name.upper(), eval_result)
        ax.set_title(title)

    def _viz_gfs(self, epoch, gfs, suffix=""):
        assert "gfs" in self._axes
        ax = self._axes["gfs"]
        D, H = gfs.shape
        no_channels = 3 if self._isrgb else 1
        patch_height, patch_width = self._patch_size

        grid, cmap, vmin, vmax, scale_suff = make_grid_with_black_boxes_and_white_background(
            images=gfs.T.reshape(H, no_channels, patch_height, patch_width),
            nrow=self._nrow_gfs,
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

    def _viz_priors(self, epoch, pies, suffix=""):
        assert "priors" in self._axes
        ax = self._axes["priors"]
        xdata = np.arange(1, len(pies) + 1)
        ydata = pies
        if self._handles["priors"] is None:
            (self._handles["priors"],) = ax.plot(
                xdata,
                ydata,
                "b",
                linestyle="none",
                marker=".",
                markersize=4,
            )
            ax.set_ylabel(
                r"$\pi_h$ @ {}".format(epoch) + ("\n" + suffix) if suffix else "",
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
            self._handles["priors"].set_xdata(xdata)
            self._handles["priors"].set_ydata(ydata)
            ax.set_ylabel(r"$\pi_h$ @ {}".format(epoch) + ("\n" + suffix) if suffix else "")
            ax.relim()
            ax.autoscale_view()
            self._handles["pies_summed"].set_text(
                r"$\sum_h \pi_h$ = " + "{:.2f}".format(pies.sum())
            )

    def viz_epoch(self, epoch, priors, gfs, rec):
        inds_sort = np.argsort(priors)[::-1] if self._sort_gfs else np.arange(len(priors))
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
        self._viz_gfs(epoch, gfs.copy()[:, inds_sort_gfs], suffix_gfs)
        self._viz_priors(epoch, priors[inds_sort], "(sorted)" if self._sort_gfs else "")

    def process_epoch(self, epoch, priors, gfs, rec):
        assert rec is not None
        self.viz_epoch(epoch, priors, gfs, rec)
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


def _load_last_and_viz(
    output_directory,
    noisy_image,
    patch_size,
    ind_epoch,
    priors,
    gfs,
    reco_image,
    target_image=None,
    eval_method=None,
):
    H = gfs.shape[1]
    visualizer = Visualizer(
        output_directory=output_directory,
        noisy_image=noisy_image,
        patch_size=patch_size,
        topk_gfs=np.min([50, H]),
        target_image=target_image,
        eval_method=eval_method,
    )
    visualizer.process_epoch(epoch=ind_epoch + 1, priors=priors, gfs=gfs, rec=reco_image)
    visualizer.finalize()


if __name__ == "__main__":
    import h5py
    from utils import compute_psnr

    args = _get_args()

    with h5py.File(args.results_file, "r") as f:
        _load_last_and_viz(
            output_directory=args.output_directory
            if args.output_directory is not None
            else os.path.split(args.results_file)[0],
            noisy_image=f["noisy"][...],
            patch_size=f["patch_size"][...],
            ind_epoch=len(f["F"]) - 1,
            priors=f["pies"][...],
            gfs=f["W"][...],
            reco_image=f["reco"][-1],
            target_image=f["target"][...],
            eval_method=("psnr", compute_psnr),
        )
