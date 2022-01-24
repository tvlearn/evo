# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from __future__ import division, print_function

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from tvutil.viz import make_grid_with_black_boxes_and_white_background

from utils import merge_dict


class Visualizer(object):
    def __init__(
        self,
        output_directory,
        viz_every,
        datapoints,
        theta_gen,
        L_gen=None,
        cmap_weights=plt.cm.jet,
        memorize=("F",),
        positions={
            "datapoints": [0.0, 0.0, 0.15, 0.9],
            "W_gen": [0.15, 0.0, 0.15, 0.9],
            "W": [0.0, 0.0, 0.25, 0.9],
            "F": [0.3, 0.0, 0.7, 1.0],
        },
        gif_framerate=None,
    ):
        self._output_directory = output_directory
        self._viz_every = viz_every
        self._gif_framerate = gif_framerate
        self._cmap_weights = cmap_weights
        self._theta_gen = theta_gen
        self._datapoints = datapoints
        self._L_gen = L_gen
        self._labelsize = 10
        self._legendfontsize = 8

        self._memory = {k: [] for k in memorize}
        self._fig = plt.figure()
        self._axes = {k: self._fig.add_axes(v) for k, v in positions.items()}
        self._handles = {k: None for k in positions}
        for k in theta_gen.keys():
            self._handles["{}_gen".format(k)] = None
        self._handles["L_gen"] = None
        self._viz_datapoints()
        self._viz_gen_weights()

    def _viz_datapoints(self):
        assert "datapoints" in self._axes
        ax = self._axes["datapoints"]
        datapoints = self._datapoints
        N, D = datapoints.shape
        grid, cmap, vmin, vmax, scale_suff = make_grid_with_black_boxes_and_white_background(
            images=datapoints.copy().reshape(N, 1, int(np.sqrt(D)), int(np.sqrt(D))),
            nrow=int(np.ceil(N / 16)),
            surrounding=4,
            padding=8,
            repeat=20,
            global_clim=True,
            sym_clim=True,
            cmap=self._cmap_weights,
            eps=0.02,
        )

        self._handles["datapoints"] = ax.imshow(np.squeeze(grid), interpolation="none")
        ax.axis("off")

        self._handles["datapoints"].set_cmap(cmap)
        self._handles["datapoints"].set_clim(vmin=vmin, vmax=vmax)
        ax.set_title(r"$\vec{y}^{\,(n)}$")

    def _viz_gen_weights(self):
        assert "W_gen" in self._axes
        ax = self._axes["W_gen"]
        W_gen = self._theta_gen["W"]
        D, H = W_gen.shape
        grid, cmap, vmin, vmax, scale_suff = make_grid_with_black_boxes_and_white_background(
            images=W_gen.copy().T.reshape(H, 1, int(np.sqrt(D)), int(np.sqrt(D))),
            nrow=int(np.ceil(H / 16)),
            surrounding=4,
            padding=8,
            repeat=20,
            global_clim=True,
            sym_clim=True,
            cmap=self._cmap_weights,
            eps=0.02,
        )

        if self._handles["W_gen"] is None:
            self._handles["W_gen"] = ax.imshow(np.squeeze(grid), interpolation="none")
            ax.axis("off")
        else:
            self._handles["W_gen"].set_data(np.squeeze(grid))
        self._handles["W_gen"].set_cmap(cmap)
        self._handles["W_gen"].set_clim(vmin=vmin, vmax=vmax)
        ax.set_title(r"$W^{\mathrm{gen}}$")

    def _viz_weights(self, epoch, W, inds_sort=None):
        assert "W" in self._axes
        ax = self._axes["W"]
        D, H = W.shape
        W = W[:, inds_sort] if inds_sort is not None else W.copy()
        grid, cmap, vmin, vmax, scale_suff = make_grid_with_black_boxes_and_white_background(
            images=W.T.reshape(H, 1, int(np.sqrt(D)), int(np.sqrt(D))),
            nrow=int(np.ceil(H / 16)),
            surrounding=4,
            padding=8,
            repeat=20,
            global_clim=True,
            sym_clim=True,
            cmap=self._cmap_weights,
            eps=0.02,
        )

        if self._handles["W"] is None:
            self._handles["W"] = ax.imshow(np.squeeze(grid), interpolation="none")
            ax.axis("off")
        else:
            self._handles["W"].set_data(np.squeeze(grid))
        self._handles["W"].set_cmap(cmap)
        self._handles["W"].set_clim(vmin=vmin, vmax=vmax)
        ax.set_title("W @ {}".format(epoch))

    def _viz_free_energy(self):
        memory = self._memory
        assert "F" in memory
        assert "F" in self._axes
        ax = self._axes["F"]
        xdata = np.arange(1, len(memory["F"]) + 1)
        ydata_learned = memory["F"]
        if self._handles["F"] is None:
            (self._handles["F"],) = ax.plot(
                xdata,
                ydata_learned,
                "b",
                label=r"$\mathcal{F}(\mathcal{K},\Theta)$",
            )
            ax.set_xlabel("Epoch", fontsize=self._labelsize)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            add_legend = True
        else:
            self._handles["F"].set_xdata(xdata)
            self._handles["F"].set_ydata(ydata_learned)
            ax.relim()
            ax.autoscale_view()
            add_legend = False

        if self._L_gen is not None:
            ydata_gen = self._L_gen * np.ones_like(ydata_learned)
            if self._handles["L_gen"] is None:
                (self._handles["L_gen"],) = ax.plot(
                    xdata,
                    ydata_gen,
                    "b--",
                    label=r"$\mathcal{L}(\Theta^{\mathrm{gen}})$",
                )
            else:
                self._handles["L_gen"].set_xdata(xdata)
                self._handles["L_gen"].set_ydata(ydata_gen)

        if add_legend:
            ax.legend(prop={"size": self._legendfontsize}, ncol=2)

    def _viz_epoch(self, epoch, F, theta):
        self._viz_weights(epoch, theta["W"])
        self._viz_free_energy()

    def process_epoch(self, epoch, F, theta):
        memory = self._memory
        [memory[k].append(merge_dict(theta, {"F": F})[k]) for k in memory.keys()]
        if epoch % self._viz_every == 0:
            self._viz_epoch(epoch, F, theta)
            self._save_epoch(epoch)

    def _save_epoch(self, epoch):
        output_directory = self._output_directory
        png_file = "{}/training{}.png".format(
            output_directory, "_epoch{}".format(epoch) if self._gif_framerate is not None else ""
        )
        plt.savefig(png_file)
        print("\tWrote " + png_file)

    def _write_gif(self, framerate):
        output_directory = self._output_directory
        gif_file = "{}/training.gif".format(output_directory)
        print("Creating {} ...".format(gif_file), end="")
        # work-around for correct color display from https://stackoverflow.com/a/58832086
        os.system(
            "ffmpeg -y -framerate {} -i {}/training_epoch%d.png -vf palettegen \
            {}/palette.png".format(
                framerate, output_directory, output_directory
            )
        )
        os.system(
            "ffmpeg -y -framerate {} -i {}/training_epoch%d.png -i {}/palette.png -lavfi \
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


class BSCVisualizer(Visualizer):
    def __init__(self, **kwargs):
        super(BSCVisualizer, self).__init__(
            memorize=("F", "pi", "sigma"),
            positions={
                "datapoints": [0.0, 0.0, 0.07, 0.94],
                "W_gen": [0.08, 0.0, 0.1, 0.94],
                "W": [0.2, 0.0, 0.1, 0.94],
                "F": [0.4, 0.76, 0.58, 0.23],
                "sigma": [0.4, 0.43, 0.58, 0.23],
                "pi": [0.4, 0.1, 0.58, 0.23],
            },
            **kwargs
        )

    def _viz_sigma2(self):
        memory = self._memory
        assert "sigma" in memory
        assert "sigma" in self._axes
        ax = self._axes["sigma"]
        xdata = np.arange(1, len(memory["sigma"]) + 1)
        ydata_learned = np.squeeze(np.array(memory["sigma"])) ** 2
        if self._handles["sigma"] is None:
            (self._handles["sigma"],) = ax.plot(
                xdata,
                ydata_learned,
                "b",
                label=r"$\sigma^2$",
            )
            ax.set_xlabel("Epoch", fontsize=self._labelsize)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            add_legend = True
        else:
            self._handles["sigma"].set_xdata(xdata)
            self._handles["sigma"].set_ydata(ydata_learned)
            ax.relim()
            ax.autoscale_view()
            add_legend = True

        ydata_gen = self._theta_gen["sigma"] ** 2 * np.ones_like(ydata_learned)
        if self._handles["sigma_gen"] is None:
            (self._handles["sigma_gen"],) = ax.plot(
                xdata,
                ydata_gen,
                "b--",
                label=r"$(\sigma^{\mathrm{gen}})^2$",
            )
        else:
            self._handles["sigma_gen"].set_xdata(xdata)
            self._handles["sigma_gen"].set_ydata(ydata_gen)

        if add_legend:
            ax.legend(prop={"size": self._legendfontsize}, ncol=2)

    def _viz_pi(self):
        memory = self._memory
        assert "pi" in memory
        assert "pi" in self._axes
        ax = self._axes["pi"]
        xdata = np.arange(1, len(memory["pi"]) + 1)
        ydata_learned = memory["pi"]
        if self._handles["pi"] is None:
            (self._handles["pi"],) = ax.plot(
                xdata,
                ydata_learned,
                "b",
                label=r"$\pi$",
            )
            ax.set_xlabel("Epoch", fontsize=self._labelsize)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            add_legend = True
        else:
            self._handles["pi"].set_xdata(xdata)
            self._handles["pi"].set_ydata(ydata_learned)
            ax.relim()
            ax.autoscale_view()
            add_legend = False

        ydata_gen = self._theta_gen["pi"] * np.ones_like(ydata_learned)
        if self._handles["pi_gen"] is None:
            (self._handles["pi_gen"],) = ax.plot(
                xdata,
                ydata_gen,
                "b--",
                label=r"$\pi^{\mathrm{gen}}$",
            )
        else:
            self._handles["pi_gen"].set_xdata(xdata)
            self._handles["pi_gen"].set_ydata(ydata_gen)

        if add_legend:
            ax.legend(prop={"size": self._legendfontsize}, ncol=2)

    def _viz_epoch(self, epoch, F, theta):
        super(BSCVisualizer, self)._viz_epoch(epoch, F, theta)
        self._viz_sigma2()
        self._viz_pi()


class SSSCVisualizer(Visualizer):
    def __init__(self, sort_acc_to_desc_priors=False, **kwargs):
        super(SSSCVisualizer, self).__init__(
            memorize=("F", "sigma2"),
            positions={
                "datapoints": [0.0, 0.0, 0.07, 0.94],
                "W_gen": [0.09, 0.0, 0.1, 0.94],
                "W": [0.21, 0.0, 0.1, 0.94],
                "F": [0.40, 0.76, 0.25, 0.23],
                "sigma2": [0.40, 0.43, 0.25, 0.23],
                "pies": [0.40, 0.1, 0.25, 0.23],
                "mus": [0.74, 0.76, 0.25, 0.23],
                "Psi": [0.74, 0.36, 0.25, 0.23],
                "Psi_gen": [0.74, 0.04, 0.25, 0.23],
            },
            **kwargs
        )
        self._sort_acc_to_desc_priors = sort_acc_to_desc_priors
        if sort_acc_to_desc_priors:
            print("Sorting according to priors ascendingly")
        self._viz_Psi_gen()

    def _viz_sigma2(self):
        memory = self._memory
        assert "sigma2" in memory
        assert "sigma2" in self._axes
        ax = self._axes["sigma2"]
        xdata = np.arange(1, len(memory["sigma2"]) + 1)
        ydata_learned = np.squeeze(np.array(memory["sigma2"]))
        if self._handles["sigma2"] is None:
            (self._handles["sigma2"],) = ax.plot(
                xdata,
                ydata_learned,
                "b",
                label=r"$\sigma^2$",
            )
            ax.set_xlabel("Epoch", fontsize=self._labelsize)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            add_legend = True
        else:
            self._handles["sigma2"].set_xdata(xdata)
            self._handles["sigma2"].set_ydata(ydata_learned)
            ax.relim()
            ax.autoscale_view()
            add_legend = True

        ydata_gen = self._theta_gen["sigma2"] * np.ones_like(ydata_learned)
        if self._handles["sigma2_gen"] is None:
            (self._handles["sigma2_gen"],) = ax.plot(
                xdata,
                ydata_gen,
                "b--",
                label=r"$(\sigma^{\mathrm{gen}})^2$",
            )
        else:
            self._handles["sigma2_gen"].set_xdata(xdata)
            self._handles["sigma2_gen"].set_ydata(ydata_gen)

        if add_legend:
            ax.legend(prop={"size": self._legendfontsize}, ncol=2)

    def _viz_pies(self, epoch, pies, inds_sort=None):
        assert "pies" in self._axes
        ax = self._axes["pies"]
        xdata = np.arange(1, len(pies) + 1)
        ydata_learned = pies[inds_sort] if inds_sort is not None else pies
        if self._handles["pies"] is None:
            (self._handles["pies"],) = ax.plot(
                xdata,
                ydata_learned,
                "b",
                linestyle="none",
                marker=".",
                markersize=4,
                label=r"$\pi_h$ @ {}".format(epoch),
            )
            ax.set_xlabel(r"$h$", fontsize=self._labelsize)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            self._handles["pies"].set_xdata(xdata)
            self._handles["pies"].set_ydata(ydata_learned)
            self._handles["pies"].set_label(r"$\pi_h$ @ {}".format(epoch))
            ax.relim()
            ax.autoscale_view()

        ydata_gen = (
            self._theta_gen["pies"][inds_sort] if inds_sort is not None else self._theta_gen["pies"]
        )
        if self._handles["pies_gen"] is None:
            (self._handles["pies_gen"],) = ax.plot(
                xdata,
                ydata_gen,
                "b",
                linestyle="none",
                marker="o",
                fillstyle=Line2D.fillStyles[-1],
                markersize=4,
                label=r"$\pi_h^{\mathrm{gen}}$",
            )
        else:
            self._handles["pies_gen"].set_xdata(xdata)
            self._handles["pies_gen"].set_ydata(ydata_gen)

        ax.legend(prop={"size": self._legendfontsize}, ncol=2)

    def _viz_mus(self, epoch, mus, inds_sort=None):
        assert "mus" in self._axes
        ax = self._axes["mus"]
        xdata = np.arange(1, len(mus) + 1)
        ydata_learned = mus[inds_sort] if inds_sort is not None else mus
        if self._handles["mus"] is None:
            (self._handles["mus"],) = ax.plot(
                xdata,
                ydata_learned,
                "b",
                linestyle="none",
                marker=".",
                markersize=4,
                label=r"$\mu_h$ @ {}".format(epoch),
            )
            ax.set_xlabel(r"$h$", fontsize=self._labelsize)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            self._handles["mus"].set_xdata(xdata)
            self._handles["mus"].set_ydata(ydata_learned)
            self._handles["mus"].set_label(r"$\mu_h$ @ {}".format(epoch))
            ax.relim()
            ax.autoscale_view()

        ydata_gen = (
            self._theta_gen["mus"][inds_sort] if inds_sort is not None else self._theta_gen["mus"]
        )
        if self._handles["mus_gen"] is None:
            (self._handles["mus_gen"],) = ax.plot(
                xdata,
                ydata_gen,
                "b",
                linestyle="none",
                marker="o",
                fillstyle=Line2D.fillStyles[-1],
                markersize=4,
                label=r"$\mu_h^{\mathrm{gen}}$",
            )
        else:
            self._handles["mus_gen"].set_xdata(xdata)
            self._handles["mus_gen"].set_ydata(ydata_gen)

        ax.legend(prop={"size": self._legendfontsize}, ncol=2)

    def _viz_Psi(self, epoch, Psi):
        assert "Psi" in self._axes
        ax = self._axes["Psi"]
        if self._handles["Psi"] is None:
            self._handles["Psi"] = ax.imshow(Psi)
            ax.axis("off")
        else:
            self._handles["Psi"].set_data(Psi)
        self._handles["Psi"].set_cmap(plt.cm.jet)
        max_abs = np.max(np.abs(Psi))
        self._handles["Psi"].set_clim(vmin=-max_abs, vmax=max_abs)
        ax.set_title(r"$\Psi$ @ {}".format(epoch))

    def _viz_Psi_gen(self):
        assert "Psi_gen" in self._axes
        ax = self._axes["Psi_gen"]
        Psi_gen = self._theta_gen["Psi"]
        self._handles["Psi_gen"] = ax.imshow(Psi_gen)
        ax.axis("off")
        self._handles["Psi_gen"].set_cmap(plt.cm.jet)
        max_abs = np.max(np.abs(Psi_gen))
        self._handles["Psi_gen"].set_clim(vmin=-max_abs, vmax=max_abs)
        ax.set_title(r"$\Psi^{\mathrm{gen}}$")

    def _viz_epoch(self, epoch, F, theta):
        inds_sort = np.argsort(theta["pies"])[::-1] if self._sort_acc_to_desc_priors else None
        self._viz_free_energy()
        self._viz_sigma2()
        self._viz_weights(epoch, theta["W"], inds_sort=inds_sort)
        self._viz_pies(epoch, theta["pies"], inds_sort=inds_sort)
        self._viz_mus(epoch, theta["mus"], inds_sort=inds_sort)
        self._viz_Psi(epoch, theta["Psi"])
