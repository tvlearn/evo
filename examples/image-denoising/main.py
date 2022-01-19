# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import os
import sys
import time
import datetime
import numpy as np
from mpi4py import MPI

from evo.utils.datalog import DataLog, TextPrinter, StoreToH5
from evo.utils.parallel import pprint, scatter_to_processes, gather_from_processes
from evo.variational.utils import init_states
from evo.models import BSC, SSSC

from tvutil.prepost import (
    OverlappingPatches,
    MultiDimOverlappingPatches,
    mean_merger,
    median_merger,
)

from params import get_args
from utils import stdout_logger, eval_fn, get_image
from viz import Visualizer


if __name__ == "__main__":

    # get MPI communicator
    comm = MPI.COMM_WORLD

    # get hyperparameters
    args = get_args()

    # initialize logs
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%y-%m-%d_%H-%M-%S")
    output_directory = (
        "./output/{}".format(timestamp) if args.output_directory is None else args.output_directory
    )
    h5output_file = os.path.join(output_directory, "training.h5")
    if not os.path.exists(output_directory) and comm.rank == 0:
        os.makedirs(output_directory)
    dlog = DataLog()
    dlog.set_handler(
        [
            "*",
        ],
        StoreToH5,
        h5output_file,
        warnings=False,
    )
    dlog.set_handler(["F", "psnr_mean", "psnr_median"], TextPrinter)
    if args.save_theta_all_epochs:
        theta_log_fn = dlog.append_all
    else:

        def theta_log_fn(val_dict):
            for k, v in val_dict.items():
                dlog.assign(k, v)

    txt_file = output_directory + "/terminal.txt"
    if comm.rank == 0:
        sys.stdout = stdout_logger(txt_file)
    pprint("Running on {} process{}".format(comm.size, "es" if comm.size > 1 else ""))
    pprint("Will write training output to {}".format(h5output_file))
    pprint("Will write terminal output to {}".format(txt_file))

    # print hyperparameters
    pprint("Hyperparameter used:")
    for k in sorted(vars(args), key=lambda s: s.lower()):
        pprint("{: <25} : {}".format(k, vars(args)[k]))

    comm.Barrier()

    # generating noisy image and extract image patches
    patch_width = args.patch_width if args.patch_width is not None else args.patch_height
    if comm.rank == 0:
        clean = get_image(args.clean_image, args.rescale)
        isrgb = np.ndim(clean) == 3 and clean.shape[2] == 3
        noisy = np.random.normal(clean, scale=args.noise_level)
        print("Added white Gaussian noise with σ={}".format(args.noise_level))
        dlog.assign("noisy_image", noisy)
        OVP = MultiDimOverlappingPatches if isrgb else OverlappingPatches
        ovp = OVP(noisy, args.patch_height, patch_width, patch_shift=1)
        Y = ovp.get().T
    else:
        Y = None
        isrgb = None
    isrgb = comm.bcast(isrgb)

    comm.Barrier()

    # instantiate model
    no_channels = 3 if isrgb else 1
    D = args.patch_height * patch_width * no_channels
    MODEL = {"ebsc": BSC, "es3c": SSSC}[args.algo]
    model = MODEL(D, args.H, args.Ksize)

    # distribute data to MPI processes
    pprint("Scattering data to processes")
    my_y, split_sizes, displacements = scatter_to_processes(Y)
    my_N = my_y.shape[0]
    my_data = {
        "y": my_y,  # local dataset
        "x_infr": np.logical_not(np.isnan(my_y)),  # reliable for learning
        "x": np.zeros_like(my_y),  # not x will be reconstructed
    }

    # initialize model and variational states
    pprint("Initializing model parameters")
    theta = model.check_params(model.standard_init(my_data))
    dlog.append_all({"{}_init".format(k): v for k, v in theta.items()})

    pprint("Initializing variational parameters")
    my_suff_stat = init_states(
        my_N,
        args.Ksize,
        args.H,
        args.parent_selection,
        args.mutation_algorithm,
        args.no_parents,
        args.no_children,
        args.no_generations,
        args.bitflip_prob,
    )

    # define strategies to merge reconstructed patches
    merge_strategies = {"mean": mean_merger, "median": median_merger}

    # initialize visualizer
    pprint("Initializing visualizer")
    viz_every = args.viz_every if args.viz_every is not None else args.no_epochs
    visualizer = (
        Visualizer(
            viz_every=viz_every,
            output_directory=output_directory,
            clean_image=clean,
            noisy_image=noisy,
            patch_size=(args.patch_height, patch_width),
            sort_gfs=True,
            ncol_gfs=4,
            gif_framerate=args.gif_framerate,
        )
        if comm.rank == 0
        else None
    )
    if args.algo == "ebsc":
        pprint(
            "Note: The BSC M-step yields a single prior value pi=1/H sum_h {pi_h}; visualized "
            "will be the pi_h."
        )
    comm.Barrier()

    merge_every = viz_every if args.merge_every is None else args.merge_every
    for e in range(args.no_epochs):
        dlog.progress("Epoch {} of {}".format(e + 1, args.no_epochs))
        start = time.time()

        # run training epochs
        do_reconstruction = e == 0 or (e + 1) % merge_every == 0
        F, S_nunique, S_sub, theta = model.step(theta, my_suff_stat, my_data, do_reconstruction)
        dlog.append_all({"F": F, "S_nunique": S_nunique, "S_sub": S_sub})
        theta_log_fn(theta)

        # merge reconstructed image patches and generate reconstructed image
        assert "y_reconstructed" in my_data if do_reconstruction else True
        Y_rec_T = (
            gather_from_processes(my_data["y_reconstructed"], split_sizes, displacements).T
            if do_reconstruction
            else None
        )
        merge = do_reconstruction and (comm.rank == 0)
        imgs = {
            k: ovp.set_and_merge(Y_rec_T, merge_method=v) if merge else None
            for k, v in merge_strategies.items()
        }

        # assess reconstruction quality in terms of PSNR
        psnrs = {k: eval_fn(clean, v) if merge else None for k, v in imgs.items()}

        # log reconstructions to H5 output file
        [dlog.append("psnr_{}".format(k), v) for k, v in psnrs.items()] if merge else []
        [dlog.append("reco_img_{}".format(k), v) for k, v in imgs.items()] if merge else []

        # visualize
        if comm.rank == 0:
            visualizer.process_epoch(epoch=e + 1, theta=theta, rec=imgs["mean"])
        comm.Barrier()

        pprint("\tTotal epoch runtime : %.2f s" % (time.time() - start))

    comm.Barrier()

    dlog.close()

    pprint("Finished")

    if comm.rank == 0:
        visualizer.finalize()

    comm.Barrier()
