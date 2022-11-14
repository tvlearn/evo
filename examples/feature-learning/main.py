# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from __future__ import print_function

import os
import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

from evo.utils.datalog import DataLog, TextPrinter, StoreToH5
from evo.utils.parallel import pprint, scatter_to_processes
from evo.variational.utils import init_states
from evo.models import BSC, SSSC

from params import get_args
from utils import stdout_logger, merge_dict, viz_lower_bound
from tvutil.viz import save_grid


if __name__ == "__main__":

    # get MPI communicator
    comm = MPI.COMM_WORLD

    # get hyperparameters
    args = get_args()

    # check data
    with np.load(args.data_file) as d:
        for key in ["data", "patch_height", "patch_width", "no_channels"]:
            assert key in d, "{}: Could not find {} key".format(args.data_file, key)
            assert np.ndim(d["data"]) == 2, (
                "{}: Expected data node to be 2-dim. array "
                "with shape (no. data points, no. observables)"
            )

    # initialize logs
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%y-%m-%d_%H-%M-%S")
    output_directory = (
        "./output/{}".format(timestamp) if args.output_directory is None else args.output_directory
    )
    training_file = os.path.join(output_directory, "training.h5")
    if not os.path.exists(output_directory) and comm.rank == 0:
        os.makedirs(output_directory)
    dlog = DataLog()
    dlog.set_handler(
        [
            "*",
        ],
        StoreToH5,
        training_file,
    )
    dlog.set_handler(
        [
            "F",
        ],
        TextPrinter,
    )
    txt_file = output_directory + "/terminal.txt"
    if comm.rank == 0:
        sys.stdout = stdout_logger(txt_file)
    pprint("Running on {} process{}".format(comm.size, "es" if comm.size > 1 else ""))
    pprint("Will write training output to {}".format(training_file))
    pprint("Will write terminal output to {}".format(txt_file))

    # print hyperparameters
    pprint("Hyperparameter used:")
    for k in sorted(vars(args), key=lambda s: s.lower()):
        pprint("{: <25} : {}".format(k, vars(args)[k]))

    comm.Barrier()

    # read data
    if comm.rank == 0:
        with np.load(args.data_file) as d:
            Y = d["data"].astype(np.float64)
            patch_height = d["patch_height"]
            patch_width = d["patch_width"]
            no_channels = d["no_channels"]
            assert Y.shape[1] == patch_height * patch_width * no_channels
    else:
        Y = None
        patch_height, patch_width, no_channels = None, None, None
    patch_height, patch_width, no_channels = (
        comm.bcast(patch_height),
        comm.bcast(patch_width),
        comm.bcast(no_channels),
    )
    D = patch_height * patch_width * no_channels
    comm.Barrier()

    # distribute data to MPI processes
    pprint("Scattering data to processes")
    my_y = scatter_to_processes(Y)[0]
    my_N = my_y.shape[0]
    my_data = {"y": my_y, "x_infr": np.logical_not(np.isnan(my_y))}
    comm.Barrier()

    # visualize a few samples
    if comm.rank == 0:
        N_viz = int(np.min([my_N, 16]))
        samples = my_y[:N_viz].reshape(N_viz, no_channels, patch_height, patch_width)
        for ch in range(no_channels):
            save_grid(
                png_file=output_directory
                + "/samples{}.png".format(
                    "-ch{}".format(ch) if no_channels > 1 else "",
                ),
                images=samples[:, ch][:, None, :, :],
                nrow=int(np.sqrt(N_viz)),
                surrounding=2,
                padding=4,
                repeat=5,
                global_clim=False,
                sym_clim=False,
                cmap=plt.cm.gray,
            )
    comm.Barrier()

    # instantiate model
    MODEL = {"bsc": BSC, "sssc": SSSC}[args.model]
    model = MODEL(D, args.H, args.Ksize)
    comm.Barrier()

    # initialize model and variational states
    pprint("Initializing model parameters")
    theta = model.check_params(model.standard_init(my_data))
    dlog.append_all({"{}_init".format(k): v for k, v in theta.items()})
    comm.Barrier()

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
    comm.Barrier()

    lower_bound_log = []
    for e in range(args.no_epochs):

        # run training epoch
        dlog.progress("Epoch {} of {}".format(e + 1, args.no_epochs))
        start = time.time()
        F, S_nunique, S_sub, theta = model.step(theta, my_suff_stat, my_data)
        dlog.append_all(merge_dict({"F": F, "S_nunique": S_nunique, "S_sub": S_sub}, theta))
        pprint("\tTotal epoch runtime : %.2f s" % (time.time() - start))

        if comm.rank == 0:

            # visualize generative fields
            gfs = (
                theta["W"] if not args.sort_gfs else theta["W"][:, np.argsort(theta["pies"][::-1])]
            )
            gfs = gfs.T.reshape(args.H, no_channels, patch_height, patch_width)
            for ch in range(no_channels):
                save_grid(
                    png_file=output_directory
                    + "/gfs{}{}.png".format(
                        "-sorted" if args.sort_gfs else "",
                        "-ch{}".format(ch) if no_channels > 1 else "",
                    ),
                    images=gfs[:, ch][:, None, :, :],
                    nrow=int(np.sqrt(args.H)),
                    surrounding=2,
                    padding=4,
                    repeat=5,
                    global_clim=False,
                    sym_clim=False,
                    cmap=plt.cm.gray,
                )

            # visualize lower bound
            lower_bound_log.append(F)
            viz_lower_bound(output_directory + "/lower_bound.png", lower_bound_log)

        comm.Barrier()

    dlog.close()

    pprint("Finished")
