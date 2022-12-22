# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from __future__ import print_function

import os
import sys
import time
import copy
import datetime
import numpy as np
from mpi4py import MPI

from evo.utils.datalog import DataLog, TextPrinter, StoreToH5
from evo.utils.parallel import pprint, scatter_to_processes
from evo.variational.utils import init_states
from evo.models import BSC, SSSC

from params import get_args
from utils import generate_bars_dict, merge_dict, stdout_logger
from viz import BSCVisualizer, SSSCVisualizer


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
    dlog.set_handler(["F", "L_gen"], TextPrinter)
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

    # instantiate model
    D = (args.H / 2) ** 2
    MODEL = {"ebsc": BSC, "es3c": SSSC}[args.algo]
    model = MODEL(D, args.H, args.Ksize)

    # generate data
    pprint("Generating data")
    pi_gen = args.pi_gen if args.pi_gen is not None else 2.0 / args.H
    theta_gen = (
        {
            "W": args.bar_amp * generate_bars_dict(args.H, neg_bars=args.neg_bars),
            "pi": pi_gen,
            "sigma": args.sigma_gen,
        }
        if args.algo == "ebsc"
        else {
            "W": args.bar_amp * generate_bars_dict(args.H, neg_bars=args.neg_bars),
            "pies": np.ones(args.H) * pi_gen,
            "sigma2": np.array(args.sigma_gen**2),
            "mus": np.ones(args.H) * args.mu_gen,
            "Psi": np.eye(args.H) * args.psi_gen**2,
        }
    )
    comm.Barrier()
    dlog.append("model", args.algo.upper())
    dlog.append_all({"{}_gen".format(k): v for k, v in theta_gen.items()})
    data = model.generate_data(theta_gen, args.no_data_points) if comm.rank == 0 else None
    theta_gen = model.check_params(theta_gen)
    Y = data["y"] if comm.rank == 0 else None
    dlog.append("Y", Y)
    comm.Barrier()

    # distribute data to MPI processes
    pprint("Scattering data to processes")
    my_y = scatter_to_processes(Y)
    my_N = my_y.shape[0]
    my_data = {"y": my_y, "x_infr": np.logical_not(np.isnan(my_y))}
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

    # compute likelihood of generative parameters
    pprint("Computing exact likelihood")
    if args.H <= 10:
        L_gen = model.free_energy(my_data, copy.deepcopy(theta_gen), my_suff_stat, full=True)
        dlog.append("L_gen", L_gen)
    else:
        L_gen = None
        pprint(
            "Did not compute likelihood of generative parameters (too expensive for H={})".format(
                args.H
            )
        )
    comm.Barrier()

    # initialize visualizer
    pprint("Initializing visualizer")
    Visualizer = (
        {"ebsc": BSCVisualizer, "es3c": SSSCVisualizer}[args.algo] if comm.rank == 0 else None
    )
    visualizer = (
        Visualizer(
            viz_every=args.viz_every if args.viz_every is not None else args.no_epochs,
            output_directory=output_directory,
            datapoints=Y[:15],
            theta_gen=theta_gen,
            L_gen=L_gen,
            gif_framerate=args.gif_framerate,
        )
        if comm.rank == 0
        else None
    )
    comm.Barrier()

    for e in range(args.no_epochs):

        # run training epoch
        dlog.progress("Epoch {} of {}".format(e + 1, args.no_epochs))
        start = time.time()
        F, S_nunique, S_sub, theta = model.step(theta, my_suff_stat, my_data)
        dlog.append_all(merge_dict({"F": F, "S_nunique": S_nunique, "S_sub": S_sub}, theta))
        pprint("\tTotal epoch runtime : %.2f s" % (time.time() - start))

        # visualize
        if comm.rank == 0:
            visualizer.process_epoch(epoch=e + 1, F=F, theta=theta)

    comm.Barrier()

    dlog.close()

    pprint("Finished")

    if comm.rank == 0:
        visualizer.finalize()

    comm.Barrier()
