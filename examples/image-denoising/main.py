# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import os
import sys
import time
import h5py
import datetime
import argparse
import numpy as np
from mpi4py import MPI

from evo.utils.datalog import DataLog, TextPrinter, StoreToH5
from evo.utils.parallel import pprint, scatter_to_processes, gather_from_processes
from evo.variational.utils import init_states
from evo.models import BSC, SSSC

from tvutil.prepost import (
    OverlappingPatches,
    MultiDimOverlappingPatches,
)

from utils import stdout_logger, compute_psnr, print_dict
from viz import Visualizer


def get_args():
    p = argparse.ArgumentParser(
        description="Gaussian Image Denoising",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "input_path",
        type=str,
        help="Full path of H5 file holding target and noise image",
    )

    p.add_argument(
        "patch_size",
        type=int,
        nargs=2,
        help="Patch size, specified as (patch_height, patch_width)",
    )

    p.add_argument(
        "model",
        type=str,
        choices=["bsc", "sssc"],
        help="Generative model",
    )

    p.add_argument(
        "no_latents",
        type=int,
        help="Number of latents",
    )

    p.add_argument(
        "no_epochs",
        type=int,
        help="Number of epochs to train",
    )

    return p.parse_args()


def _get_images_from_h5(path, comm=MPI.COMM_WORLD):
    if comm.rank == 0:
        with h5py.File(path, "r") as f:
            target, noisy = f["target"][...], f["noisy"][...]
            assert np.ndim(target) == np.ndim(noisy)
            assert (np.ndim(target) == 3 and target.shape[-1] == 3) or np.ndim(
                target
            ) == 2, "Inputs must be either grayscale or RGB"
    else:
        target, noisy = None, None
    comm.Barrier()
    comm.bcast(target)
    comm.bcast(noisy)
    return target, noisy


def run_image_denoising(
    noisy_img,
    patch_size,
    model_name,
    no_latents,
    no_epochs,
    Ksize=20,
    parent_selection="fit",
    mutation_algorithm="randflip",
    no_parents=10,
    no_children=2,
    no_generations=1,
    bitflip_prob=None,
    save_theta_all_epochs=False,
    target_img=None,
    eval_fn=compute_psnr,
    viz_every=1,
    gif_framerate=None,
    comm=MPI.COMM_WORLD,
):
    """Image Denoising

    :param noisy_img: Noisy image array, (height, width) or (height, width, 3) shape expected
    :type noisy_img: np.ndarray
    :param patch_size: Patch size specified as [height, width]
    :type patch_size: List[Int]
    :param model_name: Generative Model, one of ["bsc", "sssc"]
    :type model_name: str
    :param no_latents: Model size: number of binary latents (H)
    :type no_latents: int
    :param no_epochs: Number of epochs to train
    :type no_epochs: int
    :param Ksize: Size of the K sets (i.e., S=|K])
    :type Ksize: int
    :param parent_selection: Selection operator (EVO hyperparameter, one of ["fit", "rand"])
    :type parent_selection: str
    :param mutation_algorithm: Mutation strategy (EVO hyperparameter, one of ["randflip", \
                               "sparseflip", "cross", "cross_randflip", "cross_sparseflip"])
    :type mutation_algorithm: str
    :param no_parents: Number of parental states to select per generation (EVO hyperparameter)
    :type no_parents: int
    :param no_children: Number of children to evolve per generation (EVO hyperparameter)
    :type no_children: int
    :param no_generations: Number of generations to evolve (EVO hyperparameter)
    :type no_generations: int
    :param bitflip_prob: Bitflip probability (EVO hyperparameter; only relevant for \
                         sparseflip-based mutation algorithms)
    :type bitflip_prob: float
    :param save_theta_all_epochs: Whether to log theta dictionary at every training epoch \
                                  (otherwise only last epoch will be logged)
    :type save_theta_all_epochs: bool
    :param target_img: Target image array (only used for evaluation, i.e., PSNR calculation)
    :type target_img: np.ndarray
    :param eval_fn: Method to evaluate performance (e.g., method to calculate PSNR)
    :type eval_fn: Callable
    :param viz_every: Create visualizations every X'th epoch
    :type viz_every: int
    :param gif_framerate: If specified, the training output will be additionally saved as animated \
                          gif. The framerate is given in frames per second. If not specified, no \
                          gif will be produced.
    :type gif_framerate: int
    :param comm: MPI communication inferface
    :type comm: mpi4py.MPI.Intracomm
    """

    # sanity checks
    assert model_name in ["bsc", "sssc"]
    assert no_epochs > 0
    assert Ksize > no_parents
    assert parent_selection in ["fit", "rand"]
    assert mutation_algorithm in [
        "randflip",
        "sparseflip",
        "cross",
        "cross_randflip",
        "cross_sparseflip",
    ]

    # define directory to save results
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%y-%m-%d_%H-%M-%S")
    output_directory = "./output/{}".format(
        os.environ["SLURM_JOBID"] if "SLURM_JOBID" in os.environ else timestamp
    )
    if not os.path.exists(output_directory) and comm.rank == 0:
        os.makedirs(output_directory)

    # initialize logs
    h5output_file = os.path.join(output_directory, "training.h5")
    dlog = DataLog()
    dlog.set_handler(
        [
            "*",
        ],
        StoreToH5,
        h5output_file,
        warnings=False,
    )
    dlog.set_handler(["F", "psnr"], TextPrinter)
    if save_theta_all_epochs:
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
    comm.Barrier()

    # extract patches from noisy image
    isrgb = np.ndim(noisy_img) == 3
    OVP = MultiDimOverlappingPatches if isrgb else OverlappingPatches
    ovp = OVP(noisy_img, *patch_size, patch_shift=1) if comm.rank == 0 else None
    Y = ovp.get().T if comm.rank == 0 else None
    comm.Barrier()

    # instantiate model
    no_channels = 3 if isrgb else 1
    D, H = patch_size[0] * patch_size[1] * no_channels, no_latents
    MODEL = {"bsc": BSC, "sssc": SSSC}[model_name]
    model = MODEL(D, H, Ksize)

    # distribute data to MPI processes
    pprint("Scattering data to processes")
    my_y, split_sizes, displacements = scatter_to_processes(Y)
    my_N = my_y.shape[0]
    my_data = {
        "y": my_y,  # local dataset
        "x_infr": np.logical_not(np.isnan(my_y)),  # reliable for learning
        "x": np.zeros_like(my_y),  # not x will be reconstructed
    }

    # initialize model and variational parameters
    pprint("Initializing model parameters")
    theta = model.check_params(model.standard_init(my_data))
    dlog.append_all({"{}_init".format(k): v for k, v in theta.items()})
    pprint("Initializing variational parameters")
    my_suff_stat = init_states(
        my_N,
        Ksize,
        H,
        parent_selection,
        mutation_algorithm,
        no_parents,
        no_children,
        no_generations,
        bitflip_prob,
    )

    # initialize visualizer
    pprint("Initializing visualizer")
    visualizer = (
        Visualizer(
            output_directory=output_directory,
            noisy_image=noisy_img,
            patch_size=patch_size,
            viz_every=viz_every,
            target_image=target_img,
            gif_framerate=gif_framerate,
            topk_gfs=np.min([50, H]),
        )
        if comm.rank == 0
        else None
    )
    if model_name == "bsc":
        pprint(
            "Note: The BSC M-step yields a single prior value pi=1/H sum_h {pi_h}; visualized "
            "will be the pi_h."
        )
    comm.Barrier()

    if comm.rank == 0:
        print("Using configuration as follows:")
        print_dict(
            {
                "output_directory": output_directory,
                "no_channels": no_channels,
                "no_epochs": no_epochs,
                "patch_size": patch_size,
                "no_latents": no_latents,
                "model_name": model_name,
                "Ksize": Ksize,
                "parent_selection": parent_selection,
                "mutation_algorithm": mutation_algorithm,
                "no_parents": no_parents,
                "no_children": no_children,
                "no_generations": no_generations,
                "bitflip_prob": bitflip_prob,
            }
        )

    for e in range(no_epochs):
        dlog.progress("Epoch {} of {}".format(e + 1, no_epochs))
        start = time.time()

        # run training epochs
        do_reconstruction = e == 0 or (e + 1) % viz_every == 0
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
        if do_reconstruction and (comm.rank == 0):
            reco_img = ovp.set_and_merge(Y_rec_T)
            dlog.append("reco", reco_img)
            if target_img is not None:
                psnr = compute_psnr(target_img, reco_img)
                dlog.append("psnr", psnr)

        # visualize
        if comm.rank == 0:
            visualizer.process_epoch(epoch=e + 1, theta=theta, rec=reco_img)
        comm.Barrier()

        pprint("\tTotal epoch runtime : %.2f s" % (time.time() - start))

    comm.Barrier()

    dlog.close()

    pprint("Finished")

    if comm.rank == 0:
        visualizer.finalize()

    comm.Barrier()


if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    args = get_args()
    if comm.rank == 0:
        print("Arguments were specified as follows:")
        print_dict(vars(args))

    target_img, noisy_img = _get_images_from_h5(args.input_path)

    run_image_denoising(
        noisy_img,
        args.patch_size,
        args.model,
        args.no_latents,
        args.no_epochs,
        target_img=target_img,
    )
