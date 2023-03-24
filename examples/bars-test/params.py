# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import argparse


output_parser = argparse.ArgumentParser(add_help=False)
output_parser.add_argument(
    "--output_directory",
    type=str,
    help="Directory to write training output and visualizations to (will be output/<TIMESTAMP> "
    "if not specified)",
    default=None,
)

bars_parser = argparse.ArgumentParser(add_help=False)
bars_parser.add_argument(
    "-H",
    type=int,
    help="Number of bars",
    default=8,
)

bars_parser.add_argument(
    "--bar_amp",
    type=float,
    help="Bar amplitude",
    default=10.0,
)

bars_parser.add_argument(
    "--neg_bars",
    action="store_true",
    help="Whether to randomly make bar amplitudes negative",
    default=False,
)

bars_parser.add_argument(
    "--no_data_points",
    type=int,
    help="Number of datapoints",
    default=500,
)

bsc_parser = argparse.ArgumentParser(add_help=False)
bsc_parser.add_argument(
    "--pi_gen",
    type=float,
    help="Sparsity used for data generation (defaults to 2/H if not specified)",
    default=None,
)

bsc_parser.add_argument(
    "--sigma_gen",
    type=float,
    help="Noise level used for data generation",
    default=1.0,
)

sssc_parser = argparse.ArgumentParser(add_help=False)
sssc_parser.add_argument(
    "--pi_gen",
    type=float,
    help="Sparsity used for data generation (defaults to 2/H if not specified)",
    default=None,
)

sssc_parser.add_argument(
    "--sigma_gen",
    type=float,
    help="Noise level used for data generation",
    default=1.0,
)

sssc_parser.add_argument(
    "--mu_gen",
    type=float,
    help="Generative value of latent means",
    default=0.0,
)

sssc_parser.add_argument(
    "--psi_gen",
    type=float,
    help="Generative value of latent standard deviation",
    default=1.0,
)

variational_parser = argparse.ArgumentParser(add_help=False)
variational_parser.add_argument(
    "--Ksize",
    type=int,
    help="Size of the K sets (i.e., S=|K])",
    default=20,
)

variational_parser.add_argument(
    "--parent_selection",
    type=str,
    help="Selection operator",
    choices=["fit", "rand"],
    default="fit",
)

variational_parser.add_argument(
    "--mutation_algorithm",
    type=str,
    help="Mutation strategy",
    choices=["randflip", "sparseflip", "cross", "cross_randflip", "cross_sparseflip"],
    default="randflip",
)

variational_parser.add_argument(
    "--no_parents",
    type=int,
    help="Number of parental states to select per generation",
    default=10,
)

variational_parser.add_argument(
    "--no_children",
    type=int,
    help="Number of children to evolve per generation",
    default=1,
)

variational_parser.add_argument(
    "--no_generations",
    type=int,
    help="Number of generations to evolve",
    default=1,
)

variational_parser.add_argument(
    "--bitflip_prob",
    type=float,
    help="Bitflip probability (only relevant for sparseflip-based mutation algorithms)",
    default=None,
)


experiment_parser = argparse.ArgumentParser(add_help=False)
experiment_parser.add_argument(
    "--no_epochs",
    type=int,
    help="Number of epochs to train",
    default=40,
)


viz_parser = argparse.ArgumentParser(add_help=False)
viz_parser.add_argument(
    "--viz_every",
    type=int,
    help="Create visualizations every X'th epoch. Set to no_epochs if not specified.",
    default=1,
)

viz_parser.add_argument(
    "--gif_framerate",
    type=str,
    help="Frames per second for gif animation (e.g., 2/1 for 2 fps). If not specified, no gif will "
    "be produced.",
    default=None,
)


def get_args():
    parser = argparse.ArgumentParser(prog="Standard Bars Test")
    algo_parsers = parser.add_subparsers(help="Select algorithm to run", dest="algo")
    comm_parents = [output_parser, bars_parser, variational_parser, experiment_parser, viz_parser]
    algo_parsers.add_parser(
        "ebsc",
        help="Run experiment with EBSC",
        parents=comm_parents
        + [
            bsc_parser,
        ],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    algo_parsers.add_parser(
        "emca",
        help="Run experiment with EMCA",
        parents=comm_parents
        + [
            bsc_parser,
        ],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    algo_parsers.add_parser(
        "es3c",
        help="Run experiment with ES3C",
        parents=comm_parents
        + [
            sssc_parser,
        ],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    return parser.parse_args()
