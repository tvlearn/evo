# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import argparse


io_parser = argparse.ArgumentParser(add_help=False)
io_parser.add_argument(
    "--data_file",
    type=str,
    help=".npz file with training data set",
    default="./data/barbara-2k-patches.npz",
)

io_parser.add_argument(
    "--output_directory",
    type=str,
    help="Directory to write training output and visualizations to (will be output/<TIMESTAMP> "
    "if not specified)",
    default=None,
)

model_parser = argparse.ArgumentParser(add_help=False)
model_parser.add_argument(
    "--model", type=str, help="Generative Model", choices=["bsc", "sssc"], default="bsc"
)

model_parser.add_argument(
    "-H",
    type=int,
    help="Number of generative fields to learn",
    default=100,
)

variational_parser = argparse.ArgumentParser(add_help=False)
variational_parser.add_argument(
    "--Ksize",
    type=int,
    help="Size of the K sets (i.e., S=|K])",
    default=15,
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
    default=5,
)

variational_parser.add_argument(
    "--no_children",
    type=int,
    help="Number of children to evolve per generation",
    default=2,
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
    default=200,
)


viz_parser = argparse.ArgumentParser(add_help=False)
viz_parser.add_argument(
    "--sort_gfs",
    action="store_true",
    help="Whether to visualize learned generative fields according to prior activation",
    default=False,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Feature Learning",
        parents=[io_parser, model_parser, variational_parser, experiment_parser, viz_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    return parser.parse_args()
