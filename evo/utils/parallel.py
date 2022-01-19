# -*- coding: utf-8 -*-

from __future__ import division

import sys
import numpy as np
from mpi4py import MPI


typemap = {
    np.dtype("float64"): MPI.DOUBLE,
    np.dtype("float32"): MPI.FLOAT,
    np.dtype("int16"): MPI.SHORT,
    np.dtype("int32"): MPI.INT,
    np.dtype("int64"): MPI.LONG,
    np.dtype("uint16"): MPI.UNSIGNED_SHORT,
    np.dtype("uint32"): MPI.UNSIGNED_INT,
    np.dtype("uint64"): MPI.UNSIGNED_LONG,
}


def pprint(obj="", comm=MPI.COMM_WORLD, end="\n"):
    """
    Parallel print: Make sure only one of the MPI processes
    calling this function actually prints something. All others
    (comm.rank != 0) return without doing enything.

    Source: https://github.com/ml-uol/prosper/blob/master/prosper/utils/parallel.py

    For LICENSING and COPYRIGHT for this function see prosper's license at:
    https://github.com/ml-uol/prosper/blob/master/LICENSE.txt
    """
    if comm.rank != 0:
        return

    if isinstance(obj, str):
        sys.stdout.write(obj + end)
    else:
        sys.stdout.write(repr(obj))
        sys.stdout.write(end)
        sys.stdout.flush()


def allmean(my_a, axis=None, dtype=None, out=None, comm=MPI.COMM_WORLD):
    """Parallel (collective) version of numpy.mean

    Source: https://github.com/ml-uol/prosper/blob/master/prosper/utils/parallel.py

    For LICENSING and COPYRIGHT for this function see prosper's license at:
    https://github.com/ml-uol/prosper/blob/master/LICENSE.txt
    """
    shape = my_a.shape
    if axis is None:
        N = comm.allreduce(my_a.size)
    else:
        N = comm.allreduce(shape[axis])

    my_sum = np.sum(my_a, axis, dtype)

    if my_sum is np.ndarray:
        sum = np.empty_like(my_sum)
        comm.Allreduce((my_sum, typemap[my_sum.dtype]), (sum, typemap[sum.dtype]))
        sum /= N
        return sum
    else:
        return comm.allreduce(my_sum) / N


def scatter_to_processes(to_scatter, comm=MPI.COMM_WORLD):
    """Split data set and scatter chunks to processes

    :param to_scatter: Dataset to be split and scattered; expected on rank 0
    :type to_scatter: np.ndarray
    :param comm: MPI communication inferface
    :type comm: mpi4py.MPI.Intracomm
    :return: Data chunks (living on ranks 0, 1, ..., comm.size-1), indices required to gather again
    :type return: np.ndarray

    Inspired by: https://stackoverflow.com/a/36082684

    Licensed under the Academic Free License version 3.0
    """
    if comm.rank == 0:
        chunks = np.array_split(
            to_scatter, comm.size, axis=0
        )  # Split input array by the no available cores
        split_sizes = []
        split_shapes = np.empty((len(chunks), 2), dtype=int)
        for i in range(0, len(chunks), 1):
            split_sizes = np.append(split_sizes, len(chunks[i]))
            split_shapes[i] = np.array(chunks[i].shape)
        split_sizes_input = split_sizes * np.prod(to_scatter.shape[1:])
        displacements_input = np.insert(np.cumsum(split_sizes_input), 0, 0)[0:-1]
    else:
        split_shapes = None
        split_sizes_input = None
        displacements_input = None
    split_sizes = comm.bcast(split_sizes_input, root=0).astype(int)
    split_shapes = comm.bcast(split_shapes, root=0).astype(int)
    displacements = comm.bcast(displacements_input, root=0).astype(int)

    # Create array to receive  subset of data on each core, where rank specifies + the core
    chunk = np.zeros(split_shapes[comm.rank])

    # Scatter data
    comm.Scatterv([to_scatter, split_sizes_input, displacements_input, MPI.DOUBLE], chunk, root=0)

    return chunk, split_sizes, displacements


def gather_from_processes(chunk, split_sizes, displacements, comm=MPI.COMM_WORLD):
    """Gather data chunks on rank zero

    :param chunk: Data chunks, living on ranks 0, 1, ..., comm.size-1
    :type chunk: np.ndarray
    :param split_sizes: Chunk lenghts on individual ranks
    :type split_sizes: np.ndarray
    :param displacements: Chunk displacements (compare scatter_to_processes)
    :type displacements: np.ndarray
    :return: Dataset gathered again, living on rank 0
    :type return: np.ndarray

    Inspired by: https://stackoverflow.com/a/36082684

    Licensed under the Academic Free License version 3.0
    """
    comm.Barrier()
    total_length = np.array(chunk.shape[0])
    gathered = np.empty((comm.allreduce(total_length), chunk.shape[1]), dtype=chunk.dtype)
    comm.Gatherv(chunk, [gathered, split_sizes, displacements, MPI.DOUBLE], root=0)
    return gathered
