# -*- coding: utf-8 -*-

from __future__ import division

import sys
import numpy as np
from mpi4py import MPI


typemap = {
    np.dtype("float64"): MPI.DOUBLE,
    np.dtype("float32"): MPI.FLOAT,
    np.dtype("bool"): MPI.BOOL,
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


def bcast_dtype(data, comm=MPI.COMM_WORLD):
    """Broadcast dtype of data on root rank.

    :param data: Tensor on root rank
    :type data: np.ndarray
    :param comm: MPI communication inferface
    :type comm: mpi4py.MPI.Intracomm
    :returns: dtype on each rank
    """
    ind_dtype = np.empty((1,), dtype=np.uint8)
    dtypes = list(typemap.keys())
    if comm.rank == 0:
        dtype = data.dtype
        ind_dtype[:] = [*map(str, dtypes)].index(str(dtype))
    ind_dtype = comm.bcast(ind_dtype)
    return dtypes[ind_dtype.item()]


def get_chunk_dimensions(to_split, no_chunks, axis=0):
    """Infer chunk dimensions as required as input by comm.Scatterv and comm.Gatherv

    :param to_split: Array to be split along specified axis into specified number of chunks
    :type to_split: np.ndarray
    :param no_chunks: Number to chunks
    :type no_chunks: int
    :return: (chunk shapes, chunk lenghts, displacements (compare scatter_to_processes))
    :type return: Tuple[np.ndarray, np.ndarray, np.ndarray]

    Inspired by: https://stackoverflow.com/a/36082684

    Licensed under the Academic Free License version 3.0
    """
    chunks = np.array_split(
        to_split, no_chunks, axis=axis
    )  # Split input array into specified number of chunks

    split_sizes = []
    split_shapes = np.empty((len(chunks), np.ndim(to_split)), dtype=int)
    for i in range(0, len(chunks), 1):
        split_sizes = np.append(split_sizes, len(chunks[i]))
        split_shapes[i] = np.array(chunks[i].shape)
    split_sizes *= np.prod(to_split.shape[1:])
    displacements = np.insert(np.cumsum(split_sizes), 0, 0)[0:-1]

    return split_shapes.astype(int), split_sizes.astype(int), displacements.astype(int)


def scatter_to_processes(to_scatter, comm=MPI.COMM_WORLD):
    """Split data set and scatter chunks to processes

    :param to_scatter: Array to be split and scattered; expected on rank 0
    :type to_scatter: np.ndarray
    :param comm: MPI communication inferface
    :type comm: mpi4py.MPI.Intracomm
    :return: Data chunks (living on ranks 0, 1, ..., comm.size-1)
    :type return: np.ndarray

    Inspired by: https://stackoverflow.com/a/36082684

    Licensed under the Academic Free License version 3.0
    """

    # get split dimensions
    split_shapes, split_sizes, displacements = (
        get_chunk_dimensions(to_scatter, comm.size) if comm.rank == 0 else (None, None, None)
    )
    split_sizes = comm.bcast(split_sizes, root=0)
    split_shapes = comm.bcast(split_shapes, root=0)
    displacements = comm.bcast(displacements, root=0)

    # Create array to receive subset of data on each core, where rank specifies + the core
    dtype = bcast_dtype(to_scatter)
    chunk = np.zeros(split_shapes[comm.rank]).astype(dtype)

    # Scatter data
    comm.Scatterv(
        [to_scatter, split_sizes, displacements, typemap[dtype]],
        chunk,
        root=0,
    )

    return chunk


def gather_from_processes(chunk, comm=MPI.COMM_WORLD):
    """Gather data chunks on rank zero

    :param chunk: Data chunks, living on ranks 0, 1, ..., comm.size-1
    :type chunk: np.ndarray
    :return: Dataset gathered again, living on rank 0
    :type return: np.ndarray

    Inspired by: https://stackoverflow.com/a/36082684

    Licensed under the Academic Free License version 3.0
    """
    comm.Barrier()

    total_length = np.array(chunk.shape[0])
    gathered = np.empty((comm.allreduce(total_length), *chunk.shape[1:]), dtype=chunk.dtype)
    _, split_sizes, displacements = get_chunk_dimensions(gathered, comm.size)

    comm.Gatherv(chunk, [gathered, split_sizes, displacements, typemap[chunk.dtype]], root=0)
    return gathered
