# -*- coding: utf-8 -*-
# Copyright (C) 2017 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import numpy as np
from itertools import combinations
import warnings


def randflip(parents, n_children, sparseness, p_bf):
    """Evolve new states through andom uniform bitflips

    :param parents: Parental states
    :type parents: np.ndarray
    :param n_children: Number of children (new states) to evolve
    :type n_children: in
    :param sparseness: Un-used (for consistency with interface of sparseness-driven EAs)
    :type sparseness: int
    :param p_bf: Un-used (for consistency with interface of sparseness-driven EAs)
    :type p_bf: float
    :return: Evolved states
    :type return: np.ndarray
    """

    # Each parent is "repeated" n_children times and inserted in children. We then flips bits in
    # the children states
    children = np.repeat(parents, n_children, axis=0)

    # Indices to be flipped. Bitflips for a given parent are ensured to be unique.
    n_parents, H = parents.shape
    ind_flip = np.argpartition(np.random.rand(n_parents, H), n_children - 1, axis=1)[
        :, :n_children
    ]  # is (n_parents, n_children)
    ind_flip_flat = (
        ind_flip.flatten()
    )  # [ parent1bitflip1, parent1bitflip2, parent2bitflip1, parent2bitflip2 ]

    # for each new state (0 to n_children*n_parents-1), flip bit at the position indicated by
    # ind_flip_flat
    children[np.arange(n_children * n_parents), ind_flip_flat] = np.logical_not(
        children[np.arange(n_children * n_parents), ind_flip_flat]
    )
    return children


def sparseflip(parents, n_children, sparseness, p_bf):
    """Take a set of parent bitstrings, generate n_children new bitstrings by performing bitflips
    # on each of the parents.
    The returned object has shape(parents.shape[0]*n_children, parents.shape[1])

    sparseness and p_bf regulate the probabilities of flipping each bit:
    - sparseness: the algorithm will strive to produce children with the given sparseness
      (expressed by number of ON bits)
    - p_bf: overall probability that a bit is flipped. the average number of bitflips per children
      is p_bf*parents.shape[1]

    :param parents: Parental states
    :type parents: np.ndarray
    :param n_children: Number of children (new states) to evolve
    :type n_children: in
    :param sparseness: Target sparseness
    :type sparseness: int
    :param p_bf: bitflit probability
    :type p_bf: float
    :return: Evolved states
    :type return: np.ndarray
    """
    assert p_bf is not None, "Please specify the bitflip probability"

    # Initialization
    n_parents, H = parents.shape
    s_abs = parents.sum(axis=1)  # is (no_parents)
    children = np.repeat(parents, n_children, axis=0)  # is (no_parents*no_children, H)
    eps = 1e-100

    # Probability to flip a 1 to a 0 and vice versa
    alpha = (
        (H - s_abs)
        * ((H * p_bf) - (sparseness - s_abs))
        / ((sparseness - s_abs + H * p_bf) * s_abs + eps)
    )  # is (no_parents)
    p_0 = (H * p_bf) / (H + (alpha - 1.0) * s_abs + eps)  # is (no_parents,)
    p_1 = alpha * p_0  # is (no_parents,)

    # Array handling
    p_0 = np.repeat(
        np.repeat(p_0[:, None], H, axis=1), n_children, axis=0
    )  # is (no_parents*no_children, H)
    p_1 = np.repeat(
        np.repeat(p_1[:, None], H, axis=1), n_children, axis=0
    )  # is (no_parents*no_children, H)
    p = np.empty_like(p_0, dtype=float)
    p[children] = p_1[children]
    p[np.logical_not(children)] = p_0[np.logical_not(children)]

    # Determine bits to be flipped and do the bitflip
    flips = np.random.random((n_parents * n_children, H)) < p
    children[flips] = np.logical_not(children[flips])

    return children


def cross(parents):
    """Parents are crossed in any possible way leading to
    n_parents * ( n_parents - 1 ) children.

    :param parents: Parental states
    :type parents: np.ndarray
    :return: Evolved states
    :type return: np.ndarray
    """

    # Initialization
    n_parents, H = parents.shape
    ind_children = np.arange(2)

    # All combinations of parents lead to n_parents*(n_parents-1) new children
    children = np.empty((n_parents * (n_parents - 1), H), dtype=bool)
    for p in combinations(range(n_parents), 2):
        # Cross tails
        cp = np.random.randint(low=1, high=H)
        children[ind_children] = parents[p, :]
        children[ind_children, cp:] = parents[p[-1::-1], cp:]
        ind_children += 2
    return children


def cross_randflip(parents, n_children, sparseness, p_bf):
    children = randflip(cross(parents), 1, sparseness, p_bf)
    return children


def cross_sparseflip(parents, n_children, sparseness, p_bf):
    children = sparseflip(cross(parents), 1, sparseness, p_bf)
    return children


def fitparents(candidates, n_parents, lpj):
    lpj_fitness = lpj - 2 * np.min([np.min(lpj), 0.0])
    try:
        lpj_fitness = lpj_fitness / lpj_fitness.sum()
    except Exception:
        print(lpj)
    return candidates[
        np.random.choice(candidates.shape[0], size=n_parents, replace=False, p=lpj_fitness)
    ]  # is (n_parents, H)


def randparents(candidates, n_parents, lpj=None):
    return candidates[np.random.choice(candidates.shape[0], size=n_parents, replace=False)]


def evolve_states(my_suff_stat, model_params, eval_lpj):
    """
    Take K states s of size H and return n_parents*n_children*n_generations new states and their
    log-pseudo-joints.

    :param my_suff_stat: Storage containing hyperparameters of genetic algorithms, as well as the
                         current K^{n} and respective log-pseudo joints
    :param my_suff_stat: dict
    :param model_params: Current Thetas
    :type model_params: dict
    :param eval_lpj: Callable to evaluate log-pseudo joints
    :type eval_lpj: callable
    :return: evolved states and respective log-pseudo joints
    :type return: tuple

    Each generation of new states is obtained by selecting `n_parents` parents from the previous
    generation following the strategy indicated by `parent_selection` and then mutating each
    parent `n_children` times following the strategy indicated by `mutation_algorithm`.

    parent_selection can be one of the following:
    - randparents
        parents are selected by sampling from a random uniform distribution
    - fitparents
        parents are selected using fitness-proportional sampling

    mutation_algorithm can be one of the following:
    - randflip
        each children is obtained by flipping one bit of the parent. every bit has the same
        probability of being flipped.
    - sparseflip
        each children is obtained by flipping bits in the parent.
        the probability of each bit being flipped depends on the sparseness and p_bh parameters
        (see method's description)
    - cross
        children are generated by one-point-crossover of the parents. each parent is crossed-over
        with each other parent at a point chosen via random uniform sampling.
    - cross_randflip
        as above, but the children additionally go through mutation_random_uniform
    - cross_sparseflip
        as above, but the children additionally go through mutation_sparseness_driven

    Pre-conditions: H >= n_children, K >= n_parents
    """
    lpj = my_suff_stat["this_lpj"]
    s = my_suff_stat["this_states"]
    incl = my_suff_stat["incl"]
    permanent = my_suff_stat["permanent"]
    n_parents = my_suff_stat["n_parents"]
    n_children = my_suff_stat["n_children"]
    n_generations = my_suff_stat["n_generations"]
    parent_selection = my_suff_stat["parent_selection"]
    mutation_algorithm = my_suff_stat["mutation_algorithm"]
    bitflip_prob = my_suff_stat["bitflip_prob"]
    sparseness = model_params["piH"]

    K, H = s.shape
    s_unique = np.concatenate((incl, s), axis=0)
    lpj_unique = lpj
    n_unique = s_unique.shape[0]
    this_start_ind = 0

    # don't consider background unit
    if permanent["background"]:
        ind_par = H - 1
    else:
        ind_par = H

    # print("g%i: s_unique=\n%s\n" % (-1, s_unique.astype(int)))
    for g in range(n_generations):

        if g == 0:
            parents = parent_selection(s, np.min([K, n_parents]), lpj)
        else:
            parents = parent_selection(
                new_states[slice_ind],  # noqa
                np.min([len(slice_ind), n_parents]),  # noqa
                new_lpj[slice_ind],  # noqa
            )

        # generate children
        this_states = mutation_algorithm(parents[:, :ind_par], n_children, sparseness, bitflip_prob)

        # eventually add background unit
        if permanent["background"]:
            this_states = np.concatenate(
                (this_states, np.ones((this_states.shape[0], 1), dtype=bool)), axis=1
            )

        # print("g%i: this_states=\n%s\n" % (g, this_states.astype(int)))

        if g == 0:
            # Allocation is performed to ensure that the no_children per generation has correctly
            # been determined
            n_children_per_gen = this_states.shape[0]
            new_states = np.zeros((n_children_per_gen * n_generations, H), dtype=bool)
            new_lpj = np.zeros(n_children_per_gen * n_generations)
            new_and_unique = np.zeros(new_lpj.size, dtype=bool)

        # check for already existing ones
        s_conc = np.ascontiguousarray(np.concatenate((s_unique, this_states), axis=0), dtype=int)
        _, ind_new_uniq = np.unique(
            s_conc.view(np.dtype((np.void, s_conc.dtype.itemsize * H))),
            return_index=True,
        )
        ind_new_uniq = ind_new_uniq[ind_new_uniq >= n_unique]
        this_n_new_uniq = ind_new_uniq.size

        # print("g%i: this_n_new_uniq=\n%s\n" % (g, this_n_new_uniq))

        # store new and unique states and compute log-pseudo-joint
        this_end_ind_new_uniq = this_start_ind + this_n_new_uniq
        if this_n_new_uniq > 0:
            slice_ind_new_uniq = range(this_start_ind, this_end_ind_new_uniq)
            new_states[slice_ind_new_uniq] = s_conc[ind_new_uniq, :].astype(np.bool)
            new_lpj[slice_ind_new_uniq] = eval_lpj(new_states[slice_ind_new_uniq])
            new_and_unique[slice_ind_new_uniq] = True

        # copy new but not unique states together with their lpjs
        s_conc_ = np.ascontiguousarray(s_conc[::-1], dtype=int)
        _, ind_old = np.unique(
            s_conc_.view(np.dtype((np.void, s_conc_.dtype.itemsize * H))),
            return_index=True,
        )
        ind_old = (
            ind_old[
                np.logical_and(
                    ind_old >= n_children_per_gen,
                    ind_old < (n_children_per_gen + n_unique - 1),
                )
            ]
            - n_children_per_gen
        )  # '-1' to neglect all-zero state
        if ind_old.size > 0:
            ind_tmp = np.arange(n_unique - 1)
            ind_old = ind_tmp[::-1][ind_old]
            ind_old = np.setdiff1d(ind_tmp, ind_old)
            this_n_new_not_uniq = ind_old.size
            this_end_ind = this_end_ind_new_uniq + this_n_new_not_uniq
            slice_ind_new_not_uniq = range(this_end_ind_new_uniq, this_end_ind)
            new_states[slice_ind_new_not_uniq] = s_unique[ind_old + 1].astype(bool)
            new_lpj[slice_ind_new_not_uniq] = lpj_unique[ind_old]
        else:
            this_end_ind = this_end_ind_new_uniq

        # print("g%i: this_n_new_not_uniq=\n%s\n" % (g, this_n_new_not_uniq))

        # update array of unique states
        if this_n_new_uniq > 0:
            s_unique = np.append(s_unique, new_states[slice_ind_new_uniq], axis=0)
            lpj_unique = np.append(lpj_unique, new_lpj[slice_ind_new_uniq])
            n_unique = s_unique.shape[0]

        # for selection of new parents
        if this_start_ind == this_end_ind:
            warnings.warn("No new and unique states. Skipping evolutionary loop.")
            break
        else:
            slice_ind = range(this_start_ind, this_end_ind)  # noqa
            this_start_ind = this_end_ind

    return new_states[new_and_unique], new_lpj[new_and_unique]
