# -*- coding: utf-8 -*-
# Copyright (C) 2017 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import numpy as np
from itertools import combinations
from evo.utils.parallel import pprint
from evo.variational.eas import (
    randflip,
    sparseflip,
    cross,
    cross_randflip,
    cross_sparseflip,
    fitparents,
    randparents,
)


def _init_lpj_and_state_arrays(N, S, H, p_init_Kn=None, permanent=None):
    """Helper function to Initialize the variational parameters, i.e. the state array K^{n}. This
    method ensures that each K^{n} containes unique states. If singletons are permanently considered
    for learning, it is ensured that they are not included in the initial K^{n}.

    :param N: Number of data points
    :type  N: int
    :param S: Number of variational states S=|K^{n}| per data point
    :type  S: int
    :param H: Number of latent dimensions
    :type  H: int
    :param p_init_Kn: Bernoulli probability to draw initial states, defaults to 1/H if not
                      specified
    :type  p_init_Kn: float
    :param permanent: Dictionary to specify whether permanent states shall be evaluated
    :type  permanent: dict
    :return: storage containing initial K^{n}, and array to store respective log-pseudo joints
    :type return: dict
    """

    if permanent is None:
        permanent = {"background": False, "allzero": False, "singletons": False}

    if permanent["background"]:
        S_perm = 0
        H_ = H - 1
    else:
        H_ = H
        if permanent["allzero"] == 1 and permanent["singletons"] == 0:
            S_perm = 1
        # elif permanent["allzero"] == 1 and permanent["singletons"] == 1:
        #     S_perm = H_ + 1
        #     incl = np.concatenate((np.zeros((1,H_), dtype=bool), np.eye(H_, dtype=bool)), axis=0)
        else:
            S_perm = 0
    incl = np.zeros((S_perm, H_), dtype=bool)
    exact_esteps = S == 2 ** H_

    # compute full state array for potential likelihood evaluations
    if H_ < 12:
        sl = []
        for g in range(H_ + 1):
            for s in combinations(range(H_), g):
                sl.append(np.array(s, dtype=np.int8))
        ss_size = len(sl)
        sm = np.zeros((ss_size, H_), dtype=bool)
        for i in range(ss_size):
            s = sl[i]
            sm[i, s] = True
    else:
        sm = None

    if exact_esteps:
        assert H_ < 12, "Exact E-steps too expensive for H={})".format(H_)
        pprint("Computing exact E-steps")

        if permanent["background"]:
            ss = np.concatenate((sm, np.ones((ss_size, 1), dtype=bool)), axis=1)
            lpj = np.empty((N, 2 ** H_))
        else:
            incl = np.zeros((S_perm, H_), dtype=bool)
            lpj = np.empty((N, S + S_perm))
            if permanent["allzero"] == 1 and permanent["singletons"] == 0:
                ss = sm[1:, :].copy()
            else:
                ss = sm.copy()

        ss = np.tile(ss[None, :, :], (N, 1, 1))

    else:

        if p_init_Kn is None:
            p_init_Kn = 1.0 / H

        # Allocation
        lpj = np.empty((N, S + S_perm))
        ss = np.empty((N, S, H), dtype=bool)

        if permanent["background"]:
            ss[:, :, -1] = True

        for n in range(N):

            # Draw S samples from prior
            this_p = np.random.random(size=(S, H_))
            this_s = this_p < p_init_Kn

            # Ensure initial states are unique for each sample
            s_conc = np.ascontiguousarray(np.concatenate((incl, this_s), axis=0), dtype=int)
            _, ind_uniq = np.unique(
                s_conc.view(np.dtype((np.void, s_conc.dtype.itemsize * H_))),
                return_index=True,
            )
            ind_uniq = ind_uniq[ind_uniq >= S_perm]
            # if permanent["background"]:
            #     ind_uniq = ind_uniq[s_conc[ind_uniq,:][:,-1]==1]
            this_s = s_conc[ind_uniq, :].astype(np.bool)

            while this_s.shape[0] < S:
                this_p_ = np.random.random(size=(S, H_))
                this_s_new = this_p_ < p_init_Kn
                s_conc = np.ascontiguousarray(
                    np.concatenate((incl, this_s, this_s_new), axis=0), dtype=int
                )

                # The most appropriate method np.unique(...,axis=0) for obtaining unique states
                # is only available in numpy version 1.13. Here we use a workaround that is
                # supported also by older versions.
                _, ind_uniq = np.unique(
                    s_conc.view(np.dtype((np.void, s_conc.dtype.itemsize * H_))),
                    return_index=True,
                )
                ind_uniq = ind_uniq[ind_uniq >= (S_perm + this_s.shape[0])]
                # if permanent["background"]:
                #     ind_uniq = ind_uniq[s_conc[ind_uniq,:][:,-1]==1]
                this_s_new = s_conc[ind_uniq, :].astype(np.bool)

                this_s = np.concatenate((this_s, this_s_new), axis=0)

            ss[n, :, :H_] = this_s[:S]

    if permanent["background"]:
        incl = np.zeros((S_perm, H), dtype=bool)

    my_suff_stat = {
        "ss": ss,
        "lpj": lpj,
        "permanent": permanent,
        "incl": incl,
        "S_perm": S_perm,
        "sm": sm,
    }

    return my_suff_stat


def init_states(
    N,
    S,
    H,
    parent_selection,
    mutation_algorithm,
    no_parents,
    no_children,
    no_generations,
    bitflip_prob=None,
    Mprime=None,
    p_init_Kn=None,
    permanent=None,
):
    """Initialize the initial K^{n}. Write hyperparameters for the evolutionary algorithms to
    dictionary.

    :param N: Number of data points
    :type  N: int
    :param S: Number of variational states S=|K^{n}| per data point
    :type  S: int
    :param H: Number of latent dimensions
    :type  H: int
    :param parent_selection: Parent selection procedure
    :type parent_selection: str
    :param mutation_algorithm: Mutation strategy
    :type mutation_algorithm: str
    :param no_parents: Number of parents to select per generations
    :type no_parents: int
    :param no_children: Number of children to evolve per parent
    :type no_children: int
    :param no_generations: Number of generations to evolve
    :type no_generations: int
    :param bitflip_prob: Probability for sparseness-driven bitflips
    :type bitflip_prob: float
    :param Mprime: Maximum number of new states to replace (defaults to S)
    :type Mprime: int
    :param p_init_Kn: Bernoulli probability to draw initial states, defaults to 1/H if not
                      specified
    :type  p_init_Kn: float
    :param permanent: Dictionary to specify whether permanent states shall be evaluated
    :type  permanent: dict
    :return: storage containing initial K^{n}, and array to store respective log-pseudo joints
    :type return: dict
    """
    my_suff_stat = _init_lpj_and_state_arrays(N, S, H, p_init_Kn=p_init_Kn, permanent=permanent)

    if "cross" in mutation_algorithm:
        no_children = no_parents - 1
        pprint(
            "Setting no_children to pre-determined value `no_parents - 1` "
            "({}) when using crossover".format(no_parents - 1)
        )

    assert no_parents <= S
    my_suff_stat["n_parents"] = no_parents
    my_suff_stat["n_children"] = no_children
    my_suff_stat["n_generations"] = no_generations
    my_suff_stat["parent_selection"] = {"fit": fitparents, "rand": randparents}[parent_selection]
    my_suff_stat["mutation_algorithm"] = {
        "randflip": randflip,
        "sparseflip": sparseflip,
        "cross": cross,
        "cross_randflip": cross_randflip,
        "cross_sparseflip": cross_sparseflip,
    }[mutation_algorithm]
    my_suff_stat["bitflip_prob"] = bitflip_prob
    if Mprime is None:
        Mprime = S
    else:
        assert Mprime <= S
    my_suff_stat["Mprime"] = Mprime

    return my_suff_stat


def vary_Kn(
    lpj_old,
    lpj_new,
    lpj,
    states,
    states_new,
    H,
    S,
    S_perm,
    incl,
    Mprime,
    unification=True,
    reject_worse=True,
):
    """Compute a variation of K^{n} for given new log-pseudo joints
    :param lpj_old: Log-pseudo joints of old states
    :type lpj_old: np.ndarray
    :param lpj_new: Log-pseudo joints of new states
    :type lpj_new: np.ndarray
    :param lpj: Free energies and M-steps look log-pseudo joints up in this array
    :type lpj: np.ndarray
    :param states: Free energies and M-steps look states up in this array
    :type states: np.ndarray
    :param states_new: New states
    :type states_new: np.ndarray
    :param H: Number of latent dimensions
    :type H: int
    :param S: Number of states per S=|K^{n}|
    :type S: int
    :param S_perm: Number of permanent states that are evaluated
    :type S_perm: int
    :param incl: Array containing permanent states
    :type incl: np.ndarray
    :param Mprime: Maximum number of states to replace per datapoint per E-step
    :type Mprime: int
    :param unification: Whether to unify old and new log-pseudo joints to determine best states
    :type unification: bool
    :param reject_worse: If True and unification is False, either the set of old states or a set
                         of best new states will be used as new states, depending on which has the
                         higher cumulated joints. Otherwise only the set of best new states will be
                         used as new states.
    :type reject_worse: bool
    :return: average number of new and unique states obtained through the evolutionary procedure,
             average number of states replaced in the K^{n} sets
    :type return: tuple
    """

    # Find unique new states
    s_conc = np.ascontiguousarray(np.concatenate((incl, states, states_new), axis=0), dtype=int)
    _, ind_uniq = np.unique(
        s_conc.view(np.dtype((np.void, s_conc.dtype.itemsize * H))), return_index=True
    )
    # _, ind_uniq = np.unique(s_conc.view(np.dtype((np.void, H))),return_index=True)
    # _, ind_uniq = np.unique(s_conc, axis=0, return_index=True)
    ind_uniq_ = ind_uniq[ind_uniq >= (S + S_perm)]

    if unification:

        # Remove duplicate states
        states_new = s_conc[ind_uniq_, :].astype(np.bool)
        lpj_new = lpj_new[ind_uniq_ - (S_perm + S)]

        # Find highest joints for new states and lowest joints for old states
        Mprime_ = min([lpj_new.size, Mprime])
        ind_new_highest = np.argpartition(lpj_new, -Mprime_)[-Mprime_:]
        ind_lowest = np.argpartition(lpj_old, Mprime_ - 1)[:Mprime_]

        # Swap the very best new states with the very worst old states
        # NEW VERSION by E. Guiraud
        sortedInds = np.array(
            np.unravel_index(
                np.argsort(
                    np.stack((lpj_new[ind_new_highest], lpj_old[ind_lowest]))
                    if len(ind_lowest) > 0
                    else lpj_new[ind_new_highest],
                    axis=None,
                )[::-1],
                (2, ind_new_highest.size),
            )
        )
        bestInd = sortedInds[:, :Mprime_]
        goodSInd = ind_new_highest[bestInd[1, bestInd[0] == 0]]
        worstInd = sortedInds[:, -1 : -1 - Mprime_ : -1]
        badKInd = ind_lowest[worstInd[1, worstInd[0] == 1]]
        for j in range(goodSInd.size):
            # Insert the new configuration
            states[badKInd[j]] = states_new[goodSInd[j]]
            assert lpj_new[goodSInd[j]] >= lpj_old[badKInd[j]]
            lpj_old[badKInd[j]] = lpj_new[goodSInd[j]]

        lpj[:] = lpj_old
        my_S_nunique_add = ind_uniq_.size
        my_S_sub_add = goodSInd.size

    else:

        if reject_worse and (lpj_new.sum() < lpj_old.sum()):
            lpj[:] = lpj_old
            my_S_nunique_add = 0
            my_S_sub_add = 0
        else:
            lpj[:] = lpj_new
            states[:, :] = states_new
            my_S_nunique_add = ind_uniq_.size
            my_S_sub_add = ind_uniq_.size

    return my_S_nunique_add, my_S_sub_add
