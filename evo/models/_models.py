# -*- coding: utf-8 -*-
# Copyright (C) 2017 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from __future__ import division
import numpy as np
from mpi4py import MPI
from abc import ABCMeta, abstractmethod
from scipy.special import logsumexp

import evo.utils.tracing as tracing
import evo.utils.parallel as parallel
from evo.variational.eas import evolve_states
from evo.variational.utils import vary_Kn


class Model:
    __metaclass__ = ABCMeta

    def __init__(self, D, H, S, to_learn=["W", "pi", "sigma"], comm=MPI.COMM_WORLD):
        """Abstract base class to derive concrete models.

        :param D: Number of observables
        :type  D: int
        :param H: Number of latent dimensions
        :type  H: int
        :param S: Number of variational states S=|K^{n}| per data point
        :type  S: int
        :param to_learn: List of model parameters Theta to update in the M-step
        :type to_learn: list
        :param comm: MPI communication inferface
        :type comm: mpi4py.MPI.Intracomm

        Inspired by https://github.com/ml-uol/prosper/blob/master/prosper/em/camodels/__init__.py
        ::CAModel.__init__

        For LICENSING and COPYRIGHT for the respective function in prosper see prosper's license
        at: https://github.com/ml-uol/prosper/blob/master/LICENSE.txt
        """
        self.comm = comm
        self.noise_policy = {}
        self.to_learn = to_learn
        self.D = D
        self.H = H
        self.S = S

        tol = 1e-5
        self.noise_policy = {
            "W": (-np.inf, +np.inf, False, None),
            "pi": (tol, 1.0 - tol, False, None),
            "sigma": (tol, +np.inf, False, None),
        }

        # Numerical stabilization for lpj computation
        self.B_max = 0.0
        self.B_max_shft = np.inf

    @abstractmethod
    def generate_from_hidden(self, model_params, my_hdata):
        """Generate data according to the BSC generative model. The latents are
        given in my_data['s'], the observed variables in my_data['y'].

        :param model_params: Generative model parameters
        :type  model_params: dict
        :param my_hdata: Storage containing latent states for data generation
        :type param: dict
        :return: Generated data, respective generative latent states, Mean of observables (Gaussian
                 mean
        :type return: dict
        """
        pass

    def generate_data(self, model_params, my_N):
        """Generate data according to the model. Internally uses generate_data_from_hidden.

        :param model_params: Generative model parameters
        :type  model_params: dict
        :param my_N: number of datapoints to generate on this MPI rank
        :type  my_N: int
        :return: Generated datapoints and generative latent states
        :type return: dict

        Source: https://github.com/ml-uol/prosper/blob/master/prosper/em/camodels/__init__.py
        ::CAModel.generate_data

        For LICENSING and COPYRIGHT for the respective function in prosper see prosper's license
        at: https://github.com/ml-uol/prosper/blob/master/LICENSE.txt
        """
        assert "pies" in model_params or "pi" in model_params
        W = model_params["W"]

        D, H_gen = W.shape
        pies = model_params["pies"] if "pies" in model_params else model_params["pi"]
        # np.random.seed(0)
        p = np.random.random(size=(my_N, H_gen))  # Create latent vector
        s = p <= pies  # Translate into boolean latent vector
        my_data = self.generate_from_hidden(model_params, {"s": s})

        return my_data

    @tracing.traced
    def check_params(self, model_params):
        """Sanity-check the given model parameters. Raises an exception if something
        is severely wrong.

        :param model_params: Model parameters Theta
        :type model_params: dict
        :return: Potentially adjusted Thetas
        :type return: dict

        Inspired by https://github.com/ml-uol/prosper/blob/master/prosper/em/__init__.py::
        Model.noisify_params

        For LICENSING and COPYRIGHT for the respective function in prosper see prosper's license
        at: https://github.com/ml-uol/prosper/blob/master/LICENSE.txt
        """

        comm = self.comm
        for param, policy in self.noise_policy.items():
            low_bound, up_bound, absify, low_bound_diagonal = policy
            new_pvalue = model_params[param]
            if np.isscalar(new_pvalue):  # Param to be noisified is scalar
                if comm.rank == 0:
                    if new_pvalue < low_bound:
                        print("check_params: Reset lower bound of %s" % param)
                        new_pvalue = low_bound
                    if new_pvalue >= up_bound:
                        print("check_params: Reset upper bound of %s" % param)
                        new_pvalue = up_bound
                    if absify:
                        print("check_params: Taking abs of %s" % param)
                        new_pvalue = np.abs(new_pvalue)
                    if (
                        low_bound_diagonal is not None
                    ):  # when using isotropic instead of full matrix
                        if new_pvalue < low_bound_diagonal:
                            print("check_params: Reset lower bound of %s (diagonal)" % param)
                            new_pvalue = low_bound_diagonal
                new_pvalue = comm.bcast(new_pvalue)
            else:
                if comm.rank == 0:
                    if (new_pvalue < low_bound).any():
                        print("check_params: Reset lower bound of %s" % param)
                    if (new_pvalue >= up_bound).any():
                        print("check_params: Reset upper bound of %s" % param)
                    new_pvalue = np.maximum(low_bound, new_pvalue)
                    new_pvalue = np.minimum(up_bound, new_pvalue)
                    if absify:
                        print("check_params: Taking abs of %s" % param)
                        new_pvalue = np.abs(new_pvalue)
                    if low_bound_diagonal is not None:
                        mask = np.diag(new_pvalue) < low_bound_diagonal
                        if mask.any():
                            print("check_params: Reset lower bound of %s (diagonal)" % param)
                        new_pvalue[np.diag(mask)] = low_bound_diagonal
                comm.Bcast([new_pvalue, MPI.DOUBLE])
            model_params[param] = new_pvalue

        return model_params

    @tracing.traced
    def step(self, model_params, my_suff_stat, my_data, do_reconstruction=False):
        """Perform an EM-step

        :param model_params: Current Thetas
        :type model_params: dict
        :param my_suff_stat: Storage containing current K^{n} and respective log-pseudo joints
        :param my_suff_stat: dict
        :param my_data: Local dataset including indices of reliable (non-missing) entries and
                        entries to be reconstructed
        :type my_data: dict
        :param do_reconstruction: Toggle whether to estimate data at this epoch
        :type do_reconstruction: bool
        :return: Free energy between E- and M-step, average number of new and unique states
                 obtained through the evolutionary generation of new states, average number of
                 states substituted within the K^{n} sets, updated model parameters Theta^new
        :type return: tuple

        Inspired by https://github.com/ml-uol/prosper/blob/master/prosper/em/camodels/__init__.py::
        CAModel.step

        For LICENSING and COPYRIGHT for the respective function in prosper see prosper's license
        at: https://github.com/ml-uol/prosper/blob/master/LICENSE.txt
        """

        # Sanity check model parameters
        model_params = self.check_params(model_params)

        # Do E-step and vary state space and calculate joint-probabilities
        F, S_nunique, S_sub = self.E_step(model_params, my_suff_stat, my_data)

        # Reconstruct data if necessary
        if do_reconstruction:
            self.reconstruct(my_data, my_suff_stat, model_params)

        # Do M-step and use joint-probabilities to derive new parameter set
        new_model_params = self.M_step(model_params, my_suff_stat, my_data)

        return F, S_nunique, S_sub, new_model_params

    @tracing.traced
    def standard_init(self, my_data, W_init=None, pi_init=None, sigma_init=None):
        """Standard initial estimation for model parameters.
        Each *W* raw is set to the average over the data plus WGN of mean zero
        and var *sigma*/4. *sigma* is set to the variance of the data around
        the computed mean. *pi* is set to 1./H . Returns a dict with the
        estimated parameter set with entries "W", "pi" and "sigma".

        :param my_data: Local dataset including indices of reliable (non-missing) entries and
                        entries to be reconstructed
        :type my_data: dict
        :param W_init: strategy to initialize W (one of np.ndarray, "random_uniform",
                       "normal", "data_mean"; if None, W will be initiliazed with a noisy data mean)
        :type W_init: np.ndarray, str, or None
        :param pi_init: Initial sparsity, will be set to 1/H if not specified
        :type pi_init: float or None
        :param sigma_init: Initial standard deviation, will be set to the average stabdard
                           deviation of the data if not specified
        :return: Initial model parameters Theta^{init}
        :type return: dict

        Based on https://github.com/ml-uol/prosper/blob/master/prosper/em/camodels/__init__.py::
        CAModel.standard_init

        For LICENSING and COPYRIGHT for the respective function in prosper see prosper's license
        at: https://github.com/ml-uol/prosper/blob/master/LICENSE.txt
        """
        comm = self.comm
        H = self.H
        my_y = my_data["y"]
        my_x_infr = my_data["x_infr"]
        my_N, D = my_y.shape
        assert D == self.D
        incmpl_data = not my_x_infr.all()

        # Calculate averarge y
        if not incmpl_data:
            y_mean = parallel.allmean(my_y, axis=0, comm=comm)  # shape: (D, )
        else:
            y_mean = np.zeros(D)  # shape: (D, )
            for n in range(my_N):
                this_y = my_y[n]
                this_x_infr = my_x_infr[n]
                y_mean[this_x_infr] += this_y[this_x_infr]
            y_mean = comm.allreduce(y_mean / my_N)

        # Calculate data variance
        if sigma_init is None:
            if not incmpl_data:
                tmp = parallel.allmean((my_y - y_mean) ** 2, axis=0, comm=comm)  # shape: (D, )
                sigma_init = np.sqrt(tmp.sum() / D)
            else:
                tmp = np.zeros(D)
                for n in range(my_N):
                    this_y = my_y[n]
                    this_x_infr = my_x_infr[n]
                    tmp[this_x_infr] += ((this_y[this_x_infr] - y_mean[this_x_infr])) ** 2
                sigma_sq = comm.allreduce(tmp.sum() / my_x_infr.flatten().sum())
                sigma_init = np.sqrt(sigma_sq)

            assert sigma_init > 0.0

        # Initial W
        if type(W_init) is not np.ndarray:
            if W_init == "random_uniform":
                W_init = comm.bcast(np.random.random((D, H)))
            elif W_init == "normal":
                W_init = comm.bcast(np.random.normal(0, 5, [D, H]))
            elif W_init == "data_mean":
                W_init = np.tile(y_mean[:, None], (1, H))
            else:
                noise = comm.bcast(np.random.normal(scale=sigma_init / 4.0, size=[D, H]))
                W_init = y_mean[:, None] + noise  # shape: (H, D)`

        # Initial pi
        if pi_init is None:
            pi_init = 1.0 / H

        return {"W": W_init, "pi": pi_init, "sigma": sigma_init}

    @abstractmethod
    def E_step_precompute(self, model_params, my_suff_stat, my_data):
        """Pre-evaluate state-independent terms that are repeatedly used in during E-/M-step

        :param model_params: Current Thetas
        :type model_params: dict
        :param my_suff_stat: Storage containing current K^{n} and respective log-pseudo joints
        :param my_suff_stat: dict
        :param my_data: Local dataset including indices of reliable (non-missing) entries and
                        entries to be reconstructed
        :type my_data: dict
        :return: log-pseudo joints
        :type return: np.ndarray
        """

    @abstractmethod
    def log_pseudo_joint_permanent_states(self, model_params, my_suff_stat, my_data):
        """Otionally evaluate log-pseudo-joints for permanent states (e.g., all-zero state,
        or singleton states). Skipped if my_suff_stat["permanent"] is empty.

        :param model_params: Current Thetas
        :type model_params: dict
        :param my_suff_stat: Storage containing current K^{n} and respective log-pseudo joints
        :param my_suff_stat: dict
        :param my_data: Local dataset including indices of reliable (non-missing) entries and
                        entries to be reconstructed
        :type my_data: dict
        :return: log-pseudo joints
        :type return: np.ndarray
        """
        pass

    @abstractmethod
    def log_pseudo_joint(self, model_params, my_suff_stat, my_data):
        """Evaluate log-pseudo-joints.

        :param model_params: Current Thetas
        :type model_params: dict
        :param my_suff_stat: Storage containing current K^{n} and respective log-pseudo joints
        :param my_suff_stat: dict
        :param my_data: Local dataset including indices of reliable (non-missing) entries and
                        entries to be reconstructed
        :type my_data: dict
        :return: log-pseudo joints
        :type return: np.ndarray
        """
        pass

    @tracing.traced
    def free_energy(self, my_data, model_params, my_suff_stat, full=True, compute_lpj=True):
        """Compute the likelihood of the data (my_data) under the generative model using
        the parameters stored in model_params. If ss=="all" all 2^H states will be taken
        into account. Otherwise only the states stored in the array ss will be considered.

        :param my_data: Local dataset including indices of reliable (non-missing) entries and
                        entries to be reconstructed
        :type my_data: dict
        :param model_params: Current Thetas
        :type model_params: dict
        :param my_suff_stat: Storage containing current K^{n} and respective log-pseudo joints
        :type my_suff_stat: dict
        :param full: Whether to evaluate the full likelihood
        :type full: bool
        :param compute_lpj: Whether to (re-) evaluate the free energy (otherwise, it will be
                            evaluted based on previously computed log-pseudo joints)
        :type full: bool
        :return: Free energy value
        :type return: float
        """

        # Array handling
        comm = self.comm
        B_max = self.B_max
        B_max_shft = self.B_max_shft

        my_y = my_data["y"]  # is (my_N, D)
        my_x_infr = my_data["x_infr"]  # is (my_N, D)
        my_N, D = my_y.shape

        incmpl_data = not my_x_infr.all()

        if (
            full
            and not my_suff_stat["permanent"]["allzero"]
            and not my_suff_stat["permanent"]["background"]
        ):
            reset_permanent = True
            my_suff_stat["S_perm"] = 1
            my_suff_stat["permanent"]["allzero"] = True
        else:
            reset_permanent = False

        S_perm = my_suff_stat["S_perm"]
        permanent = my_suff_stat["permanent"]

        N = comm.allreduce(my_N)

        self.E_step_precompute(model_params, my_suff_stat, my_data)
        ljc = model_params["ljc"]

        if full:
            sm = my_suff_stat["sm"]
            assert sm is not None

            if permanent["background"]:
                ss = np.concatenate((sm, np.ones((sm.shape[0], 1), dtype=np.int8)), axis=1)
            else:
                ss = sm[1:, :]

            ss = np.tile(ss[None, :, :], (my_N, 1, 1))

        else:
            ss = my_suff_stat["ss"]
        no_states = ss.shape[1] + S_perm

        ss = ss.astype(bool)

        if "storage" not in my_suff_stat:
            my_suff_stat["storage"] = {"storagekeys": (), "counts_norm": 0, "counts": 0}

        lpj = np.zeros((my_N, no_states))

        # Iterate over datapoints
        for n in range(my_N):

            this_y = my_y[n]
            this_x_infr = my_x_infr[n]
            this_states = ss[n]  # is (S, H)

            my_data["this_y"] = this_y
            my_data["this_x_infr"] = this_x_infr
            my_suff_stat["this_states"] = this_states

            if compute_lpj:
                if not permanent["background"] and S_perm > 0:
                    lpj[n, 0:S_perm] = self.log_pseudo_joint_permanent_states(
                        model_params, my_suff_stat, my_data
                    )
                lpj[n, S_perm:] = self.log_pseudo_joint(model_params, my_suff_stat, my_data)
            else:
                lpj[n] = my_suff_stat["lpj"][n, :]

            if incmpl_data:
                my_suff_stat["storage"] = {
                    "storagekeys": (),
                    "counts_norm": 0,
                    "counts": 0,
                }

        B = np.minimum(B_max - lpj.max(axis=1), B_max_shft)  # is: (my_N,)
        lpj_shft = lpj + B[:, None]
        Fs = (logsumexp(lpj_shft, axis=1) - B).sum()  # is: scalar
        F = ljc + comm.allreduce(Fs) / N

        # neglected_lpj_mask = np.exp(lpj_shft) == 0
        # neglected_lpj_mass = comm.allreduce(lpj_shft[neglected_lpj_mask].sum()) / comm.allreduce(
        #     lpj_shft.sum()
        # )
        # ratio_neglected = float(comm.allreduce((neglected_lpj_mask).sum())) / N / lpj.shape[1]
        # if ratio_neglected or neglected_lpj_mass:
        #     parallel.pprint("On average %.2e%% joints (capturing %.2e%% lpj mass) neglected due\
        #         to precision." % (ratio_neglected, neglected_lpj_mass))

        if reset_permanent:
            my_suff_stat["S_perm"] = 0
            my_suff_stat["permanent"]["allzero"] = False

        return F

    @tracing.traced
    def E_step(self, model_params, my_suff_stat, my_data):
        """E-step: Yield new variational states, evaluate log-pseudo joints for given Thetas, and
           replace better with worse states in K^{n}.

        :param model_params: Current Thetas
        :type model_params: dict
        :param my_suff_stat: Storage containing current K^{n} and respective log-pseudo joints
        :param my_suff_stat: dict
        :param my_data: Local dataset including indices of reliable (non-missing) entries and
                        entries to be reconstructed
        :type my_data: dict
        :return: Free energy for new K^{n} and old Theta, average number of new and unique states
                 obtained through the evolutionary generation of new states, average number of
                 states replaced in the K^{n} sets
        :type return: tuple
        """

        # Array handling
        comm = self.comm
        H = self.H
        S = self.S
        B_max = self.B_max
        B_max_shft = self.B_max_shft

        my_y = my_data["y"]  # is (my_N, D)
        my_x_infr = my_data["x_infr"]  # is (my_N, D)
        my_N, D = my_y.shape
        N = comm.allreduce(my_N)

        lpj = my_suff_stat["lpj"]  # is (my_N, S+S_perm)
        incl = my_suff_stat["incl"]  # is (S_perm, H)
        ss = my_suff_stat["ss"]  # is (my_N, S, H)
        S_perm = my_suff_stat["S_perm"]
        permanent = my_suff_stat["permanent"]
        Mprime = my_suff_stat["Mprime"]

        self.E_step_precompute(model_params, my_suff_stat, my_data)
        ljc = model_params["ljc"]

        my_S_nunique = 0.0
        my_S_sub = 0.0

        tracing.tracepoint("E_step:iterating")
        for n in range(my_N):

            # Array handling
            this_y = my_y[n]
            this_x_infr = my_x_infr[n]
            this_states = ss[n]  # is (S, H)

            my_data["this_y"] = this_y
            my_data["this_x_infr"] = this_x_infr
            my_suff_stat["this_states"] = this_states

            # Compute log-pseudo joints of old states given the current parameters
            if not permanent["background"] and S_perm > 0:
                lpj[n, 0:S_perm] = self.log_pseudo_joint_permanent_states(
                    model_params, my_suff_stat, my_data
                )
            this_lpj = self.log_pseudo_joint(model_params, my_suff_stat, my_data)

            # Sample new states
            my_suff_stat["this_lpj"] = this_lpj

            def eval_lpj(states):
                my_suff_stat["this_states"] = states
                return self.log_pseudo_joint(model_params, my_suff_stat, my_data)

            this_states_new, this_lpj_new = evolve_states(my_suff_stat, model_params, eval_lpj)

            # Compute a variation of Kn which optimizes the log-pseudo joint
            my_S_nunique_add, my_S_sub_add = vary_Kn(
                this_lpj,
                this_lpj_new,
                lpj[n, S_perm:],
                this_states,
                this_states_new,
                H,
                S,
                S_perm,
                incl,
                Mprime,
            )

            my_S_nunique += my_S_nunique_add
            my_S_sub += my_S_sub_add

        S_nunique = comm.allreduce(my_S_nunique) / N
        S_sub = comm.allreduce(my_S_sub) / N

        # Calculate free energy
        B = np.minimum(B_max - lpj.max(axis=1), B_max_shft)  # is: (my_N,)
        lpj_shft = lpj + B[:, None]
        Fs = (logsumexp(lpj_shft, axis=1) - B).sum()  # is: scalar
        F = ljc + comm.allreduce(Fs) / N

        # neglected_lpj_mask = np.exp(lpj_shft) == 0
        # neglected_lpj_mass = comm.allreduce(lpj_shft[neglected_lpj_mask].sum()) / comm.allreduce(
        #     lpj_shft.sum()
        # )
        # ratio_neglected = float(comm.allreduce((neglected_lpj_mask).sum())) / N / lpj.shape[1]
        # if ratio_neglected or neglected_lpj_mass:
        #     parallel.pprint("On average %.2e%% joints (capturing %.2e%% lpj mass) neglected due\
        #         to precision." % (ratio_neglected, neglected_lpj_mass))
        # if "storage" in my_suff_stat.keys():
        #     storage = my_suff_stat["storage"]
        #     no_storage_entries = len(storage['storagekeys'])
        #     if no_storage_entries > 0:
        #         parallel.pprint("Storage has %i entries." % no_storage_entries)
        #         parallel.pprint("storage usage = %.2f" % (comm.allreduce(storage['counts']) / \
        #             comm.allreduce(storage['counts_norm']) ))

        return F, S_nunique, S_sub

    def lpj_reset_check(self, lpj, my_suff_stat):
        """Manually adjust infinite or extremely small log-pseudo joint values

        :param lpj: Current log-pseudo joints
        :type lpj: np.ndarray
        :param my_suff_stat: Storage containing current K^{n} and respective log-pseudo joints
        :type my_suff_stat: dict
        :return: Potentially adjusted log-pseudo joint values
        :return: np.ndarray
        """

        eps_lpj = self.eps_lpj
        B_max = self.B_max

        lpj_is_nan = np.isnan(lpj)
        lpj_s_lpj_eps = lpj < eps_lpj
        lpj_is_inf = np.isinf(lpj)

        if lpj_is_nan.any():
            my_suff_stat["reset_lpj_isnan"] += 1
        elif lpj_s_lpj_eps.any():
            my_suff_stat["reset_lpj_smaller_eps_lpj"] += 1
        elif lpj_is_inf.any():
            my_suff_stat["reset_lpj_isinf"] += 1

        lpj[lpj_is_nan] = eps_lpj
        lpj[lpj_s_lpj_eps] = eps_lpj
        lpj[lpj_is_inf] = B_max

        return lpj

    @abstractmethod
    def modelmean(self, model_params, this_data, this_suff_stat):
        """Evaluate expectation of data given p(data|latents, Thetas), to be used by
        evo.models.Model.reconstruct.

        :param model_params: Current Thetas
        :type model_params: dict
        :param this_data: Storage containing local data point and indices of the entries to be
                          reconstructed
        :type this_data: dict
        :param this_suff_stat: Storage containing current K^{n} and respective log-pseudo joints
                               for given data point n
        :param this_suff_stat: dict
        """
        pass

    def reconstruct(self, my_data, my_suff_stat, model_params):
        """(Re-) estimate corrupted data based on posterior predictive distribution.
        Data reconstructions will be added as y_reconstructed to the my_data dict.

        :param my_data: Local dataset including indices of reliable (non-missing) entries and
                        entries to be reconstructed
        :type my_data: dict
        :param my_suff_stat: Storage containing current K^{n} and respective log-pseudo joints
        :param my_suff_stat: dict
        :param model_params: Current Thetas
        :type model_params: dict
        """

        my_x = my_data["x"]
        my_x_infr = my_data["x_infr"]
        lpj = my_suff_stat["lpj"]
        ss = my_suff_stat["ss"]  # is (my_N x S x H)
        S_perm = my_suff_stat["S_perm"]

        my_N, D = my_x.shape
        B = np.minimum(self.B_max - lpj.max(axis=1), self.B_max_shft)  # is: (my_N,)
        pjc = np.exp(lpj + B[:, None])  # is: (my_N, S+H+1)

        modelmean = self.modelmean

        this_suff_stat = {}
        if "storage" in my_suff_stat.keys():
            this_suff_stat["storage"] = my_suff_stat["storage"]

        my_data["y_reconstructed"] = my_data["y"].copy()
        my_y = my_data["y_reconstructed"]

        for n in range(my_N):

            this_x_infr = my_x_infr[n, :]  # is (D,)
            if np.logical_not(this_x_infr).all():
                continue

            this_y = my_y[n, :]  # is (D,)
            this_x = my_x[n, :]  # is (D,)
            this_pjc = pjc[n, :]  # is (S,)
            this_ss = ss[n, :, :]  # is (S, H)

            this_data = {"y": this_y, "x": this_x, "x_infr": this_x_infr}
            this_suff_stat["ss"] = this_ss
            this_mu = modelmean(model_params, this_data, this_suff_stat)  # is (D_miss, S)

            this_pjc_sum = this_pjc.sum()

            this_estimate = (this_mu * this_pjc[None, S_perm:]).sum(
                axis=1
            ) / this_pjc_sum  # is (D_miss,)
            this_y[np.logical_not(this_x)] = this_estimate
