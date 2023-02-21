# -*- coding: utf-8 -*-
# Copyright (C) 2017 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from __future__ import division
import numpy as np
from mpi4py import MPI
from scipy.special import logsumexp

import evo.utils.parallel as parallel
import evo.utils.tracing as tracing
from evo.models import Model
from evo.variational.eas import evolve_states
from evo.variational.utils import vary_Kn


class SSSC(Model):
    def __init__(
        self,
        D,
        H,
        S,
        use_storage=True,
        precision=np.float64,
        to_learn=["W", "pies", "mus", "sigma2", "Psi"],
        comm=MPI.COMM_WORLD,
    ):
        """Based on https://github.com/ml-uol/prosper/blob/master/prosper/em/camodels/gsc_et.py::
        GSC.__init__

        For LICENSING and COPYRIGHT for the respective function in prosper see prosper's license
        at: https://github.com/ml-uol/prosper/blob/master/LICENSE.txt
        """
        Model.__init__(self, D, H, S, to_learn, comm)

        tiny = np.finfo(np.float64).tiny
        log_tiny = np.finfo(np.float64).min
        eps = np.finfo(np.float64).eps
        eps_ = 5e-5
        tol = 1e-5

        self.eps_lpj = log_tiny
        self.eps_pjc_sum = tiny
        self.eps_W = eps_
        self.eps_pies = eps_
        self.eps_mus = eps
        self.eps_Psi = tol
        self.eps_sigma2 = tol
        self.dtype_precision = precision

        self.noise_policy = {
            "W": (-np.inf, +np.inf, False, None),
            "pies": (tol, 1.0 - tol, False, None),
            "mus": (-np.inf, +np.inf, False, None),
            "Psi": (-np.inf, +np.inf, False, tol),
        }

        self.noise_policy["sigma2"] = (tol, +np.inf, False, None)

        self.s_ids = np.empty(H, dtype=object)
        for h in range(H):
            self.s_ids[h] = 2**h
        self.use_storage = use_storage

    @tracing.traced
    def generate_from_hidden(self, model_params, my_hdata):
        """Based on https://github.com/ml-uol/prosper/blob/master/prosper/em/camodels/gsc_et.py::
        GSC.generate_from_hidden

        For LICENSING and COPYRIGHT for the respective function in prosper see prosper's license
        at: https://github.com/ml-uol/prosper/blob/master/LICENSE.txt
        """

        W = model_params["W"]
        D, H_gen = W.shape

        # Create output arrays, y is data, s and z are ground-truth hiddens
        s = my_hdata["s"]
        my_N, _ = s.shape
        y = np.zeros((my_N, D))
        y_mean = np.zeros((my_N, D))
        z = np.zeros((my_N, H_gen))

        sigma = np.sqrt(model_params["sigma2"]) * np.ones(D)

        for n in range(my_N):
            if np.sum(s[n]) > 0:
                Ws = np.matrix(model_params["W"][:, s[n]])

                z_n = np.random.multivariate_normal(
                    model_params["mus"][s[n]],
                    (model_params["Psi"][s[n], :])[:, s[n]],
                    1,
                ).flatten()

                z[n, s[n]] = z_n

                y_mean[n] = np.array(Ws * z_n[:, None]).flatten()

            y[n] = y_mean[n] + sigma * np.random.randn(D)

        return {"y": y, "s": s, "z": z, "y_mean": y_mean}

    def standard_init(self, my_data, W_init=None, pi_init=None, sigma_init=None):
        """Based on https://github.com/ml-uol/prosper/blob/master/prosper/em/camodels/gsc_et.py::
        GSC.standard_init

        For LICENSING and COPYRIGHT for the respective function in prosper see prosper's license
        at: https://github.com/ml-uol/prosper/blob/master/LICENSE.txt
        """
        comm = self.comm
        H = self.H
        my_y = my_data["y"]
        my_x_infr = my_data["x_infr"]
        my_N, D = my_y.shape
        N = comm.allreduce(my_N)
        incmpl_data = not my_x_infr.all()

        model_params = {}

        # pies
        model_params["pies"] = comm.bcast(
            np.random.uniform(
                low=0.1,
                high=0.5,
                size=[
                    H,
                ],
            )
        )

        # mus
        model_params["mus"] = (
            comm.bcast(np.random.normal(0, 1, [self.H]))
            if "mus" in self.to_learn
            else comm.bcast(np.ones(H))
        )

        # Psi
        Psi_diag = comm.bcast(np.ones(H))
        model_params["Psi"] = np.diag(Psi_diag)

        # Calculate averarge y
        if not incmpl_data:
            y_mean = parallel.allmean(my_y, axis=0, comm=comm)  # shape: (D, )
        else:
            y_mean = np.zeros(D)  # shape: (D, )
            for n in range(my_N):
                this_y = my_y[n]
                this_x_infr = my_x_infr[n]
                y_mean[this_x_infr] += this_y[this_x_infr]
            y_mean = comm.allreduce(y_mean) / N

        # sigma2
        if sigma_init is None:
            if not incmpl_data:
                tmp = parallel.allmean((my_y - y_mean) ** 2, axis=0, comm=comm)  # shape: (D, )
                my_cov = np.cov(my_y.T)
            else:
                tmp = np.zeros(D)
                my_cov = np.zeros((D, D))
                for n in range(my_N):
                    this_y = my_y[n]
                    this_x_infr = my_x_infr[n]
                    this_y_zm = this_y[this_x_infr] - y_mean[this_x_infr]
                    tmp[this_x_infr] += this_y_zm**2
                    my_cov[np.outer(this_x_infr, this_x_infr)] = np.outer(
                        this_y_zm, this_y_zm
                    ).flatten()
                my_cov = comm.allreduce(my_cov) / N

            model_params["sigma2"] = (
                comm.allreduce(tmp.sum() / my_x_infr.flatten().sum()) + 0.001
                if incmpl_data
                else np.mean(np.diag(my_cov)) + 0.001
            )

        else:
            model_params["sigma2"] = sigma_init

        # W
        if type(W_init) is not np.ndarray:
            if W_init == "random_uniform":
                model_params["W"] = comm.bcast(np.random.random((D, H)))
            if W_init == "normal":
                model_params["W"] = comm.bcast(np.random.normal(0, 5, [D, H]))
            else:
                noise = np.sqrt(model_params["sigma2"]) / 4.0
                model_params["W"] = y_mean[:, None] + np.random.normal(
                    scale=noise, size=[D, H]
                )  # shape: (H, D)
        else:
            model_params["W"] = W_init

        model_params = comm.bcast(model_params)

        return model_params

    def check_params(self, model_params):
        """Based on https://github.com/ml-uol/prosper/blob/master/prosper/em/camodels/gsc_et.py::
        GSC.check_params

        For LICENSING and COPYRIGHT for the respective function in prosper see prosper's license
        at: https://github.com/ml-uol/prosper/blob/master/LICENSE.txt
        """
        comm = self.comm

        model_params = Model.check_params(self, model_params)

        if comm.rank == 0:
            assert np.isfinite(model_params["W"]).all()

            assert np.isfinite(model_params["mus"]).all()

            assert np.isfinite(model_params["pies"]).all()

            assert np.isfinite(model_params["Psi"]).all()

            assert np.isfinite(model_params["sigma2"])
            assert model_params["sigma2"] > 0

        return model_params

    @tracing.traced
    def log_pseudo_joint_permanent_states(self, model_params, my_suff_stat, my_data):
        this_y = my_data["this_y"]
        this_x_infr = my_data["this_x_infr"]
        sigma2_inv = model_params["sigma2_inv"]

        permanent = my_suff_stat["permanent"]
        S_perm = my_suff_stat["S_perm"]

        lpj = np.empty((S_perm,))

        # all-zero state
        if permanent["allzero"]:
            lpj[0] = -0.5 * (this_y[this_x_infr] ** 2).sum() * sigma2_inv

        return self.lpj_reset_check(lpj, my_suff_stat)

    @tracing.traced
    def log_pseudo_joint(self, model_params, my_suff_stat, my_data):
        s_ids = self.s_ids

        this_y = my_data["this_y"]
        this_x_infr = my_data["this_x_infr"]

        mus = model_params["mus"]
        W = model_params["W"]
        Psi = model_params["Psi"]
        sigma2_inv = model_params["sigma2_inv"]
        pil_bar = model_params["pil_bar"]

        states = my_suff_stat["this_states"]
        storage = my_suff_stat["storage"]

        S = states.shape[0]
        lpjpt = np.zeros(S)
        pre_lpjpt = np.zeros(S)

        for s in range(S):
            state = states[s]  # is (H,)
            this_s_id = (s_ids * state).sum()

            mus_s = mus[state]  # is (|state|,)
            pre_lpjpt[s] = pil_bar[state].sum()

            storage["counts_norm"] += 1

            if this_s_id in storage["storagekeys"]:
                W_s_mus_s = storage[np.str(this_s_id)]["W_s_mus_s"]
                C_det = storage[np.str(this_s_id)]["C_det"]
                C_inv = storage[np.str(this_s_id)]["C_inv"]
                storage["counts"] += 1
            else:
                W_s = W[this_x_infr, :][:, state]  # is (D_obs, |state|)
                Psi_s = Psi[state, :][:, state]  # is (|state|, |state|)
                try:
                    Psi_s_inv = np.linalg.inv(Psi_s)
                except np.linalg.linalg.LinAlgError:
                    Psi_s_inv = np.linalg.pinv(Psi_s)
                    my_suff_stat["Psi_s_pinv"] += 1
                    # print("log_pseudo_joint: Taking pinv for Psi_s_inv")
                log_det_Psi_s = np.linalg.slogdet(Psi_s)[1]  # is scalar
                W_s_mus_s = np.dot(W_s, mus_s)  # is (D_obs,)

                sigma2_inv_W_s = sigma2_inv * W_s

                M_s = (
                    np.dot(W_s.T, sigma2_inv_W_s) + Psi_s_inv
                )  # cf. eq. 19 in Sheikh et al., 2014, JMLR; eq. 2.19 in Sheikh, Dissertation,
                # 2017
                log_det_M_s = np.linalg.slogdet(M_s)[1]

                try:
                    lambda_s = np.linalg.inv(
                        M_s
                    )  # cf. q 19 Sheikh et al., 2014, JMLR, is # is (|state|, |state|)
                except np.linalg.linalg.LinAlgError:
                    lambda_s = np.linalg.pinv(M_s)
                    # print("log_pseudo_joint: Taking pinv for lambda_s")

                lambda_s_W_s_sigma2_inv = np.dot(lambda_s, W_s.T) * sigma2_inv

                C_det = log_det_M_s + log_det_Psi_s

                C_inv = -np.dot(sigma2_inv_W_s, lambda_s_W_s_sigma2_inv) + sigma2_inv * np.eye(
                    this_x_infr.sum()
                )

                storage["storagekeys"] += (this_s_id,)
                storage[np.str(this_s_id)] = {
                    "W_s_mus_s": W_s_mus_s,
                    "C_det": C_det,
                    "C_inv": C_inv,
                    "lambda_s": lambda_s,
                    "lambda_s_W_s_sigma2_inv": lambda_s_W_s_sigma2_inv,
                }

            this_y_norm = this_y[this_x_infr] - W_s_mus_s

            lpjpt[s] = -0.5 * (C_det + (this_y_norm * np.dot(C_inv, this_y_norm)).sum())

        lpj = lpjpt + pre_lpjpt

        return self.lpj_reset_check(lpj, my_suff_stat)

    @tracing.traced
    def E_step_precompute(self, model_params, my_suff_stat, my_data):
        comm = self.comm
        D = self.D
        dtype_precision = self.dtype_precision
        pies = model_params["pies"]
        sigma2 = model_params["sigma2"]
        my_x_infr = my_data["x_infr"]
        my_N = my_x_infr.shape[0]
        N = comm.allreduce(my_N)
        incmpl_data = not my_x_infr.all()

        model_params["ljc"] = np.log(1.0 - pies).sum() - D / 2 * np.log(2 * np.pi)  # is scalar
        model_params["piH"] = pies.sum()
        model_params["pil_bar"] = np.log(pies / (1.0 - pies))

        sigma2_float128 = sigma2.astype("longdouble")
        try:
            model_params["sigma2_inv"] = (1.0 / sigma2_float128).astype(
                dtype_precision
            )  # is scalar
            log_det_sigma2 = D * np.log(sigma2_float128).astype(dtype_precision)
        except Exception:
            model_params["sigma2_inv"] = 1.0 / sigma2  # is scalar
            log_det_sigma2 = D * np.log(sigma2)
        model_params["ljc"] -= 0.5 * log_det_sigma2

        # Check missing-data case
        if incmpl_data:
            sum_n_d = comm.allreduce(my_x_infr.sum())
            model_params["ljc"] = (
                np.log(1.0 - pies).sum() + (-np.log(2 * np.pi) - np.log(sigma2)) * sum_n_d / N / 2
            )

        my_suff_stat["storage"] = {"storagekeys": (), "counts_norm": 0, "counts": 0}
        my_suff_stat["reset_lpj_isnan"] = 0
        my_suff_stat["reset_lpj_smaller_eps_lpj"] = 0
        my_suff_stat["reset_lpj_isinf"] = 0
        my_suff_stat["Psi_s_pinv"] = 0

    @tracing.traced
    def modelmean(self, model_params, this_data, this_suff_stat):
        H = self.H
        S = self.S
        s_ids = self.s_ids
        dtype_precision = self.dtype_precision

        W = model_params["W"]
        mus = model_params["mus"]

        this_y = this_data["y"]
        this_x_infr = this_data["x_infr"]
        this_x = this_data["x"]

        this_ss = this_suff_stat["ss"]  # is (S,H)
        storage = this_suff_stat["storage"]

        this_W = W[np.logical_not(this_x), :]

        this_sz = np.zeros((H, S), dtype=dtype_precision)

        for s in range(S):
            this_ss_s = this_ss[s]  # is (H,)
            this_s_id = (s_ids * this_ss_s).sum()

            mus_s = mus[this_ss_s]  # is (|this_ss_s|,)

            W_s_mus_s = storage[np.str(this_s_id)]["W_s_mus_s"]
            lambda_s_W_s_sigma2_inv = storage[np.str(this_s_id)]["lambda_s_W_s_sigma2_inv"]

            this_y_norm = this_y[this_x_infr] - W_s_mus_s  # is (D,)

            kappa_s = np.dot(lambda_s_W_s_sigma2_inv, this_y_norm)  # is (|this_ss_s|,)
            kappa_s += mus_s

            this_sz[this_ss_s, s] = kappa_s

        return np.dot(this_W, this_sz)

    @tracing.traced
    def step(self, model_params, my_suff_stat, my_data, do_reconstruction=False):
        # Sanity check model parameters
        model_params = self.check_params(model_params)

        # Do E-step and vary state space and calculate joint-probabilities
        F, S_nunique, S_sub, new_model_params = self.EM_step(
            model_params, my_suff_stat, my_data, do_reconstruction
        )

        return F, S_nunique, S_sub, new_model_params

    @tracing.traced
    def EM_step(self, model_params, my_suff_stat, my_data, do_reconstruction=False):
        """E- and M-step: 1) Yield new variational states, evaluate log-pseudo joints for given
        Thetas, and replace better with worse states in K^{n}. 2) Potentially re-estimate corrupted
        entries in the data. 3) Update Thetas using given K^{n} and respective log-pseudo joints.

        :param model_params: Current Thetas
        :type model_params: dict
        :param my_suff_stat: Storage containing current K^{n} and respective log-pseudo joints
        :param my_suff_stat: dict
        :param my_data: Local dataset including indices of reliable (non-missing) entries and
                        entries to be reconstructed
        :type my_data: nd.ndarray
        :param do_reconstruction: Whether to evaluate data estimator (data reconstructions will
                                  be added as y_reconstructed to the my_data dict)
        :type do_reconstruction: bool
        :return: Free energy for new K^{n} (and Theta of previous iteration), average number of
                 new and unique states obtained through the evolutionary generation of new
                 states, average number of states replaces in the K^{n} sets, updated Thetas^{new}
        :type return: tuple
        """

        # Array handling
        comm = self.comm
        H = self.H
        S = self.S
        B_max = self.B_max
        B_max_shft = self.B_max_shft
        dtype_precision = self.dtype_precision
        eps_pjc_sum = self.eps_pjc_sum
        eps_W = self.eps_W
        eps_pies = self.eps_pies
        eps_mus = self.eps_mus
        eps_Psi = self.eps_Psi
        eps_sigma2 = self.eps_sigma2
        s_ids = self.s_ids

        my_y = my_data["y"]  # is (my_N, D)
        my_x_infr = my_data["x_infr"]  # is (my_N, D)
        my_N, D = my_y.shape
        N = comm.allreduce(my_N)

        lpj = my_suff_stat["lpj"]  # is (my_N, S+1)
        incl = my_suff_stat["incl"]  # is (1, H)
        ss = my_suff_stat["ss"]  # is (my_N, S+1, H)
        S_perm = my_suff_stat["S_perm"]
        permanent = my_suff_stat["permanent"]
        Mprime = my_suff_stat["Mprime"]

        incmpl_data = not my_x_infr.all()
        use_storage = self.use_storage if not incmpl_data else False

        self.E_step_precompute(model_params, my_suff_stat, my_data)
        ljc = model_params["ljc"]
        W = model_params["W"]
        sigma2 = model_params["sigma2"]
        mus = model_params["mus"]
        Psi = model_params["Psi"]
        storage = my_suff_stat["storage"]

        my_S_nunique = 0.0
        my_S_sub = 0.0

        Theta_new = model_params

        my_sum_xpt_s = np.zeros([H], dtype=dtype_precision)
        my_sum_xpt_ss = np.zeros([H, H], dtype=dtype_precision)
        my_sum_xpt_sz = np.zeros([H], dtype=dtype_precision)
        my_sum_xpt_szsz = np.zeros([H, H], dtype=dtype_precision)

        if "W" in self.to_learn:
            my_Wp = np.zeros((D, H))

        if "Psi" in self.to_learn:
            my_sum_xpt_s_sz_outer = np.zeros([H, H], dtype=dtype_precision)
        if "sigma2" in self.to_learn:
            if incmpl_data:
                my_sum_W_xpt_sz_sz_W = np.zeros([D, D], dtype=dtype_precision)
            else:
                my_sum_xpt_sz_sz_outer = np.zeros([H, H], dtype=dtype_precision)

        if do_reconstruction:
            modelmean = self.modelmean

            this_suff_stat = {}
            if "storage" in my_suff_stat.keys():
                this_suff_stat["storage"] = my_suff_stat["storage"]

            my_data["y_reconstructed"] = my_data["y"].copy()

        tracing.tracepoint("EM_step:iterating")
        for n in range(my_N):
            # Array handling
            this_y = my_y[n]
            this_x_infr = my_x_infr[n]
            this_states = ss[n]  # is (S, H)

            my_data["this_y"] = this_y
            my_data["this_x_infr"] = this_x_infr
            my_suff_stat["this_states"] = this_states

            # Log-pseudo joints of current states given the current parameters
            if not permanent["background"]:
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

            # Compute a variation of Kn which optimizes the lopg-pseudo joint
            this_S_nunique, this_S_sub = vary_Kn(
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
            my_S_nunique += this_S_nunique
            my_S_sub += this_S_sub

            # Compute sufficient statistics
            this_B = np.minimum(B_max - lpj[n].max(), B_max_shft)  # is scalar
            this_pjc = np.exp(lpj[n] + this_B)  # is (S+H+1)

            this_xpt_s = np.zeros(H, dtype=self.dtype_precision)
            this_xpt_ss = np.zeros((H, H), dtype=self.dtype_precision)
            this_xpt_sz = np.zeros(H, dtype=self.dtype_precision)
            this_xpt_szsz = np.zeros((H, H), dtype=self.dtype_precision)

            for s in range(S):
                this_state = this_states[s]  # is (H,)
                this_pjc_s = this_pjc[s + S_perm]  # is scalar
                this_s_id = (s_ids * this_state).sum()

                mus_s = mus[this_state]  # is (|this_state|,)

                W_s_mus_s = storage[np.str(this_s_id)]["W_s_mus_s"]
                lambda_s = storage[np.str(this_s_id)]["lambda_s"]
                lambda_s_W_s_sigma2_inv = storage[np.str(this_s_id)]["lambda_s_W_s_sigma2_inv"]

                this_y_norm = this_y[this_x_infr] - W_s_mus_s  # is (D,)

                kappa_s = np.dot(lambda_s_W_s_sigma2_inv, this_y_norm)  # is (|this_state|,)
                kappa_s += mus_s
                lambda_s_kappa_s_kappa_s = lambda_s + np.outer(
                    kappa_s, kappa_s
                )  # is (|this_state|, |this_state|)

                kappa_s_tmp = np.array(kappa_s, dtype=dtype_precision)
                kappa_s_tmp *= (
                    this_pjc_s  # don't need to broadcast manually since this_pjc_s is scalar
                )
                this_xpt_sz[this_state] += kappa_s_tmp

                lambda_s_kappa_s_kappa_s_tmp = np.array(
                    lambda_s_kappa_s_kappa_s, dtype=dtype_precision
                )
                lambda_s_kappa_s_kappa_s_tmp *= this_pjc_s  # don't need to broadcast manually
                # since this_pjc_s is scalar
                this_xpt_szsz_tmp = np.zeros((H, H), dtype=dtype_precision)
                this_xpt_szsz_tmp[
                    np.outer(this_state, this_state)
                ] = lambda_s_kappa_s_kappa_s_tmp.flatten()
                this_xpt_szsz += this_xpt_szsz_tmp

            this_xpt_s += (this_pjc[S_perm:][:, None] * this_states).sum(
                axis=0
            )  # is (H,), bool casted to float
            this_xpt_ss += np.dot(this_pjc[S_perm:].T * this_states.T, this_states)  # is (H, H)

            this_pjc_sum = this_pjc.sum() + eps_pjc_sum
            this_xpt_s /= this_pjc_sum
            this_xpt_ss /= this_pjc_sum
            this_xpt_sz /= this_pjc_sum
            this_xpt_szsz /= this_pjc_sum

            my_sum_xpt_s += this_xpt_s
            my_sum_xpt_ss += this_xpt_ss
            my_sum_xpt_sz += this_xpt_sz
            my_sum_xpt_szsz += this_xpt_szsz

            # Reconstruction
            if do_reconstruction:
                this_y_rec = my_data["y_reconstructed"][n, :]
                this_x = my_data["x"][n, :]

                this_data = {"y": this_y_rec, "x": this_x, "x_infr": this_x_infr}
                this_suff_stat["ss"] = this_states
                this_mus = modelmean(model_params, this_data, this_suff_stat)  # is (D_miss, S)

                this_pjc_sum = this_pjc.sum()

                this_estimate = (this_mus * this_pjc[None, S_perm:]).sum(
                    axis=1
                ) / this_pjc_sum  # is (D_miss,)
                this_y_rec[np.logical_not(this_x)] = this_estimate

            # Contribution to updates of model parameters per data point
            if "W" in self.to_learn:
                if incmpl_data:
                    my_Wp += this_xpt_sz[None, :] * this_y_rec[:, None]
                else:
                    my_Wp += this_xpt_sz[None, :] * this_y[:, None]

            if "Psi" in self.to_learn:
                my_sum_xpt_s_sz_outer += np.outer(this_xpt_s, this_xpt_sz)

            if "sigma2" in self.to_learn:
                if incmpl_data:
                    W_this_xpt_sz = np.dot(W[this_x_infr, :], this_xpt_sz)
                    my_sum_W_xpt_sz_sz_W[np.outer(this_x_infr, this_x_infr)] += np.outer(
                        W_this_xpt_sz, W_this_xpt_sz
                    ).flatten()
                else:
                    my_sum_xpt_sz_sz_outer += np.outer(this_xpt_sz, this_xpt_sz)

            if not use_storage:
                my_suff_stat["storage"] = {
                    "storagekeys": (),
                    "counts_norm": 0,
                    "counts": 0,
                }
                storage = my_suff_stat["storage"]
            if do_reconstruction:
                this_suff_stat["storage"] = storage

        sum_xpt_s = np.empty([H], dtype=dtype_precision)
        sum_xpt_ss = np.empty([H, H], dtype=dtype_precision)
        sum_xpt_sz = np.empty([H], dtype=dtype_precision)
        sum_xpt_szsz = np.empty([H, H], dtype=dtype_precision)

        if "Psi" in self.to_learn:
            sum_xpt_s_sz_outer = np.empty([H, H], dtype=dtype_precision)
        if "sigma2" in self.to_learn:
            if incmpl_data:
                sum_W_xpt_sz_sz_W = np.empty([D, D], dtype=dtype_precision)
            else:
                sum_xpt_sz_sz_outer = np.empty([H, H], dtype=dtype_precision)

        comm.Allreduce([my_sum_xpt_s, MPI.DOUBLE], [sum_xpt_s, MPI.DOUBLE])
        comm.Allreduce([my_sum_xpt_ss, MPI.DOUBLE], [sum_xpt_ss, MPI.DOUBLE])
        comm.Allreduce([my_sum_xpt_sz, MPI.DOUBLE], [sum_xpt_sz, MPI.DOUBLE])
        comm.Allreduce([my_sum_xpt_szsz, MPI.DOUBLE], [sum_xpt_szsz, MPI.DOUBLE])

        if "Psi" in self.to_learn:
            comm.Allreduce([my_sum_xpt_s_sz_outer, MPI.DOUBLE], [sum_xpt_s_sz_outer, MPI.DOUBLE])
        if "sigma2" in self.to_learn:
            if incmpl_data:
                comm.Allreduce([my_sum_W_xpt_sz_sz_W, MPI.DOUBLE], [sum_W_xpt_sz_sz_W, MPI.DOUBLE])
            else:
                comm.Allreduce(
                    [my_sum_xpt_sz_sz_outer, MPI.DOUBLE],
                    [sum_xpt_sz_sz_outer, MPI.DOUBLE],
                )

        # Calculate updated W
        if "W" in self.to_learn:
            tracing.tracepoint("M_step:update W")
            Wp = np.empty_like(my_Wp)
            comm.Allreduce([my_Wp, MPI.DOUBLE], [Wp, MPI.DOUBLE])
            try:
                sum_xpt_szsz_inv = np.linalg.inv(sum_xpt_szsz)
                W_new = np.dot(Wp, sum_xpt_szsz_inv)
            except np.linalg.linalg.LinAlgError:
                # if Sum of the expected values of the second moments was not invertable.
                # Adding some noise and taking pinv.
                try:
                    noise = np.random.normal(0, eps_W, H)
                    noise = np.outer(noise, noise)
                    sum_xpt_szsz_inv = np.linalg.pinv(sum_xpt_szsz + noise)
                    W_new = np.dot(Wp, sum_xpt_szsz_inv)
                    parallel.pprint("Use pinv and additional noise for W update.")
                except np.linalg.linalg.LinAlgError:
                    # Sum of the expected values of the second moments was not invertable.
                    # Skip the update of parameter W but add some noise to it.
                    W_new = W + (eps_W * np.random.normal(0, 1, [D, H]))
                    parallel.pprint("Skipped W update. Added some noise to it.")
            Theta_new["W"] = W_new

        # Calculate updated pi
        if "pies" in self.to_learn:
            tracing.tracepoint("M_step:update pies")
            pies_new = sum_xpt_s / N
            pies_new[pies_new <= eps_pies] = eps_pies
            pies_new[pies_new >= (1 - eps_pies)] = 1 - eps_pies

            if permanent["background"]:
                pies_new[H - 1] = 1.0 - 1.1e-5

            Theta_new["pies"] = pies_new

        # Calculate updated mus
        if "mus" in self.to_learn:
            tracing.tracepoint("M_step:update mus")
            mus_new = sum_xpt_sz * 1.0 / (sum_xpt_s + eps_mus)
            Theta_new["mus"] = mus_new

        # Calculate updated Psi
        if "Psi" in self.to_learn:
            tracing.tracepoint("M_step:update Psi")
            Psi = np.zeros((H, H))
            mus_outer = np.outer(Theta_new["mus"], Theta_new["mus"])
            Psi += mus_outer * sum_xpt_ss
            Psi += sum_xpt_szsz
            Psi -= 2 * Theta_new["mus"][:, None] * sum_xpt_s_sz_outer

            Psi_new = Psi * np.linalg.inv(sum_xpt_ss + eps_Psi * np.eye(self.H))
            +(eps_Psi * np.eye(self.H))

            Theta_new["Psi"] = Psi_new

        # Calculate updated sigma
        if "sigma2" in self.to_learn:
            tracing.tracepoint("M_step:update sigma sq")

            sigma2_new = 0.0

            if incmpl_data:
                my_y_inner = (my_y[my_x_infr] ** 2).sum()

                sigma2_new += comm.allreduce(my_y_inner) - np.trace(sum_W_xpt_sz_sz_W)

                correction = comm.allreduce(my_x_infr.sum()) * sigma2

                sigma2_new = ((sigma2_new + correction) / N / D) + eps_sigma2

            else:
                WT_outer = np.dot(Theta_new["W"].T, Theta_new["W"])

                my_y_outer_diag = (my_y**2).sum(axis=0)
                y_outer_diag = np.empty((D,))
                comm.Allreduce([my_y_outer_diag, MPI.DOUBLE], [y_outer_diag, MPI.DOUBLE])

                sigma2_new += y_outer_diag.sum()
                sigma2_new -= np.trace(np.dot(sum_xpt_sz_sz_outer, WT_outer))

                sigma2_new = (sigma2_new / N / D) + eps_sigma2

            Theta_new["sigma2"] = sigma2_new

        # Sampler evaluation
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
        # parallel.pprint("storage usage = %.2f" % (comm.allreduce(storage['counts']) \
        #     / comm.allreduce(storage['counts_norm']) ))

        # Check manual adjustments
        no_reset_lpj_isnan = comm.allreduce(my_suff_stat["reset_lpj_isnan"])
        no_reset_lpj_smaller_eps_lpj = comm.allreduce(my_suff_stat["reset_lpj_smaller_eps_lpj"])
        no_reset_lpj_isinf = comm.allreduce(my_suff_stat["reset_lpj_isinf"])
        no_Psi_s_pinv = comm.allreduce(my_suff_stat["Psi_s_pinv"])
        if no_reset_lpj_isnan > 0:
            parallel.pprint("no reset_lpj_isnan = %i" % no_reset_lpj_isnan)
        if no_reset_lpj_smaller_eps_lpj > 0:
            parallel.pprint("no reset_lpj_smaller_eps_lpj = %i" % no_reset_lpj_smaller_eps_lpj)
        if no_reset_lpj_isinf > 0:
            parallel.pprint("no reset_lpj_isinf = %i" % no_reset_lpj_isinf)
        if no_Psi_s_pinv > 0:
            parallel.pprint("no Psi_s_pinv = %i" % no_Psi_s_pinv)

        return F, S_nunique, S_sub, Theta_new
