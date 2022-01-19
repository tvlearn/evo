# -*- coding: utf-8 -*-
# Copyright (C) 2017 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from __future__ import division
import numpy as np
from mpi4py import MPI

import evo.utils.parallel as parallel
import evo.utils.tracing as tracing
from evo.models import Model


class BSC(Model):
    def __init__(self, D, H, S, to_learn=["W", "pi", "sigma"], comm=MPI.COMM_WORLD):
        """Based on https://github.com/ml-uol/prosper/blob/master/prosper/em/camodels/bsc_et.py::
        BSC_ET.__init__

        For LICENSING and COPYRIGHT for the respective function in prosper see prosper's license
        at: https://github.com/ml-uol/prosper/blob/master/LICENSE.txt
        """
        Model.__init__(self, D, H, S, to_learn, comm)
        log_tiny = np.finfo(np.float64).min
        self.eps_lpj = log_tiny

    @tracing.traced
    def generate_from_hidden(self, model_params, my_hdata):
        """Based on https://github.com/ml-uol/prosper/blob/master/prosper/em/camodels/bsc_et.py::
        BSC_ET.generate_from_hidden

        For LICENSING and COPYRIGHT for the respective function in prosper see prosper's license
        at: https://github.com/ml-uol/prosper/blob/master/LICENSE.txt
        """

        W = model_params["W"].T
        sigma = model_params["sigma"]
        H_gen, D = W.shape

        s = my_hdata["s"]
        my_N, _ = s.shape

        # Create output arrays, y is data
        y = np.zeros((my_N, D))

        for n in range(my_N):
            # Linear superposition
            for h in range(H_gen):
                if s[n, h]:
                    y[n] += W[h]

        y_mean = y.copy()

        # Add noise according to the model parameters
        y += np.random.normal(scale=sigma, size=(my_N, D))

        # Build return structure
        return {"y": y, "s": s, "y_mean": y_mean}

    @tracing.traced
    def log_pseudo_joint_permanent_states(self, model_params, my_suff_stat, my_data):
        this_y = my_data["this_y"]
        this_x_infr = my_data["this_x_infr"]
        pre1 = model_params["pre1"]

        permanent = my_suff_stat["permanent"]
        S_perm = my_suff_stat["S_perm"]

        lpj = np.empty((S_perm,))

        # all-zero state
        if permanent["allzero"]:
            lpj[0] = pre1 * (this_y[this_x_infr] ** 2).sum()

        lpj = self.lpj_reset_check(lpj, my_suff_stat)

        return lpj

    @tracing.traced
    def log_pseudo_joint(self, model_params, my_suff_stat, my_data):
        this_y = my_data["this_y"]
        this_x_infr = my_data["this_x_infr"]

        W = model_params["W"].T
        pre1 = model_params["pre1"]
        pil_bar = model_params["pil_bar"]

        states = my_suff_stat["this_states"]

        state_abs = states.sum(axis=1)  # is (curr_S,)
        pre_lpjpt = pil_bar * state_abs
        Wbar = np.dot(states, W[:, this_x_infr])

        lpjpt = pre1 * ((Wbar - this_y[this_x_infr]) ** 2).sum(axis=1)

        lpj = lpjpt + pre_lpjpt

        return self.lpj_reset_check(lpj, my_suff_stat)

    @tracing.traced
    def E_step_precompute(self, model_params, my_suff_stat, my_data):
        comm = self.comm
        D = self.D
        H = self.H
        pi = model_params["pi"]
        sigma = model_params["sigma"]
        my_x_infr = my_data["x_infr"]
        my_N = my_x_infr.shape[0]
        N = comm.allreduce(my_N)
        incmpl_data = not my_x_infr.all()

        model_params["piH"] = pi * H
        model_params["pre1"] = -1.0 / 2.0 / sigma / sigma
        model_params["pil_bar"] = np.log(pi / (1.0 - pi))

        if incmpl_data:
            sum_n_d = comm.allreduce(my_x_infr.sum())
            model_params["ljc"] = (
                H * np.log(1.0 - pi) - np.log(2 * np.pi * sigma * sigma) * sum_n_d / N / 2
            )
        else:
            model_params["ljc"] = H * np.log(1.0 - pi) - D / 2 * np.log(2 * np.pi * sigma * sigma)

        my_suff_stat["reset_lpj_isnan"] = 0
        my_suff_stat["reset_lpj_smaller_eps_lpj"] = 0
        my_suff_stat["reset_lpj_isinf"] = 0

    @tracing.traced
    def M_step(self, model_params, my_suff_stat, my_data):
        """M-step: Update Thetas using given K^{n} and respective log-pseudo joints.

        :param model_params: Current Thetas
        :type model_params: dict
        :param my_suff_stat: Storage containing current K^{n} and respective log-pseudo joints
        :param my_suff_stat: dict
        :param my_data: Local dataset including indices of reliable (non-missing) entries and
                        entries to be reconstructed
        :type my_data: np.ndarray
        :return: Updated Thetas^{new}
        :type return: dict

        Inspired by
        https://github.com/ml-uol/prosper/blob/master/prosper/em/camodels/bsc_et.py::BSC_ET.M_step

        For LICENSING and COPYRIGHT for the respective function in prosper see prosper's license
        at: https://github.com/ml-uol/prosper/blob/master/LICENSE.txt
        """

        # Array handling
        comm = self.comm
        my_x_infr = my_data["x_infr"]
        my_N, D = my_x_infr.shape
        N = comm.allreduce(my_N)
        H = self.H
        W = model_params["W"].T
        pi = model_params["pi"]
        sigma = model_params["sigma"]
        lpj = my_suff_stat["lpj"]  # is (my_N x (S+H+1))
        ss = my_suff_stat["ss"]  # is (my_N x S x H)
        S_perm = my_suff_stat["S_perm"]
        permanent = my_suff_stat["permanent"]
        incmpl_data = not my_x_infr.all()

        # Check if lpj have been manually adjusted
        no_reset_lpj_isnan = comm.allreduce(my_suff_stat["reset_lpj_isnan"])
        no_reset_lpj_smaller_eps_lpj = comm.allreduce(my_suff_stat["reset_lpj_smaller_eps_lpj"])
        no_reset_lpj_isinf = comm.allreduce(my_suff_stat["reset_lpj_isinf"])
        if no_reset_lpj_isnan > 0:
            parallel.pprint("no reset_lpj_isnan = %i" % no_reset_lpj_isnan)
        if no_reset_lpj_smaller_eps_lpj > 0:
            parallel.pprint("no reset_lpj_smaller_eps_lpj = %i" % no_reset_lpj_smaller_eps_lpj)
        if no_reset_lpj_isinf > 0:
            parallel.pprint("no reset_lpj_isinf = %i" % no_reset_lpj_isinf)

        # Some data handling
        B = np.minimum(self.B_max - lpj.max(axis=1), self.B_max_shft)  # is: (my_N,)
        pjc = np.exp(lpj + B[:, None])  # is: (my_N, S+H+1)

        my_Wp = np.zeros_like(W)  # is (H, D)
        my_Wq = np.zeros((H, H))  # is (H, H)
        my_pies = np.zeros((H))  # is (H, D)
        my_sigma = 0.0

        # Check missing-data case
        if incmpl_data:
            assert "y_reconstructed" in my_data.keys()
            my_y = my_data["y_reconstructed"]
        else:
            my_y = my_data["y"]

        # Iterate over all datapoints
        tracing.tracepoint("M_step:iterating")
        for n in range(my_N):
            this_y = my_y[n, :]  # is (D,)
            this_x_infr = my_x_infr[n, :]

            this_pjc = pjc[n, :]  # is (S,)
            this_ss = ss[n, :, :]  # is (S, H)

            this_Wp = np.zeros_like(my_Wp)  # numerator for current datapoint   (H, D)
            this_Wq = np.zeros_like(my_Wq)  # denominator for current datapoint (H, H)
            this_pies = np.zeros((H))
            this_sigma = 0.0

            # Zero active hidden causes
            if permanent["allzero"]:
                this_sigma += this_pjc[0] * (this_y[this_x_infr] ** 2).sum()

            # Handle hidden states with more than 1 active cause
            this_pies += (this_pjc[S_perm:].T * this_ss.T).sum(axis=1)
            this_Wp += np.outer((this_pjc[S_perm:].T * this_ss.T).sum(axis=1), this_y)
            this_Wq += np.dot(this_pjc[S_perm:].T * this_ss.T, this_ss)
            # this_pi += np.inner(this_pjc[S_perm:], this_ss.sum(axis=1))
            this_sigma += (
                this_pjc[S_perm:]
                * ((this_y[this_x_infr] - np.dot(this_ss, W[:, this_x_infr])) ** 2).sum(axis=1)
            ).sum()

            this_pjc_sum = this_pjc.sum()
            my_pies += this_pies / this_pjc_sum
            my_Wp += this_Wp / this_pjc_sum
            my_Wq += this_Wq / this_pjc_sum
            my_sigma += this_sigma / this_pjc_sum

        # Calculate updated W
        if "W" in self.to_learn:
            tracing.tracepoint("M_step:update W")
            Wp = np.empty_like(my_Wp)
            Wq = np.empty_like(my_Wq)
            comm.Allreduce([my_Wp, MPI.DOUBLE], [Wp, MPI.DOUBLE])
            comm.Allreduce([my_Wq, MPI.DOUBLE], [Wq, MPI.DOUBLE])
            if float(np.__version__[2:]) >= 14.0:
                rcond = None
            else:
                rcond = -1
            try:
                W_new = np.linalg.lstsq(Wq, Wp, rcond=rcond)[0]
            except np.linalg.linalg.LinAlgError:
                eps_W = 5e-5
                try:
                    noise = np.random.normal(0, eps_W, H)
                    noise = np.outer(noise, noise)
                    Wq_inv = np.linalg.pinv(Wq + noise)
                    W_new = np.dot(Wq_inv, Wp)
                    parallel.pprint("Use pinv and additional noise for W update.")
                except np.linalg.linalg.LinAlgError:
                    # Sum of the expected values of the second moments was not invertable.
                    # Skip the update of parameter W but add some noise to it.
                    W_new = W + (eps_W * np.random.normal(0, 1, [H, D]))
                    parallel.pprint("Skipped W update. Added some noise to it.")

        # Calculate updated pi
        if "pi" in self.to_learn:
            tracing.tracepoint("M_step:update pi")
            pies_new = np.empty(H)
            comm.Allreduce([my_pies, MPI.DOUBLE], [pies_new, MPI.DOUBLE])
            pies_new /= N
            if permanent["background"]:
                pies_new[-1] = 1.0 - 1.1e-5
            pi_new = pies_new.sum() / H

        # Calculate updated sigma
        if "sigma" in self.to_learn:
            tracing.tracepoint("M_step:update sigma")
            if incmpl_data:
                sigma_new = np.sqrt(
                    (comm.allreduce(my_sigma) + comm.allreduce(my_x_infr.sum()) * sigma ** 2)
                    / N
                    / D
                )
            else:
                sigma_new = np.sqrt(comm.allreduce(my_sigma) / N / D)

        Theta_old = {"W": W.T, "pi": pi, "sigma": sigma}
        Theta_new = {"W": W_new.T, "pi": pi_new, "sigma": sigma_new}

        to_return = {
            "{}".format(k): Theta_new[k] if k in self.to_learn else Theta_old[k]
            for k in Theta_old.keys()
        }
        to_return["pies"] = pies_new  # \pi_h used only for evaluation, not for learning

        return to_return

    def modelmean(self, model_params, this_data, this_suff_stat):
        W = model_params["W"].T

        this_x = this_data["x"]
        this_ss = this_suff_stat["ss"]

        this_W = W[:, np.logical_not(this_x)]

        return np.dot(this_ss, this_W).T  # is (D_miss, S)
