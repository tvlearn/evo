# -*- coding: utf-8 -*-
# Copyright (C) 2018 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from __future__ import division
import numpy as np
from mpi4py import MPI

import evo.utils.parallel as parallel
import evo.utils.tracing as tracing
from evo.models import Model


class GaussianMCA(Model):
    def __init__(
        self,
        D,
        H,
        S,
        magnitude=False,
        sigma2_type="scalar",
        to_learn=["W", "pies", "sigma2"],
        comm=MPI.COMM_WORLD,
    ):
        assert sigma2_type in ("scalar", "diagonal", "dictionary")
        Model.__init__(self, D, H, S, to_learn, comm)

        self.magnitude = magnitude
        self.sigma2_type = sigma2_type
        eps = 1e-5
        self.noise_policy = {
            "W": (-np.inf if magnitude else eps, +np.inf, False, None),
            "pies": (eps, 1 - eps, False, None),
            "sigma2": (eps, +np.inf, False, None),
        }
        self.eps = eps
        self.tiny = np.finfo(np.float64).tiny
        self.rho_temp_bound = 1.05
        self.eps_lpj = np.finfo(np.float64).min
        self.T_rho = 1.05  # TODO: Make argument of E- and M-step methods

    @tracing.traced
    def check_params(self, model_params):
        """
        Sanity-check the given model parameters. Raises an exception if something
        is severely wrong.
        """
        model_params = Model.check_params(self, model_params)
        if self.magnitude:
            W, eps = model_params["W"], self.eps
            # Ensure |W| >= tol
            W[np.logical_and(W >= 0.0, W < +eps)] = +eps
            W[np.logical_and(W <= 0.0, W > -eps)] = -eps

        return model_params

    @tracing.traced
    def generate_from_hidden(self, model_params, my_hdata):
        W = model_params["W"].T
        sigma = np.sqrt(model_params["sigma2"])
        H, D = W.shape

        s = my_hdata["s"]
        my_N, _ = s.shape

        if self.sigma2_type == "scalar":
            assert isinstance(sigma, float)
        elif self.sigma2_type == "diagonal":
            assert isinstance(sigma, np.ndarray) and sigma.shape == (D,)
        else:  # sigma2_type == "dictionary"
            assert isinstance(sigma, np.ndarray) and sigma.shape == (D, H)

        # Create output arrays, y is data
        y = np.zeros((my_N, D))
        y_mean = np.zeros((my_N, D))
        ind_d = np.arange(D)

        for n in range(my_N):
            t0 = s[n, :, None] * W
            ind_h = np.argmax(np.abs(t0) if self.magnitude else t0, axis=0)
            y_mean[n] = (
                t0[ind_h].diagonal()
                if self.sigma2_type in ("scalar", "diagonal")
                else np.max(np.abs(t0) if self.magnitude else t0, axis=0)
            )
            _sigma = sigma if self.sigma2_type in ("scalar", "diagonal") else sigma[ind_d, ind_h]
            y[n] = y_mean[n] + _sigma * np.random.randn(D)

        return {"y": y, "s": s, "y_mean": y_mean}

    def standard_init(self, my_data, W_init=None, pi_init=None, sigma_init=None):
        model_params = Model.standard_init(self, my_data, W_init, pi_init, sigma_init)
        model_params["sigma2"] = model_params.pop("sigma") ** 2

        comm = self.comm
        D, H = self.D, self.H

        model_params["pies"] = np.random.uniform(
            low=0.1,
            high=0.5,
            size=[
                H,
            ],
        )

        if self.sigma2_type == "diagonal":
            model_params["sigma2"] *= np.ones(D)
        elif self.sigma2_type == "dictionary":
            model_params["sigma2"] *= np.ones((D, H))

        model_params = comm.bcast(model_params)

        return model_params

    @tracing.traced
    def log_pseudo_joint_permanent_states(self, model_params, my_suff_stat, my_data):
        """Compute log-pseudo-joint."""

        sigma2_type = self.sigma2_type
        this_y = my_data["this_y"]
        this_x_infr = my_data["this_x_infr"]
        pre1 = model_params["pre1"]

        permanent = my_suff_stat["permanent"]
        S_perm = my_suff_stat["S_perm"]

        lpj = np.empty((S_perm,))

        if permanent["allzero"]:
            if sigma2_type == "scalar":
                lpj[0] = pre1 * (this_y[this_x_infr] ** 2).sum()
            elif sigma2_type == "diagonal":
                lpj[0] = (pre1[this_x_infr] * this_y[this_x_infr] ** 2).sum()

        return self.lpj_reset_check(lpj, my_suff_stat)

    @tracing.traced
    def log_pseudo_joint(self, model_params, my_suff_stat, my_data):
        """Compute log-pseudo-joint."""

        sigma2_type = self.sigma2_type
        D = self.D
        tiny = self.tiny

        this_y = my_data["this_y"]
        this_x_infr = my_data["this_x_infr"]
        sigma2 = model_params["sigma2"]
        Wrho = model_params["Wrho"].T
        pil_bar = model_params["pil_bar"]  # is (H,)
        states = my_suff_stat["this_states"]  # is (S, H)
        assert (states.sum(axis=1) > 0).all() if sigma2_type == "dictionary" else True
        rho = self.rho

        pre_lpjpt = (pil_bar[None, :] * states).sum(axis=1)  # is (S,)

        t0 = np.dot(states, Wrho[:, this_x_infr])
        Wbar = (
            np.sign(t0) * np.exp(np.log(np.abs(t0) + tiny) / rho)
            if self.magnitude
            else np.exp(np.log(t0 + tiny) / rho)
        )  # is (S, D)

        if sigma2_type == "scalar":
            lpjpt = model_params["pre1"] * ((Wbar - this_y[this_x_infr]) ** 2).sum(
                axis=1
            )  # is (S,)
        elif sigma2_type == "diagonal":
            lpjpt = (model_params["pre1"][this_x_infr] * (Wbar - this_y[this_x_infr]) ** 2).sum(
                axis=1
            )  # is (S,)
        elif sigma2_type == "dictionary":
            D, S = this_x_infr.sum(), states.shape[0]
            Adh_s = states[:, :, None] * Wrho[None, :, this_x_infr]  # is (S, H, D)
            Adh_s *= 1.0 / (Adh_s.sum(axis=1)[:, None, :] + tiny)
            ind_h = np.argmax(Adh_s, axis=1)  # is (S, D)
            sigmasqbar = (
                np.tile(sigma2[None, this_x_infr, :], (S, 1, 1))
                .reshape(S * D, -1)[np.arange(S * D), ind_h.flatten()]
                .reshape(S, D)
            )  # is (S,D)
            lpjpt = -0.5 * (
                np.log(2.0 * np.pi * sigmasqbar + tiny)
                + (Wbar - this_y[this_x_infr]) ** 2 / sigmasqbar
            ).sum(
                axis=1
            )  # is (S,)

        return self.lpj_reset_check(lpjpt + pre_lpjpt, my_suff_stat)

    @tracing.traced
    def E_step_precompute(self, model_params, my_suff_stat, my_data):
        comm = self.comm
        D = self.D
        tiny = self.tiny
        sigma2_type = self.sigma2_type
        pies = model_params["pies"]
        sigma2 = model_params["sigma2"]
        W = model_params["W"]
        my_x_infr = my_data["x_infr"]
        my_N = my_x_infr.shape[0]
        N = comm.allreduce(my_N)
        incmpl_data = not my_x_infr.all()

        T_rho = np.maximum(self.T_rho, self.rho_temp_bound)
        rho = 1.0 / (1.0 - 1.0 / T_rho)
        self.rho = rho

        model_params["piH"] = pies.sum()
        model_params["pil_bar"] = np.log(pies / (1.0 - pies) + tiny)
        if sigma2_type in ("scalar", "diagonal"):
            model_params["pre1"] = -1.0 / 2.0 / sigma2

        model_params["ljc"] = (np.log(1.0 - pies) + tiny).sum()
        if not incmpl_data:
            if sigma2_type == "scalar":
                model_params["ljc"] -= D / 2 * np.log(2 * np.pi * sigma2 + tiny)
            elif sigma2_type == "diagonal":
                model_params["ljc"] -= 1 / 2 * np.log(2 * np.pi * sigma2 + tiny).sum()
        else:
            if sigma2_type == "scalar":
                sum_n_d = comm.allreduce(my_x_infr.flatten().sum())
                model_params["ljc"] -= np.log(2 * np.pi * sigma2) * sum_n_d / N / 2
            elif sigma2_type == "diagonal":
                sum_n_d = comm.allreduce(my_x_infr.sum(axis=0))
                model_params["ljc"] -= (sum_n_d / 2 / N * np.log(2 * np.pi * sigma2 + tiny)).sum()

        model_params["Wl"] = (
            np.log(np.abs(model_params["W"]) + tiny) if self.magnitude else np.log(W + tiny)
        )
        Wrho = np.exp(rho * model_params["Wl"])
        model_params["Wrho"] = np.sign(W) * Wrho if self.magnitude else Wrho

        my_suff_stat["reset_lpj_isnan"] = 0
        my_suff_stat["reset_lpj_smaller_eps_lpj"] = 0
        my_suff_stat["reset_lpj_isinf"] = 0

    @tracing.traced
    def M_step(self, model_params, my_suff_stat, my_data):
        # Array handling
        comm = self.comm
        H = self.H
        tiny = self.tiny
        sigma2_type = self.sigma2_type
        noise_policy = self.noise_policy
        my_x_infr = my_data["x_infr"]
        incmpl_data = not my_x_infr.all()
        my_y = my_data["y_reconstructed"] if incmpl_data else my_data["y"]
        D_miss = my_data["D_miss"] if incmpl_data else None
        my_N, D = my_x_infr.shape
        N = comm.allreduce(my_N)
        W = model_params["W"].T
        Wrho = model_params["Wrho"].T
        Wl = model_params["Wl"].T
        sigma2 = model_params["sigma2"].T if sigma2_type == "dictionary" else model_params["sigma2"]
        lpj = my_suff_stat["lpj"]
        ss = my_suff_stat["ss"]
        S_perm = my_suff_stat["S_perm"]
        permanent = my_suff_stat["permanent"]
        rho = self.rho

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

        Theta_new = model_params

        B = np.minimum(self.B_max - lpj.max(axis=1), self.B_max_shft)
        lpjc = lpj + B[:, None]
        pjc = np.exp(lpjc)

        my_pies = np.zeros((H))
        my_Wp = np.zeros_like(W)
        my_Wq = np.zeros_like(W)
        my_pies = np.zeros((H))
        my_Wp = np.zeros_like(W)
        my_Wq = np.zeros_like(W)

        if sigma2_type == "scalar":
            my_sigma = 0.0
        elif sigma2_type == "diagonal":
            my_sigma = np.zeros((D))
        elif sigma2_type == "dictionary":
            my_sigmasqp = np.zeros_like(sigma2)
            my_sigmasqq = np.zeros_like(sigma2)

        tracing.tracepoint("M_step:iterating")
        for n in range(my_N):
            this_y = my_y[n, :]  # is (D,)
            this_x_infr = my_x_infr[n, :]
            this_lpjc = lpjc[n, :]  # is (S,)
            this_pjc = pjc[n, :]  # is (S,)
            this_ss = ss[n, :, :]  # is (S, H)
            this_pjc_sum = this_pjc.sum()

            this_pies = (this_pjc[S_perm:].T * this_ss.T).sum(axis=1)
            my_pies += this_pies / this_pjc_sum

            # straightforward implementation of Adh term
            Adh = this_ss[:, :, None] * Wrho[None, :, :]  # is (S, H, D)
            Adh *= 1.0 / (Adh.sum(axis=1)[:, None, :] + tiny)
            xpt_Adh = (Adh * this_pjc[S_perm:][:, None, None]).sum(axis=0)
            assert np.isfinite(xpt_Adh).all()

            # # implementation of Adh term adapted from
            # # https://github.com/ml-uol/prosper/blob/master/prosper/em/camodels/mca_et.py::MCA_ET.M_step and
            # # https://github.com/ml-uol/prosper/blob/master/prosper/em/camodels/mmca_et.py::MMCA_ET.M_step
            # t0 = np.dot(this_ss, Wrho)
            # Wlbar = np.log(np.abs(t0) + tiny) / rho
            # assert np.isfinite(Wlbar).all()
            # t = Wlbar[:, None, :] - Wl[None, :, :]
            # xpt_Adh = (
            #     this_ss[:, :, None]
            #     * np.exp(
            #         this_lpjc[S_perm:, None, None]
            #         - (rho - 1) * (np.maximum(t, 0.0) if self.magnitude else t)
            #     )
            # ).sum(axis=0)
            # assert np.isfinite(xpt_Adh).all()

            this_Wp = xpt_Adh * this_y[None, :]
            this_Wq = xpt_Adh
            my_Wp += this_Wp / this_pjc_sum
            my_Wq += this_Wq / this_pjc_sum

            if sigma2_type in ("scalar", "diagonal"):
                Wlbar = np.log(np.dot(this_ss, Wrho) + tiny) / rho  # is (S, D)
                Wbar = np.exp(Wlbar)  # equal to \bar{W^{rho}}
                # eq. 7 LueckeEtAl2008
                assert np.isfinite(Wlbar).all()
                assert np.isfinite(Wbar).all()

                if sigma2_type == "scalar":
                    this_sigma = 0.0
                    if permanent["allzero"]:
                        this_sigma += (
                            this_pjc[0] * (this_y[this_x_infr] ** 2).sum()
                        )  # Zero active hidden causes
                    this_sigma += (
                        this_pjc[S_perm:]
                        * ((Wbar[:, this_x_infr] - this_y[this_x_infr]) ** 2).sum(axis=1)
                    ).sum()

                elif sigma2_type == "diagonal":
                    this_sigma = np.zeros((D,))
                    if permanent["allzero"]:
                        this_sigma[this_x_infr] += (
                            this_pjc[0] * this_y[this_x_infr] ** 2
                        )  # Zero active hidden causes
                    this_sigma[this_x_infr] += (
                        this_pjc[S_perm:][:, None]
                        * ((Wbar[:, this_x_infr] - this_y[this_x_infr]) ** 2)
                    ).sum(axis=0)
                my_sigma += this_sigma / this_pjc_sum

            elif sigma2_type == "dictionary":
                this_sigmasqp = xpt_Adh * (this_y[None, :] - W) ** 2
                this_sigmasqq = xpt_Adh
                my_sigmasqp += this_sigmasqp / this_pjc_sum
                my_sigmasqq += this_sigmasqq / this_pjc_sum

        # Calculate updated W
        if "W" in self.to_learn:
            tracing.tracepoint("M_step:update W")
            Wp = np.empty_like(my_Wp)
            Wq = np.empty_like(my_Wq)

            assert np.isfinite(my_Wp).all()
            assert np.isfinite(my_Wq).all()

            comm.Allreduce([my_Wp, MPI.DOUBLE], [Wp, MPI.DOUBLE])
            comm.Allreduce([my_Wq, MPI.DOUBLE], [Wq, MPI.DOUBLE])

            Wp_st = Wp < tiny
            if Wp_st.any():
                parallel.pprint("M_step: Reset lower bounds of nominator in W update")
                parallel.pprint("Wp min/max = %.2f/%.2f" % (np.amin(Wp), np.amax(Wp)))
                Wp[Wp_st] = tiny
            Wq_st = Wq < tiny
            if Wq_st.any():
                parallel.pprint("M_step: Reset lower bounds of denominator in W update")
                parallel.pprint("Wq min/max = %.2f/%.2f" % (np.amin(Wq), np.amax(Wq)))
                Wq[Wq_st] = tiny

            W_new = (Wp / Wq).T
            W_new[W_new < tiny] = tiny

            if not np.isfinite(W_new).all():
                W_new = W.T + (5e5 * np.random.normal(0, 1, [D, H]))
                W_new[W_new < tiny] = tiny
                parallel.pprint("Skipped W update. Added some noise to it.")

            Theta_new["W"] = W_new

        # Calculate updated pi
        if "pies" in self.to_learn:
            tracing.tracepoint("M_step:update pies")
            piesu = np.empty(H)
            comm.Allreduce([my_pies, MPI.DOUBLE], [piesu, MPI.DOUBLE])
            pies_new = piesu / N
            if permanent["background"]:
                pies_new[H - 1] = 0.9999 * noise_policy["pies"][1]
            Theta_new["pies"] = pies_new

        # Calculate updated sigma
        if "sigma2" in self.to_learn:
            tracing.tracepoint("M_step:update sigma")
            if sigma2_type == "scalar":
                sigma2_new = (
                    ((comm.allreduce(my_sigma) + comm.allreduce(my_x_infr.sum()) * sigma2) / N / D)
                    if incmpl_data
                    else (comm.allreduce(my_sigma) / N / D)
                )

            elif sigma2_type == "diagonal":
                sigma2_new = (
                    ((comm.allreduce(my_sigma) + D_miss * sigma2) / N)
                    if incmpl_data
                    else ((comm.allreduce(my_sigma)) / N)
                )

            elif sigma2_type == "dictionary":
                sigmasqp = np.empty_like(my_sigmasqp)
                sigmasqq = np.empty_like(my_sigmasqq)

                assert np.isfinite(my_sigmasqp).all()
                assert np.isfinite(my_sigmasqq).all()

                comm.Allreduce([my_sigmasqp, MPI.DOUBLE], [sigmasqp, MPI.DOUBLE])
                comm.Allreduce([my_sigmasqq, MPI.DOUBLE], [sigmasqq, MPI.DOUBLE])

                sigmasqp_st = sigmasqp < tiny
                if sigmasqp_st.any():
                    parallel.pprint("M_step: Reset lower bounds of nominator in sigma update")
                    parallel.pprint(
                        "sigmasqp min/max = %.2f/%.2f" % (np.amin(sigmasqp), np.amax(sigmasqp))
                    )
                    sigmasqp[sigmasqp_st] = tiny
                sigmasqq_st = sigmasqq < tiny
                if sigmasqq_st.any():
                    parallel.pprint("M_step: Reset lower bounds of denominator in sigma update")
                    parallel.pprint(
                        "sigmasqq min/max = %.2f/%.2f" % (np.amin(sigmasqq), np.amax(sigmasqq))
                    )
                    sigmasqq[sigmasqq_st] = tiny

                sigma2_new = ((sigmasqp) / sigmasqq).T
                sigma2_new[sigma2_new < tiny] = tiny

                if not np.isfinite(sigma2_new).all():
                    sigma2_new = sigma2.T + (5e5 * np.random.normal(0, 1, [D, H]))
                    sigma2_new[sigma2_new < tiny] = tiny
                    parallel.pprint("Skipped sigma update. Added some noise to it.")

            Theta_new["sigma2"] = sigma2_new

        return Theta_new

    def modelmean(self, model_params, this_data, this_suff_stat):
        tiny = self.tiny
        rho = self.rho
        Wrho = model_params["Wrho"].T
        this_x = this_data["x"]
        this_ss = this_suff_stat["ss"]

        t0 = np.dot(this_ss, Wrho[:, np.logical_not(this_x)])
        Wlbar = np.log(np.abs(t0) if self.magnitude else t0 + tiny) / rho
        return (np.sign(t0) * np.exp(Wlbar)).T if self.magnitude else np.exp(Wlbar).T
