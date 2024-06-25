#!/usr/bin/env python3

import functools
import logging
import operator

import lmfit
import numpy as np
import scipy.constants
import scipy.interpolate
import scipy.optimize

logger = logging.getLogger(__name__)
k_b = scipy.constants.k / scipy.constants.erg  # J/K -> erg/K
mu_b = scipy.constants.physical_constants["Bohr magneton"][0] * 1000  # erg/g, emu


def model_sum(mlist):
    return functools.reduce(operator.add, mlist)


def voltage_drift(x, v_offset=0, v_drift=0):
    v_i = np.linspace(0, 1, len(x))
    return v_offset + v_drift * v_i


def dipole_response(x, moment=1e-4, center=2):
    R = 0.97
    L = 1.519
    F = 0.9125
    X = R**2 + (x - center) ** 2
    Y = R**2 + (L + (x - center)) ** 2
    Z = R**2 + (-L + (x - center)) ** 2
    return moment * F * (2 * X ** (-3 / 2) - Y ** (-3 / 2) - Z ** (-3 / 2))


def n_dipole_model(n):
    background = lmfit.Model(voltage_drift)
    dipoles = [lmfit.Model(dipole_response, prefix=f"d{i}_") for i in range(n)]
    model = model_sum([background] + dipoles)
    for i in range(n):
        model.set_param_hint(f"d{i}_moment", min=-1, max=1)
        model.set_param_hint(f"d{i}_center", min=-1, max=6)
    return model


def curie_weiss(T, H=200, T_C=50, N=1e17, M_0=0, g=2, J=7 / 2):
    C = mu_b**2 / (3 * k_b) * N * g**2 * J * (J + 1)
    return np.piecewise(
        T,
        [T < T_C, T >= T_C],
        [np.zeros_like, lambda T: H * (T - T_C) / (H * C - M_0 * (T - T_C))],
    )


def fit_inverse_susceptibility(T, M, H=200, T_C=50, N=1e16, g=2, J=7 / 2, M_0=0):
    if M.min() < 0:
        logger.warning("Trying to fit inverse susceptibility with negative moments")
        M_0 = 1.1 * M.min()
    elif M_0 is None:
        M_0 = 0.9 * M.min()
    inv_susc = H / (M - M_0)
    model = lmfit.Model(curie_weiss)
    model.set_param_hint("H", value=H, vary=False)
    model.set_param_hint("T_C", value=T_C, min=0, max=300)
    model.set_param_hint("N", value=N)
    model.set_param_hint("g", value=g, vary=False)
    model.set_param_hint("J", value=J, vary=False)
    res = model.fit(inv_susc, T=T, weights=0.01 * inv_susc)
    M_0_re = M_0 - res.best_values["M_0"]
    inv_susc2 = H / (M - M_0_re)
    res2 = model.fit(inv_susc2, T=T, weights=0.01 * inv_susc)
    return res, res2, M_0_re


def flood(arr, value):
    r = arr.copy()
    while True:
        idx = np.where(r == value)[0]
        idx = np.unique(np.stack((idx + 1, idx - 1)))
        idx = idx[(idx >= 0) * (idx < len(r))]
        if not np.any(r[idx] == 0):
            return r
        r[idx] = np.where(r[idx] == 0, value, r[idx])


def pad_slice(s):
    if s[0] - 1 >= 0:
        return np.insert(s, 0, s[0] - 1)
    return s


def monotonic_slices(x):
    x = np.asarray(x)
    dx = np.sign(np.ediff1d(x, to_begin=0))
    decreasing = np.where(flood(dx, -1) == -1)[0]
    increasing = np.where(flood(dx, 1) == 1)[0]
    d_infl = np.where(np.diff(decreasing) > 1)[0] + 1
    i_infl = np.where(np.diff(increasing) > 1)[0] + 1
    d_slices = ((pad_slice(s), -1) for s in np.split(decreasing, d_infl) if np.any(s))
    i_slices = ((pad_slice(s), 1) for s in np.split(increasing, i_infl) if np.any(s))
    if 0 in decreasing:
        yield next(d_slices)
    while True:
        try:
            yield next(i_slices)
            yield next(d_slices)
        except StopIteration:
            break


def signed_interpolation(x, y):
    for i, sgn in monotonic_slices(x):
        j = np.nonzero(np.ediff1d(x[i], to_begin=0))[0]
        x_i = np.array([q.mean() for q in np.split(x[i], j)])
        y_i = np.array([q.mean() for q in np.split(y[i], j)])
        yield scipy.interpolate.interp1d(
            x_i, y_i, kind="slinear", fill_value="extrapolate"
        ), sgn, x_i, y_i


def estimate_coercivity(H, M):
    for f, s, H_i, M_i in signed_interpolation(H, M):
        if not (M_i.min() < 0 < M_i.max()):
            logger.warn(
                f"Magnetic moment does not cross zero ({M_i.min():.3e} -> {M_i.max():.3e})"
            )
            continue
        yield scipy.optimize.fsolve(f, 0)[0], s


def estimate_remanence(H, M):
    for f, s, H_i, M_i in signed_interpolation(H, M):
        if not (H_i.min() < 0 < H_i.max()):
            logger.warn(
                f"Applied field does not cross zero ({H_i.min():.3e} -> {H_i.max():.3e})"
            )
            continue
        yield f(0), s


def r_square(y, yf):
    total_sum_of_squares = y.var(axis=-1, ddof=1) * y.shape[-1]
    sum_of_squares_of_residuals = np.sum((y - yf) ** 2, axis=-1)
    R2 = 1 - sum_of_squares_of_residuals / total_sum_of_squares
    return 1 - sum_of_squares_of_residuals / total_sum_of_squares
