#!/usr/bin/env python3

import collections
import csv
import functools
import json
import logging
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

from . import fitting, reference
from .fitting import (
    dipole_response,
    estimate_coercivity,
    estimate_remanence,
    fit_inverse_susceptibility,
    n_dipole_model,
    r_square,
)
from .interactive import toggle_legend

logger = logging.getLogger(__name__)
mu_b = scipy.constants.physical_constants["Bohr magneton"][0] * 1000  # erg/g, emu


class Sample:
    def __init__(self):
        pass

    @property
    def volume(self):
        pass

    @property
    def mass(self):
        pass

    @property
    def n(self):
        pass


class FilmSample(Sample):
    def __init__(
        self,
        sample_mass,
        substrate,
        substrate_thickness,
        film_thickness,
        n_density=None,
    ):
        # cgs units
        self.sample_mass = float(sample_mass)
        self.substrate = substrate
        self.substrate_thickness = float(substrate_thickness)
        self.film_thickness = float(film_thickness)
        self.n_density = float(n_density)

    @property
    def substrate_volume(self):
        return self.sample_mass / self.substrate.density

    @property
    def substrate_area(self):
        return self.substrate_volume / self.substrate_thickness

    @property
    def volume(self):
        return self.substrate_area * self.film_thickness

    @property
    def n(self):
        if self.n_density is None:
            return None
        return self.volume * self.n_density

    @classmethod
    def load_json(cls, fname):
        with open(fname, "r") as f:
            data = json.load(f)
        return cls(
            data["sample_mass"],
            reference.substrates[data["substrate"]],
            data["substrate_thickness"],
            data["film_thickness"],
            reference.n_densities[data["film_material"]],
        )


class MPMS:
    """Objects for reading and processing MPMS data files measuring
    longitudinal moment in RSO or DC modes.

    Attempts to load .dat, .ndat, .raw files with the same base filename into
    np recarrays and provides references to useful data columns as
    attributes.

    Parameters:
    fname : str_like
        File path to a specific MPMS data file (extensions .dat, .ndat or .raw)
        or to the root of one or more of the preceding files (e.g. data.rso)

    Attributes:
    root : str
        Path to the base filename of the loaded data
    moment_offset : float
        Constant background offset to be subtracted from moment values in emu
        (default = 0)
    magnetization_unit : str
        Unit of magnetization that moment is converted to
        (default = 'emu')
    magnetization_conversion : float
        Conversion factor for moment -> magnetization, i.e.
        magnetization = moment*magnetization_conversion
        (default = 1)
    background : np.ndarray
        Voltage background that is subtracted from raw voltage

    Properties:
    dat[ndat,raw] : np.ndarray or None
        Array of data loaded from {root}.dat if file exists
    dat[ndat,raw]_header : list of str or None
        Header information from start of {root}.dat
    field_dependent : bool
        True if field values in loaded data are not constant
    temperature_dependent : bool
        True if not field_dependent
    xlabel : str
        x-axis label for plotting either field- or temperature-dependent data
    time : np.ndarray
        dat['Time']
        Column of timestamp data (dat, raw)
    field : np.ndarray
        dat['Field_Oe']
        Column of applied field data (dat, raw)
    temperature : np.ndarray
        dat['Temperature_K']
        Column of temperature data (dat, raw)
    temperature_spread : np.ndarray
        dat['Delta_Temp_K']
        Column of temperature delta (dat, raw)
    moment : np.ndarray
        dat['Long_Moment_emu']
        Column of moment with moment_offset subtracted (dat, raw)
    magnetization : np.ndarray
        Magnetic moment converted to magnetization using the
        magnetization_conversion factor (dat, raw)
    fit_R2 : np.ndarray
        dat['Long_Reg_Fit']
        Column of regression fit values of each scan (dat, raw)
    scan_amplitude : np.ndarray
        dat['Amplitude_cm']
        Column of measurement scan widths (dat, raw)
    delta_temperature : np.ndarray
        dat['Delta_Temp_K']
        Column of temperature spread for each scan (dat, raw)
    ntime : np.ndarray
        ndat['Time']
        Column of timestamps for averaged data (ndat)
    nfield : np.ndarray
        ndat['Field_Oe']
        Column of averaged applied field (ndat)
    ntemperature : np.ndarray
        ndat['Avg_Temperature_K']
        Column of averaged temperatures (ndat)
    nmoment : np.ndarray
        ndat['Avg_Moment_emu']
        Column of averaged moment (ndat)
    nmagnetization : np.ndarray
        ndat['Avg_Moment_emu']
        Average magnetic moment converted to magnetization using the
        magnetization_conversion factor (ndat)
    ndelta_temperature : np.ndarray
        ndat['Delta_Temp_K']
        Column of averaged temperature spread (ndat)
    position : np.ndarray
        raw['Position_cm']
        2D array of raw position data (raw)
    raw_voltage : np.ndarray
        raw['Long_Scaled_Response']
        2D array of raw scaled voltage data (raw)
    voltage : np.ndarray
        raw['Long_Scaled_Response'] - self.background
        2D array of background-subtracted raw scaled voltage data (raw)
    raw_fit_voltage : np.ndarray
        raw['Long_Reg_Fit']*self.scale_factor
        2D array of MPMS fit to scaled voltage data (raw)
    raw_moment : np.ndarray
        Moment values from fits, defaulting to np.nan if no fit has been
        performed for a given scan index
    raw_R2 : np.ndarray
        R^2 regression values from fits
    """

    def __init__(self, fname, sample=None):
        file_extensions = ["dat", "ndat", "raw"]
        ext_prog = re.compile(
            f'({"|".join(re.escape("." + ext) for ext in file_extensions)})$'
        )
        self.fname = fname
        self.root = root = ext_prog.sub("", str(fname))
        # moment unit conversion
        self.moment_offset = 0
        self.magnetization_unit = "emu"
        self.magnetization_conversion = 1
        self.sample = sample
        self.coercivities = []
        self.remanences = []
        # Initialized when raw data accessed
        self.scale_factor = None
        self.background = None
        self.background_dipoles = []
        self.fits = {}

    def __repr__(self):
        return f"<MPMS: {self.root}>"

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def parse_mpms(fname):
        header = []
        try:
            with open(fname, "r") as f:
                for i, line in enumerate(f, start=1):
                    if "[Data]" in line:
                        break
                    header.append(line.strip())
                # csv reader handles delimiters inside quotes
                reader = csv.reader(f)
                data = np.genfromtxt(
                    ("\t".join(row) for row in reader), delimiter="\t", names=True
                )
        except FileNotFoundError:
            logger.warn(f"Error parsing {fname}: file not found")
            return None, None
        used_columns = [c for c in data.dtype.names if ~np.all(np.isnan(data[c]))]
        data = data[used_columns]
        if fname.endswith(".raw"):
            n_scan = np.ediff1d(data["Time"], to_begin=0).nonzero()[0][0]
            data = data.reshape(-1, n_scan)
        return header, data

    @property
    @functools.lru_cache()
    def dat_header(self):
        return self.parse_mpms(self.root + ".dat")[0]

    @property
    @functools.lru_cache()
    def dat(self):
        return self.parse_mpms(self.root + ".dat")[1]

    @property
    @functools.lru_cache()
    def ndat_header(self):
        return self.parse_mpms(self.root + ".ndat")[0]

    @property
    @functools.lru_cache()
    def ndat(self):
        return self.parse_mpms(self.root + ".ndat")[1]

    @property
    @functools.lru_cache()
    def raw_header(self):
        return self.parse_mpms(self.root + ".raw")[0]

    @property
    @functools.lru_cache()
    def raw(self):
        _raw = self.parse_mpms(self.root + ".raw")[1]
        if _raw is None:
            return None
        scaled_ptp = np.ptp(_raw["Long_Scaled_Response"], axis=1)
        unscaled_ptp = np.ptp(_raw["Long_Voltage"], axis=1)
        self.scale_factor = (scaled_ptp / unscaled_ptp).reshape(-1, 1)
        self.background = np.zeros(_raw.shape)
        return _raw

    @property
    @functools.lru_cache()
    def field_dependent(self):
        return bool(np.ptp(self.field))

    @property
    @functools.lru_cache()
    def temperature_dependent(self):
        return not self.field_dependent

    @property
    @functools.lru_cache()
    def xlabel(self):
        if self.field_dependent:
            return "Field (Oe)"
        return "Temperature (K)"

    @property
    def time(self):
        return self.dat["Time"]

    @property
    def field(self):
        return self.dat["Field_Oe"]

    @property
    def temperature(self):
        return self.dat["Temperature_K"]

    @property
    def set_temperature(self):
        if not self.field_dependent:
            logger.warn(f"{self.root}.dat has no set temperature value")
        return round(self.temperature.mean(), 1)

    @property
    def temperature_spread(self):
        return np.ptp(self.temperature)

    @property
    def moment(self):
        return self.dat["Long_Moment_emu"] - self.moment_offset

    @property
    def magnetization(self):
        return self.moment * self.magnetization_conversion

    @property
    def fit_R2(self):
        return self.dat["Long_Reg_Fit"]

    @property
    def fixed_fit_R2(self):
        return r_square(self.raw_voltage, self.raw_fit_voltage)

    @property
    def scan_amplitude(self):
        try:
            return self.dat["Amplitude_cm"]
        except ValueError:
            return self.dat["Scan_Length_cm"]

    @property
    def delta_temperature(self):
        return self.dat["Delta_Temp_K"]

    @property
    def ntime(self):
        return self.ndat["Time"]

    @property
    def nfield(self):
        return self.ndat["Field_Oe"]

    @property
    def ntemperature(self):
        return self.ndat["Avg_Temperature_K"]

    @property
    def nmoment(self):
        return self.ndat["Avg_Moment_emu"]

    @property
    def nmagnetization(self):
        return self.nmoment * self.moment_conversion

    @property
    def nfit_R2(self):
        return self.ndat["Avg_Reg_Fit"]

    @property
    def ndelta_temperature(self):
        return self.ndat["Delta_Temp_K"]

    @property
    def position(self):
        return self.raw["Position_cm"]

    @property
    def raw_voltage(self):
        return self.raw["Long_Scaled_Response"]

    @property
    def voltage(self):
        return self.raw_voltage - self.background

    @property
    def raw_fit_voltage(self):
        try:
            return self.raw["Long_Reg_Fit"] * self.scale_factor
        except ValueError:
            return self.raw["Long_Regression_Fit"] * self.scale_factor

    @property
    def raw_moment(self):
        moment = np.full_like(self.moment, np.nan)
        for i, fit in self.fits.items():
            moment[i] = fit.best_values["d0_moment"]
        return moment

    @property
    def raw_center(self):
        center = np.full_like(self.moment, np.nan)
        for i, fit in self.fits.items():
            center[i] = fit.best_values["d0_center"]
        return center

    @property
    def raw_R2(self):
        R2 = np.full_like(self.fit_R2, np.nan)
        for i, fit in self.fits.items():
            R2[i] = r_square(fit.data, fit.best_fit)
        return R2

    def add_dipole_background(self, moment, center, n_dipoles=1):
        model = n_dipole_model(n_dipoles)
        self.background_dipoles.append((moment, center))
        self.background += dipole_response(self.position, moment, center)

    def set_scan_background(self, idx, center, n_dipoles=1, **kwargs):
        model = n_dipole_model(n_dipoles)
        model.set_param_hint("v_drift", vary=False)
        m0 = np.ptp(self.voltage[idx]) / 2.4  # initial estimate for moment
        res = model.fit(
            self.voltage[idx],
            x=self.position[idx],
            d0_center=center,
            d0_moment=m0,
            **kwargs,
        )
        self.add_dipole_background(
            res.best_values["d0_moment"], res.best_values["d0_center"]
        )

    def reset_background(self):
        self.background = np.zeros(self.raw.shape)
        self.background_dipoles = []

    def scan_fit(self, idx, d0_center, n_dipoles=1, model=None, **kwargs):
        if model is None:
            model = n_dipole_model(n_dipoles)
        m0 = np.ptp(self.voltage[idx]) / 2.4  # initial estimate for moment
        res = model.fit(
            self.voltage[idx],
            x=self.position[idx],
            d0_center=d0_center,
            d0_moment=m0,
            **kwargs,
        )
        self.fits[idx] = res
        return res

    def scan_fit_all(self, d0_center, n_dipoles=1, **kwargs):
        model = n_dipole_model(n_dipoles)
        for idx in range(len(self.position)):
            self.scan_fit(idx, d0_center=d0_center, model=model, **kwargs)

    def export_refit(self, fname=None):
        if fname is None:
            fname = self.root + ".refit.dat"
        with open(fname, "w") as f:
            for line in self.dat_header:
                f.write(line + "\n")
            f.write("[Data]" + "\n")
            colnames = [
                "Time",
                "Field (Oe)",
                "Temperature (K)",
                "Long Moment (emu)",
                "Long Offset (cm)",
                "Long Reg Fit",
                "Delta Temp (K)",
            ]
            f.write(",".join(colnames) + "\n")
            data = np.column_stack(
                [
                    self.time,
                    self.field,
                    self.temperature,
                    self.raw_moment,
                    self.raw_center,
                    self.raw_R2,
                    self.delta_temperature,
                ]
            )
            fmt = ["%10.3f"] + ["%1.6e"] * (len(colnames) - 1)
            # filter rows with nan values
            idx = np.isnan(self.raw_moment)
            np.savetxt(f, data, fmt=fmt, delimiter=",")
        return fname

    def plot_dat(self, ax=None, label=None, min_fit=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            ax.set_xlabel(self.xlabel)
            if self.magnetization_conversion == 1:
                ax.set_ylabel("Moment (emu)")
            else:
                ax.set_ylabel(f"Magnetization ({self.magnetization_unit})")
        if min_fit is not None:
            idx = self.fit_R2 >= min_fit
            n_missed = len(self.fit_R2) - idx.sum()
            if n_missed:
                logger.warning(
                    f"{n_missed} points with fit value less than {min_fit} in {self.root}.dat"
                )
        else:
            idx = slice(None)
        if self.field_dependent:
            return ax.plot(
                self.field[idx], self.magnetization[idx], label=label, **kwargs
            )[0]
        else:
            s = np.sign(self.temperature[-1] - self.temperature[0]) < 0
            if self.field[0] == 0:
                label = "Zero field"
            else:
                label = f"{['ZFC', 'FC'][s]} {self.field[0]:.1f} Oe"
            return ax.plot(
                self.temperature[idx], self.magnetization[idx], label=label, **kwargs
            )[0]

    def plot_fit_value(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel("R^2 value")
        if self.field_dependent:
            return ax.plot(self.field, self.fixed_fit_R2, **kwargs)[0]
        else:
            return ax.plot(self.temperature, self.fixed_fit_R2, **kwargs)[0]

    def plot_raw_fit_value(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel("R^2 value")
        if self.field_dependent:
            return ax.plot(self.field, self.raw_R2, **kwargs)[0]
        else:
            return ax.plot(self.temperature, self.raw_R2, **kwargs)[0]

    def plot_inverse_susceptibility(self, ax, **kwargs):
        if self.field_dependent:
            logger.warning(
                f"{self.root}.dat is not a temperature-dependent measurement"
            )
            return
        if self.field[0] == 0:
            logger.warning(f"{self.root}.dat applied field is zero")
            return
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel(f"1/$\\chi$ (Oe/{self.magnetization_unit})")
        s = np.sign(self.temperature[-1] - self.temperature[0]) < 0
        label = f"{['ZFC', 'FC'][s]} {self.field[0]:.1f} Oe"
        return ax.plot(
            self.temperature, self.field[0] / self.magnetization, label=label, **kwargs
        )[0]

    def estimate_curie_temperature(self, window_length=11, polyorder=3):
        idx = scipy.signal.savgol_filter(
            self.moment, window_length, polyorder, deriv=2
        ).argmax()
        return self.temperature[idx]

    def estimate_coercivity(self):
        self.coercivities = list(estimate_coercivity(self.field, self.magnetization))

    def estimate_remanence(self):
        self.remanences = list(estimate_remanence(self.field, self.magnetization))

    def fit_curie_weiss(
        self, ax=None, T_C=None, ion="Gd", T_max=300, update_offset=False, **params
    ):
        if not self.temperature_dependent:
            logger.warn(
                f"Curie-Weiss can only fit temperature dependence ({self.root}.dat)"
            )
            return
        if T_C is None:
            T_C = self.estimate_curie_temperature()
        r_e = reference.ground_states[ion]
        idx = self.temperature < T_max
        res, res2, M_0 = fit_inverse_susceptibility(
            self.temperature[idx],
            self.moment[idx],
            H=self.field[0],
            T_C=T_C,
            g=r_e.g,
            J=r_e.J,
            **params,
        )
        if update_offset:
            self.moment_offset = M_0
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            ax.set_xlabel("Temperature (K)")
            ax.set_ylabel("1/$\\chi$ (Oe/emu)")
        s = np.sign(self.temperature[-1] - self.temperature[0]) < 0
        label = f"1/$\\chi$ {['ZFC', 'FC'][s]} {self.field[0]:.1f} Oe"
        ax.plot(res2.userkws["T"], res2.data, label=label, marker=".")
        ax.plot(
            res2.userkws["T"],
            res2.best_fit,
            label=f'T$_C$ = {res2.best_values["T_C"]:.1f} K\nN = {res2.best_values["N"]:.2e}',
        )
        toggle_legend(ax, loc="best")
        self.curie_weiss_fits = [res, res2, M_0]

    @property
    def sample(self):
        return self._sample

    @sample.setter
    def sample(self, s):
        self._sample = s
        if s is None:
            self.magnetization_unit = "emu"
            self.magnetization_conversion = 1
            return
        # try in order n, volume, mass
        if s.n:
            self.magnetization_unit = "$\\mu_{B}$/ion"
            self.magnetization_conversion = 1 / (mu_b * s.n)
        elif s.volume:
            self.magnetization_unit = "emu/cm$^3$"
            self.magnetization_conversion = 1 / s.volume
        elif s.mass:
            self.magnetization_unit = "emu/g"
            self.magnetization_conversion = 1 / s.mass


def plot_combined(mpms_list, fig=None, T_max=300, min_fit=0, fit_curie_weiss=True):
    if fig is None:
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16, 9))
    else:
        ((ax1, ax2, ax3), (ax4, ax5, ax6)) = fig.subplots(2, 3)
    for ax in fig.axes:
        ax.axhline(0, color="0.9")
        ax.axvline(0, color="0.9")
    for ax in [ax1, ax2, ax4, ax5, ax6]:
        ax.ticklabel_format(axis="y", scilimits=(0, 0))
    ax1.sharex(ax4)
    ax3.sharex(ax6)
    moment_unit = mpms_list[0].magnetization_unit
    ax1.set_ylabel(f"Moment ({moment_unit})")
    ax2.set_xlabel("Field (Oe)")
    ax2.set_ylabel(f"Moment ({moment_unit})")
    ax3.set_ylabel("Coercivity (Oe)")
    ax4.set_ylabel("1/$\\chi$ (Oe/emu)")
    ax4.set_xlabel("Temperature (K)")
    ax5.set_xlabel("Field (Oe)")
    ax5.set_ylabel(f"Moment ({moment_unit})")
    ax6.set_xlabel("Temperature (K)")
    ax6.set_ylabel(f"Remanence ({moment_unit})")

    temperature_dependent = [mpms for mpms in mpms_list if mpms.temperature_dependent]
    field_dependent = sorted(
        [mpms for mpms in mpms_list if mpms.field_dependent],
        key=lambda x: x.temperature[0],
    )
    for mpms in field_dependent[:]:
        if mpms.temperature_spread > 1:
            logger.warn(
                f"Temperature spread in {mpms.root} too large ({mpms.temperature_spread} K)"
            )
            field_dependent.remove(mpms)
    for mpms in temperature_dependent:
        try:
            mpms.plot_dat(ax=ax1, min_fit=min_fit)
        except ValueError:
            logger.warn(f"Error plotting {mpms.fname}")
            continue
        if mpms.field[0] and fit_curie_weiss:
            mpms.fit_curie_weiss(ax=ax4, T_max=T_max)
        else:
            mpms.plot_inverse_susceptibility(ax=ax4)
    if not field_dependent:
        toggle_legend(ax1)
        toggle_legend(ax4)
        return fig, fig.axes
    # color depending on temperature
    cmap = plt.get_cmap("plasma")
    norm = mpl.colors.Normalize(
        vmin=field_dependent[0].set_temperature,
        vmax=field_dependent[-1].set_temperature * 1.3,
    )
    for mpms in field_dependent:
        t = mpms.set_temperature
        try:
            mpms.plot_dat(
                ax=ax2, color=cmap(norm(t)), label=f"{t} K", marker=".", min_fit=min_fit
            )
        except ValueError:
            logger.warn(f"Error plotting {mpms.fname}")
            continue
        mpms.plot_dat(
            ax=ax5, color=cmap(norm(t)), label=f"{t} K", marker=".", min_fit=min_fit
        )
        mpms.estimate_coercivity()
        mpms.estimate_remanence()
    H_ci = np.array(
        [
            (mpms.set_temperature, h_c[0])
            for mpms in field_dependent
            for h_c in mpms.coercivities
            if h_c[1] == 1
        ]
    )
    H_cd = np.array(
        [
            (mpms.set_temperature, h_c[0])
            for mpms in field_dependent
            for h_c in mpms.coercivities
            if h_c[1] == -1
        ]
    )
    M_ri = np.array(
        [
            (mpms.set_temperature, m_r[0])
            for mpms in field_dependent
            for m_r in mpms.remanences
            if m_r[1] == 1
        ]
    )
    M_rd = np.array(
        [
            (mpms.set_temperature, m_r[0])
            for mpms in field_dependent
            for m_r in mpms.remanences
            if m_r[1] == -1
        ]
    )
    ax3.plot(H_ci.T[0], H_ci.T[1], color="r", marker=".", label="Increasing Field")
    ax3.plot(H_cd.T[0], -H_cd.T[1], color="b", marker=".", label="Decreasing Field")
    ax6.plot(M_ri.T[0], -M_ri.T[1], color="r", marker=".", label="Increasing Field")
    ax6.plot(M_rd.T[0], M_rd.T[1], color="b", marker=".", label="Decreasing Field")
    xlim = (
        1.5
        * np.abs(
            [h_c[0] for mpms in field_dependent for h_c in mpms.coercivities]
        ).max()
    )
    ylim = (
        1.3
        * np.abs([m_r[0] for mpms in field_dependent for m_r in mpms.remanences]).max()
    )
    ax5.set_xlim(-xlim, xlim)
    ax5.set_ylim(-ylim, ylim)
    for ax in fig.axes:
        toggle_legend(ax)
    fig.tight_layout()
    return fig, fig.axes
