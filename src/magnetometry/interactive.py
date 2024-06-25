#!/usr/bin/env python3

import functools
import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from .fitting import dipole_response

logger = logging.getLogger(__name__)


class RawScanViewer:
    def __init__(self, mpms, fig=None, toolbar=None):
        self.mpms = mpms
        if self.mpms.raw is None:
            logger.warning("No raw data in {self.mpms.root}")
            return
        self.inverse_susceptibility_mode = False
        self.add_dipole_mode = False
        self.shift_pressed = False
        self.mouse_pressed = False
        self.dipole_moment = None
        self.dipole_center = None
        self.shift_x = None
        self.shift_y = None
        self.create_viewer(fig=fig)
        self.select_scan_idx(0)
        if toolbar is not None:
            self.toolbar = toolbar

    @functools.cached_property
    def toolbar(self):
        return self.fig.canvas.manager.toolbar

    def create_viewer(self, fig=None):
        # create figure axes, labels
        if fig is None:
            self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
            ((original_ax, moment_ax), (refit_ax, fit_value_ax)) = self.axes
        else:
            self.fig = fig
            fig.clear()
            original_ax = fig.add_subplot(2, 2, 1)
            moment_ax = fig.add_subplot(2, 2, 2)
            refit_ax = fig.add_subplot(2, 2, 3, sharex=original_ax)
            fit_value_ax = fig.add_subplot(2, 2, 4, sharex=moment_ax)
            self.axes = np.array([[original_ax, moment_ax], [refit_ax, fit_value_ax]])
        self.original_ax = original_ax
        self.moment_ax = moment_ax
        self.refit_ax = refit_ax
        self.fit_value_ax = fit_value_ax
        original_ax.set_ylabel("Scaled voltage")
        refit_ax.set_xlabel("Position (cm)")
        refit_ax.set_ylabel("Scaled voltage")
        moment_ax.set_ylabel("Moment (emu)")
        fit_value_ax.set_xlabel(self.mpms.xlabel)
        fit_value_ax.set_ylabel("R$^2$ value")
        original_ax.ticklabel_format(axis="y", scilimits=(0, 0))
        refit_ax.ticklabel_format(axis="y", scilimits=(0, 0))
        moment_ax.ticklabel_format(axis="y", scilimits=(0, 0))
        self.fig.subplots_adjust(bottom=0.25, top=0.775)
        slider_ax = self.fig.add_axes([0.125, 0.125, 0.775, 0.05])
        self.scan_slider = Slider(
            ax=slider_ax,
            label="Scan #",
            valmin=0,
            valmax=len(self.mpms.raw) - 1,
            valstep=1,
        )
        self.scan_slider.on_changed(self.select_scan_idx)

        # buttons for setting background, clear background, fit one/all, export
        self.buttons = []
        buttons = [
            ("Add dipole\nbackground", self.add_dipole_background),
            ("Fit dipole\nbackground", self.fit_dipole_background),
            ("Undo last\ndipole", self.undo_last_dipole),
            ("Clear\nbackground", self.clear_background),
            ("Fit single\nscan", self.fit_single_scan),
            ("Fit all\nscans", self.fit_all_scans),
            ("Toggle 1/$\\chi$", self.toggle_inverse_susceptibility_mode),
            ("Export\nrefit", self.export_refit),
        ]
        for i, (label, callback) in enumerate(buttons):
            w = 0.8 / len(buttons)
            button_ax = self.fig.add_axes([0.125 + i * w, 0.825, w * 0.8, 0.075])
            button = Button(button_ax, label)
            button.on_clicked(callback)
            self.buttons.append(button)

        # set up lines to be updated

        # original_ax
        (self.raw_line,) = original_ax.plot(
            self.mpms.position[0],
            self.mpms.voltage[0],
            label="Raw data",
            marker=".",
            linestyle="",
        )
        (self.fit_line,) = original_ax.plot(
            self.mpms.position[0],
            self.mpms.raw_fit_voltage[0],
            label="MPMS fit",
            marker="",
            linestyle="-",
            alpha=0.8,
        )
        (self.background_line,) = original_ax.plot(
            self.mpms.position[0],
            self.mpms.background[0],
            label="Background",
            marker="",
            linestyle="-",
            alpha=0.8,
        )
        (self.original_dipole_line,) = original_ax.plot(
            [], [], label="Dipole", marker="", linestyle="-", alpha=0.8
        )
        self.center_marker_o = original_ax.axvline(
            self.mpms.scan_amplitude[0] / 2, color="C5"
        )

        # refit_ax
        (self.raw_line_corrected,) = refit_ax.plot(
            self.mpms.position[0],
            self.mpms.voltage[0],
            label="Raw - background",
            marker=".",
            linestyle="",
        )
        (self.refit_line,) = refit_ax.plot(
            [], [], label="Refit", marker="", linestyle="-", alpha=0.8
        )
        (self.refit_dipole_line,) = refit_ax.plot(
            [], [], label="Dipole", marker="", linestyle="-", alpha=0.8
        )
        self.center_marker_r = refit_ax.axvline(
            self.mpms.scan_amplitude[0] / 2, color="C5"
        )

        self.viewer_idx = None

        # moment_ax
        moment_ax.axhline(0, color="0.9")
        moment_ax.axvline(0, color="0.9")
        self.data_line = self.mpms.plot_dat(
            ax=moment_ax, marker=".", picker=True, pickradius=5, label="Original"
        )
        (self.picked_data_line,) = moment_ax.plot([], [], marker="o", linestyle="")
        (self.refit_moment_line,) = moment_ax.plot([], [], marker=".", label="Refit")

        # fit_value_ax
        self.fit_value_line = self.mpms.plot_fit_value(
            ax=fit_value_ax,
            marker=".",
            picker=True,
            pickradius=5,
            label="Original fit value",
        )
        (self.picked_fit_value_line,) = fit_value_ax.plot(
            [], [], marker="o", linestyle=""
        )
        self.refit_r_square_line = self.mpms.plot_raw_fit_value(
            ax=fit_value_ax, marker=".", label="Refit R$^2$"
        )

        def on_pick(event):
            if event.artist not in [self.data_line, self.fit_value_line]:
                return
            try:
                idx = event.ind[0]
                self.scan_slider.set_val(idx)
            except IndexError:
                pass

        self.fig.canvas.mpl_connect("pick_event", on_pick)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_click_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("key_release_event", self.on_key_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)

        for ax in self.axes.flatten():
            toggle_legend(ax, loc="upper right")

    def select_scan_idx(self, idx):
        idx = int(idx)
        logger.info(f"Selecting scan {idx}")
        self.viewer_idx = idx
        ((original_ax, moment_ax), (refit_ax, fit_value_ax)) = self.axes
        position = self.mpms.position[idx]
        corrected_voltage = self.mpms.voltage[idx]
        background_voltage = self.mpms.background[idx]
        raw_voltage = self.mpms.raw_voltage[idx]
        raw_fit_voltage = self.mpms.raw_fit_voltage[idx]
        # original_ax
        self.raw_line.set_data(position, raw_voltage)
        self.fit_line.set_data(position, raw_fit_voltage)
        self.background_line.set_data(position, background_voltage)
        # refit_ax
        self.raw_line_corrected.set_data(position, corrected_voltage)
        if idx in self.mpms.fits:
            self.refit_line.set_data(position, self.mpms.fits[idx].best_fit)
        else:
            self.refit_line.set_data([], [])
        # moment_ax
        x, y = self.data_line.get_data()
        self.picked_data_line.set_data([x[idx]], [y[idx]])
        if self.inverse_susceptibility_mode:
            y_dependent = self.mpms.field / self.mpms.raw_moment
        else:
            y_dependent = self.mpms.raw_moment
        if self.mpms.field_dependent:
            self.refit_moment_line.set_data(self.mpms.field, y_dependent)
        else:
            self.refit_moment_line.set_data(self.mpms.temperature, y_dependent)
        # fit_value_ax
        xf, yf = self.fit_value_line.get_data()
        self.picked_fit_value_line.set_data([xf[idx]], [yf[idx]])
        if self.mpms.field_dependent:
            self.refit_r_square_line.set_data(self.mpms.field, self.mpms.raw_R2)
        else:
            self.refit_r_square_line.set_data(self.mpms.temperature, self.mpms.raw_R2)
        # title
        T = self.mpms.temperature[idx]
        H = self.mpms.field[idx]
        self.fig.suptitle(
            f"{self.mpms.root}.raw\nScan {idx}, T = {T:.1f} K, H = {H:.1f} Oe"
        )
        for ax in self.axes.flatten():
            rescale_visible(ax)
        self.fig.canvas.draw_idle()

    def refresh(self):
        self.select_scan_idx(self.viewer_idx)

    def on_click_press(self, event):
        self.last_event = event
        if not event.inaxes in self.axes[:, 0]:
            return
        # check for pan/zoom mode
        try:
            # qt
            if self.fig.canvas.cursor().shape().value != 0:
                return
        except AttributeError:
            # tk
            if self.toolbar.mode != "":
                return

        if self.add_dipole_mode:
            self.mpms.add_dipole_background(self.dipole_moment, self.dipole_center)
            self.add_dipole_mode = False
            self.original_dipole_line.set_data([], [])
            self.refit_dipole_line.set_data([], [])
        else:
            logger.info(f"Updating center line position to {event.xdata:.2f}")
            self.center_marker_o.set_xdata([event.xdata, event.xdata])
            self.center_marker_r.set_xdata([event.xdata, event.xdata])
            self.mouse_pressed = True
        self.refresh()

    def on_click_release(self, event):
        self.mouse_pressed = False

    def on_key_press(self, event):
        if event.key == "shift" and event.inaxes in [self.original_ax, self.refit_ax]:
            self.shift_pressed = True
            self.shift_x, self.shift_y = event.xdata, event.ydata

    def on_key_release(self, event):
        if event.key == "shift":
            self.shift_x = None
            self.shift_y = None
            self.shift_pressed = False

    def on_motion(self, event):
        if not event.inaxes in [self.original_ax, self.refit_ax]:
            return
        if self.add_dipole_mode:
            if self.shift_pressed:
                dy = event.ydata - self.shift_y
                self.dipole_moment += dy
                self.dipole_center = event.xdata
                self.shift_y = event.ydata
                self.shift_x = event.xdata
            self.dipole_center = event.xdata
            x = self.mpms.position
            y = dipole_response(x, self.dipole_moment, self.dipole_center) + event.ydata
            self.original_dipole_line.set_data(x, y)
            self.refit_dipole_line.set_data(x, y)
        elif self.mouse_pressed:
            self.center_marker_o.set_xdata([event.xdata, event.xdata])
            self.center_marker_r.set_xdata([event.xdata, event.xdata])
        self.fig.canvas.draw()

    def toggle_inverse_susceptibility_mode(self, event):
        if not self.inverse_susceptibility_mode:
            self.inverse_susceptibility_mode = True
            self.moment_ax.set_ylabel("1/$\\chi$ (Oe/emu)")
            self.data_line.set_ydata(self.mpms.field / self.mpms.moment)
        else:
            self.inverse_susceptibility_mode = False
            self.moment_ax.set_ylabel("Moment (emu)")
            self.data_line.set_ydata(self.mpms.moment)
        self.refresh()

    @property
    def center(self):
        return self.center_marker_o.get_xdata()[0]

    def add_dipole_background(self, event):
        if self.add_dipole_mode:
            # disable dipole mode, clear visible
            self.add_dipole_mode = False
            self.original_dipole_line.set_data([], [])
            self.refit_dipole_line.set_data([], [])
            return
        self.add_dipole_mode = True
        self.dipole_moment = np.ptp(self.mpms.voltage[self.viewer_idx]) / 2.4
        self.dipole_center = self.center
        x = self.mpms.position
        y = dipole_response(x, self.dipole_moment, self.dipole_center)
        self.original_dipole_line.set_data(x, y)
        self.refit_dipole_line.set_data(x, y)

    def fit_dipole_background(self, event):
        logger.info(
            f"Adding dipole background from Scan {self.viewer_idx} at {self.center:.2f}"
        )
        self.mpms.set_scan_background(self.viewer_idx, self.center)
        self.refresh()

    def undo_last_dipole(self, event):
        try:
            moment, center = self.mpms.background_dipoles.pop()
        except IndexError:
            return
        self.mpms.background -= dipole_response(self.mpms.position, moment, center)
        self.refresh()

    def clear_background(self, event):
        logger.info(f"Resetting background for {self.mpms.root}.raw")
        self.mpms.reset_background()
        self.refresh()

    def fit_single_scan(self, event):
        self.mpms.scan_fit(self.viewer_idx, self.center)
        self.refresh()

    def fit_all_scans(self, event):
        self.mpms.scan_fit_all(self.center)
        self.refresh()

    def export_refit(self, event):
        fname = self.mpms.export_refit()
        print(f"Refit exported to {fname}")


def rescale_visible(ax):
    ax.autoscale(enable=True, axis="y")
    ax.relim(visible_only=True)
    ax.autoscale_view(scalex=False, scaley=True)


def toggle_legend(ax, **kwargs):
    fig = ax.figure
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(**kwargs)
    legend_lines = legend.get_lines()
    for line in legend_lines:
        line.set_picker(5)

    def on_pick(event):
        if event.artist not in legend_lines:
            return
        legend_line = event.artist
        idx = legend_lines.index(legend_line)
        original_line = handles[idx]
        original_line.set_visible(not original_line.get_visible())
        legend_line.set_alpha(1.0 if original_line.get_visible() else 0.2)
        rescale_visible(ax)
        fig.canvas.draw()

    fig.canvas.mpl_connect("pick_event", on_pick)
