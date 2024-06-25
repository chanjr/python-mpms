#!/usr/bin/env python3

import functools
import pathlib
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as FigureCanvas
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as NavigationToolbar
from matplotlib.figure import Figure

from .interactive import RawScanViewer
from .mpms import MPMS, plot_combined


@functools.cache
def cached_load(fname):
    return MPMS(fname)


class Window:
    windows = []

    def __init__(self, window=None, title=None):
        if window is None:
            self.window = tk.Toplevel()
        else:
            self.window = window
        if title is not None:
            self.window.title(title)
        self.window.geometry("1300x900")
        self.windows.append(self)
        self.loaded = False
        self.data = None
        self.create_menu()
        self.create_canvas()

    @classmethod
    def close_all_windows(cls):
        for window in cls.windows[::-1]:
            window.window.destroy()

    def bring_to_front(self):
        self.window.lift()

    def create_menu(self):
        self.menu_bar = tk.Menu(self.window)
        self.filemenu = tk.Menu(self.menu_bar, tearoff=0)
        self.filemenu.add_command(label="Open dat", command=self.open_dat_file)
        self.filemenu.add_command(label="Open raw", command=self.open_raw_file)
        self.filemenu.add_command(label="Exit", command=self.close_all_windows)
        self.menu_bar.add_cascade(label="File", menu=self.filemenu)
        self.window.config(menu=self.menu_bar)

    def create_canvas(self):
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure, master=self.window)
        self.toolbar = NavigationToolbar(self.canvas, self.window, pack_toolbar=True)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    @classmethod
    def new_window(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def plot_dat(self, data):
        self.data = data
        plot_combined(data, fig=self.figure, fit_curie_weiss=False)
        self.figure.tight_layout()
        self.canvas.draw()
        self.loaded = True

    def plot_raw(self, data):
        # check if already open and bring to front if so
        for window in self.windows:
            if window.data is data:
                print("Already open")
                window.bring_to_front()
                return
        self.viewer = RawScanViewer(data, fig=self.figure, toolbar=self.toolbar)
        self.data = data
        self.canvas.draw()
        self.loaded = True

    def open_dat_file(self, *args, **kwargs):
        filenames = tk.filedialog.askopenfilenames(
            filetypes=[("MPMS data files", ".dat")],
            multiple=True,
        )
        if not filenames:
            return
        data = [cached_load(fname[:-4]) for fname in filenames]
        title = str(pathlib.Path(filenames[0]).parent)
        if self.loaded:
            new_window = self.new_window(title=title)
            new_window.plot_dat(data)
        else:
            self.plot_dat(data)
            self.window.title(title)

    def open_raw_file(self, *args, **kwargs):
        filenames = tk.filedialog.askopenfilenames(
            filetypes=[("MPMS raw files", ".raw")],
            multiple=True,
        )
        if not filenames:
            return
        for fname in filenames:
            p = pathlib.Path(fname)
            data = cached_load(p.parent / p.stem)
            if not self.loaded:
                self.plot_raw(data)
                self.window.title(fname)
            else:
                if any(window.data is data for window in self.windows):
                    continue
                new_window = self.new_window(title=fname)
                new_window.plot_raw(data)


def main():
    root = tk.Tk()
    root.geometry("1300x900")
    window = Window(root)
    root.mainloop()


if __name__ == "__main__":
    main()
