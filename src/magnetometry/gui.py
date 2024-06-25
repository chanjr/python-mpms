#!/usr/bin/env python3

import functools
import logging
import re
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.terminal.embed import InteractiveShellEmbed
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox

from .interactive import RawScanViewer
from .mpms import MPMS, plot_combined

logger = logging.getLogger(__name__)
data = []


@functools.lru_cache(maxsize=None)
def cached_load(fname):
    logger.info(f"Loaded {fname}")
    mpms = MPMS(fname)
    data.append(mpms)
    return mpms


class MainWindow(QtWidgets.QMainWindow):
    windows = []

    def __init__(self):  # , shell):
        super().__init__()
        MainWindow.windows.append(self)
        # self.shell = shell
        self.loaded = False
        self.data = None
        self.resize(1300, 900)
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self.layout = QtWidgets.QVBoxLayout(self._main)
        self.create_menu()
        self.create_canvas()
        self.create_status_bar()

    @classmethod
    def close_all_windows(cls):
        for window in cls.windows[:]:
            window.close()

    def closeEvent(self, event):
        MainWindow.windows.remove(self)
        event.accept()

    def bring_to_front(self):
        self.setWindowState(
            self.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive
        )
        self.activateWindow()

    def create_menu(self):
        self.menu_bar = QtWidgets.QMenuBar(self)
        self.filemenu = self.menu_bar.addMenu("File")
        self.filemenu_open = self.filemenu.addMenu("Open")
        self.filemenu_open_dat = self.filemenu_open.addAction(".dat")
        self.filemenu_open_dat.triggered.connect(self.open_dat_file)
        self.filemenu_open_raw = self.filemenu_open.addAction(".raw")
        self.filemenu_open_raw.triggered.connect(self.open_raw_file)
        self.filemenu.addSeparator()
        self.filemenu_exit = self.filemenu.addAction("Exit")
        self.filemenu_exit.triggered.connect(self.close_all)
        self.helpmenu = self.menu_bar.addMenu("Help")
        self.helpmenu.addAction("About")
        self.setMenuBar(self.menu_bar)

    def create_canvas(self):
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        self.addToolBar(NavigationToolbar(self.canvas, self))
        self.layout.addWidget(self.canvas)
        self.canvas.setFocusPolicy(Qt.ClickFocus)
        self.canvas.setFocus()

    def create_status_bar(self):
        self.status_bar = QtWidgets.QStatusBar()
        self.status_bar.showMessage("Status...")
        self.setStatusBar(self.status_bar)

    def new_window(self):
        # return MainWindow(self.shell)
        return MainWindow()

    def close_all(self):
        msg_box = QtWidgets.QMessageBox()
        msg_box.setText("Close all windows?")
        msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg_box.setDefaultButton(QMessageBox.Cancel)
        ret = msg_box.exec_()
        if ret == QMessageBox.Ok:
            MainWindow.close_all_windows()

    def plot_dat(self, data):
        self.data = data
        plot_combined(data, fig=self.figure, fit_curie_weiss=False)
        self.figure.tight_layout()
        self.canvas.draw()
        self.loaded = True

    def plot_raw(self, data):
        # check if already open and bring to front if so
        for window in MainWindow.windows:
            if window.data is data:
                print("Already open")
                window.bring_to_front()
                return
        self.viewer = RawScanViewer(data, fig=self.figure)
        self.data = data
        self.canvas.draw()
        self.loaded = True

    def open_dat_file(self, *args, **kwargs):
        filenames, filetype = QtWidgets.QFileDialog.getOpenFileNames(
            filter="MPMS data files (*.dat)"
        )
        data = [cached_load(fname[:-4]) for fname in filenames]
        if self.loaded:
            new_window = self.new_window()
            new_window.plot_dat(data)
            new_window.show()
        else:
            self.plot_dat(data)

    def open_raw_file(self, *args, **kwargs):
        filenames, filetype = QtWidgets.QFileDialog.getOpenFileNames(
            filter="MPMS raw files (*.raw)"
        )
        print(filenames)
        data = [cached_load(fname[:-4]) for fname in filenames]
        if data and not self.loaded:
            d = data.pop()
            self.plot_raw(d)
        for d in data:
            # check if already open and bring to front if so
            for window in MainWindow.windows:
                if window.data is d:
                    print("Already open")
                    window.bring_to_front()
                    break
            else:
                new_window = self.new_window()
                new_window.plot_raw(d)
                new_window.show()


def main():
    shell = InteractiveShellEmbed.instance()
    shell.enable_gui("qt6")
    app = QtWidgets.QApplication([])
    # window = MainWindow(shell)
    window = MainWindow()
    window.show()
    shell()


if __name__ == "__main__":
    main()
