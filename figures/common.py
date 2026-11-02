"""Paths and style shared by the figure scripts."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = HERE
W = 165 / 25.4                     # full text width [in]
CL = ["small", "medium", "large"]
N = {"small": 8, "medium": 8, "large": 4}


def res(name):
    return os.path.join(RES, name)


def out(name):
    return os.path.join(OUT, name)


def style():
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8})
