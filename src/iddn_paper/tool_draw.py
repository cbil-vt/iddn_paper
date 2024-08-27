"""
Draw line plots for iDDN paper
"""

import numpy as np
import matplotlib.pyplot as plt

from cycler import cycler

default_cycler = cycler(color=["#1F77B4", "#FF7F0E", "#3A3A3A", "#5CCEB3"])
plt.rc("axes", prop_cycle=default_cycler)


def draw_lines_rho1_rho2_v1(
    res_lst,
    title="Common network",
    xlim=None,
    ylim=None,
    marker_size=0.2,
    line_width=0.2,
    idx1=2,  # recall
    idx2=4,  # precision
    colors=("#1F77B4", "#FF7F0E", "#3A3A3A", "#5CCEB3"),
    ax=None,
    xlabel=None,
    ylabel=None,
):
    n_cond = len(res_lst)
    if type(line_width) is float:
        line_width = np.zeros(n_cond) + line_width
    if ax is None:
        fig, ax = plt.subplots()
    for ii, res_now in enumerate(res_lst):
        rho2_rg = range(res_now.shape[1])
        for i in rho2_rg:
            ax.plot(
                res_now[:, i, idx1],
                res_now[:, i, idx2],
                "-o",
                color=colors[ii],
                markersize=marker_size,
                linewidth=line_width[ii],
            )

    if xlim is None:
        ax.set_xlim([-0.05, 1.05])
    else:
        ax.set_xlim(xlim)
    if ylim is None:
        ax.set_ylim([-0.05, 1.05])
    else:
        ax.set_ylim(ylim)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title)


def draw_1d_lines(
    curve_lst,
    xrange=None,
    title="F1",
    xlim=None,
    ylim=None,
    ax=None,
    xlabel=None,
    ylabel=None,
):
    ymax = np.max(np.array(curve_lst))
    if ylim is not None:
        ylim[1] = ymax + 0.05

    if ax is None:
        fig, ax = plt.subplots()
    for ii, x in enumerate(curve_lst):
        ax.plot(xrange, x)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title)
