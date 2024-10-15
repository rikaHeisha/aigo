from typing import Literal

import matplotlib
import numpy as np
from matplotlib import pyplot as plt


def intensity_to_rgb(intensity: np.ndarray, color_map: str = "plasma") -> np.ndarray:
    if not np.all((intensity >= 0.0) & (intensity <= 1.0)):
        raise ValueError("Intensity needs to be between 0 and 1")

    color_map = matplotlib.colormaps[color_map]
    return color_map(intensity)[:, :3]  # discard alpha value and only return rgb


def draw_histogram(data, fig_path: str | None, bins: int = 100):
    hist, bins = np.histogram(data, bins=bins)
    hist = hist / hist.sum()

    width = 1.0 * (bins[1] - bins[0])

    assert bins[:-1].shape == hist.shape

    fig, ax = plt.subplots()
    ax.bar(bins[:-1], hist, align="edge", width=width)

    if fig_path:
        fig.savefig(fig_path)
    else:
        plt.show()

    plt.close(fig)


def draw_bar(
    xs,
    ys,
    fig_path: str | None,
    width,
    align: Literal["edge", "center"] = "edge",
    title: str | None = None,
):
    fig, ax = plt.subplots()
    ax.bar(xs, ys, align=align, width=width)

    if title:
        ax.set_title(title)

    if fig_path:
        fig.savefig(fig_path)
    else:
        plt.show()

    plt.close(fig)


def draw_pmf(pmf, fig_path: str | None):
    assert pmf.ndim == 1 and np.isclose(pmf.sum(), 1.0)
    bins = np.linspace(0, pmf.shape[0], num=pmf.shape[0] + 1)
    width = 1.0 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots()
    ax.bar(center, pmf, align="center", width=width)

    if fig_path:
        fig.savefig(fig_path)
    else:
        plt.show()

    plt.close(fig)
