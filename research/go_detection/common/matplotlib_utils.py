import matplotlib
import numpy as np
from matplotlib import pyplot as plt


def intensity_to_rgb(intensity: np.ndarray, color_map: str = "plasma") -> np.ndarray:
    if not np.all((intensity >= 0.0) & (intensity <= 1.0)):
        raise ValueError("Intensity needs to be between 0 and 1")

    color_map = matplotlib.colormaps[color_map]
    return color_map(intensity)[:, :3]  # discard alpha value and only return rgb


def draw_histogram(data, fig_path: str, bins: int = 100):
    hist, bins = np.histogram(data, bins=bins)
    hist = hist / hist.sum()

    width = 1.0 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots()
    ax.bar(center, hist, align="center", width=width)
    fig.savefig(fig_path)
