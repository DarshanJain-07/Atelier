from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

PAPER_FIGSIZE = (10, 8)
PAPER_DPI = 220

# One restrained, colorblind-friendly palette for every generated paper figure.
PAPER_PALETTE = {
    "primary": "#0072B2",       # blue
    "secondary": "#009E73",     # green
    "tertiary": "#CC79A7",      # purple
    "accent": "#E69F00",        # orange
    "negative": "#D55E00",      # vermillion
    "positive": "#009E73",      # green
    "neutral": "#7A7F87",       # graphite
    "baseline": "#7A7F87",
    "comparison": "#0072B2",
    "treatment": "#009E73",
    "threshold": "#D55E00",
    "reference": "#B7BBC2",
    "dark": "#222222",
    "light": "#F7F8FA",
}

CATEGORICAL_COLORS = [
    PAPER_PALETTE["primary"],
    PAPER_PALETTE["secondary"],
    PAPER_PALETTE["accent"],
    PAPER_PALETTE["negative"],
    PAPER_PALETTE["tertiary"],
    "#56B4E9",
    "#F0E442",
    "#8A5A44",
    "#6C757D",
    "#111111",
]

SENTIMENT_COLORS = [
    PAPER_PALETTE["negative"],
    PAPER_PALETTE["neutral"],
    PAPER_PALETTE["positive"],
]

COMPARISON_COLORS = [
    PAPER_PALETTE["baseline"],
    PAPER_PALETTE["treatment"],
]

PAPER_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "paper_diverging",
    [PAPER_PALETTE["negative"], "white", PAPER_PALETTE["primary"]],
)


def apply_paper_style():
    """Apply a consistent global style for research paper figures."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.titleweight": "semibold",
            "axes.labelsize": 12,
            "axes.labelcolor": PAPER_PALETTE["dark"],
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 18,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.18,
            "grid.color": PAPER_PALETTE["reference"],
            "grid.linestyle": "--",
            "axes.edgecolor": PAPER_PALETTE["reference"],
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": PAPER_DPI,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 2,
            "lines.markersize": 7,
            "patch.edgecolor": "white",
            "patch.linewidth": 0.8,
        }
    )


def style_axis(ax, title=None, xlabel=None, ylabel=None):
    """Set common title/axis labels and final axis polish."""
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", colors=PAPER_PALETTE["dark"])
    ax.grid(True, alpha=0.18, linestyle="--", color=PAPER_PALETTE["reference"])
    return ax


def setup_plot(title=None, xlabel=None, ylabel=None, figsize=PAPER_FIGSIZE):
    """Create a paper-styled figure and single axis."""
    fig, ax = plt.subplots(figsize=figsize)
    style_axis(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    return fig, ax


def save_paper_figure(fig, path: str | Path):
    """Persist a figure using the paper-wide export settings."""
    fig.savefig(path, dpi=PAPER_DPI, bbox_inches="tight")


def compose_panel_grid(
    panel_paths: list[str | Path],
    output_path: str | Path,
    *,
    title: str,
    columns: int,
    panel_size: tuple[float, float] = (4.2, 3.2),
):
    """Build a documentation-friendly overview image from styled panel files."""
    paths = [Path(path) for path in panel_paths]
    rows = int(np.ceil(len(paths) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * panel_size[0], rows * panel_size[1]),
    )
    axes_array = np.asarray(axes, dtype=object).reshape(rows, columns)

    for ax in axes_array.flat:
        ax.axis("off")

    for ax, path in zip(axes_array.flat, paths, strict=False):
        ax.imshow(plt.imread(path))
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    save_paper_figure(fig, output_path)
    plt.close(fig)
