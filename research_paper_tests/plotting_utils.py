from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

PAPER_FIGSIZE = (10, 8)
PAPER_DPI = 220

PAPER_PALETTE = {
    "primary": "#1565C0",
    "secondary": "#EF6C00",
    "light": "#6A1B9A",
    "positive": "#2E7D32",
    "neutral": "#757575",
    "negative": "#C62828",
}

PAPER_COLOR_SEQUENCE = [
    PAPER_PALETTE["primary"],
    PAPER_PALETTE["secondary"],
    PAPER_PALETTE["positive"],
    PAPER_PALETTE["neutral"],
    PAPER_PALETTE["negative"],
]

CATEGORICAL_COLORS = PAPER_COLOR_SEQUENCE

SENTIMENT_COLORS = [
    PAPER_PALETTE["negative"],
    PAPER_PALETTE["neutral"],
    PAPER_PALETTE["positive"],
]

COMPARISON_COLORS = [
    PAPER_PALETTE["primary"],
    PAPER_PALETTE["secondary"],
]

PAPER_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "paper_diverging",
    [PAPER_PALETTE["negative"], PAPER_PALETTE["light"], PAPER_PALETTE["positive"]],
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
            "axes.labelcolor": PAPER_PALETTE["primary"],
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 18,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.18,
            "grid.color": PAPER_PALETTE["light"],
            "grid.linestyle": "--",
            "axes.edgecolor": PAPER_PALETTE["light"],
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": PAPER_DPI,
            "axes.prop_cycle": plt.cycler("color", PAPER_COLOR_SEQUENCE),
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
    ax.tick_params(axis="both", colors=PAPER_PALETTE["primary"])
    ax.grid(True, alpha=0.18, linestyle="--", color=PAPER_PALETTE["light"])
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
