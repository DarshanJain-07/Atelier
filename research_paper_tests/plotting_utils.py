from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

CM_TO_INCH = 1 / 2.54

# Springer conference-style figure widths commonly target one column (~8.4 cm)
# or full width (~17.4 cm). Default to a compact single-column figure.
SINGLE_COLUMN_WIDTH_CM = 8.4
DOUBLE_COLUMN_WIDTH_CM = 17.4
MAX_FIGURE_HEIGHT_CM = 23.4
DEFAULT_ASPECT_RATIO = 0.72

PAPER_FIGSIZE = (
    SINGLE_COLUMN_WIDTH_CM * CM_TO_INCH,
    SINGLE_COLUMN_WIDTH_CM * DEFAULT_ASPECT_RATIO * CM_TO_INCH,
)
PAPER_DPI = 600

PAPER_PALETTE = {
    "primary": "#1F4E79",
    "secondary": "#C55A11",
    "light": "#B8BEC7",
    "positive": "#2F6B2F",
    "neutral": "#7A7A7A",
    "negative": "#A33D3D",
    "text": "#222222",
    "axis": "#4A4A4A",
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
    """Apply Springer-like defaults for research paper figures."""
    plt.rcParams.update(
        {
            # Springer guidance commonly prefers Helvetica/Arial-style sans serif.
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "regular",
            "axes.labelsize": 8,
            "axes.labelcolor": PAPER_PALETTE["text"],
            "text.color": PAPER_PALETTE["text"],
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 12,
            "figure.titlesize": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.65,
            "grid.color": PAPER_PALETTE["light"],
            "grid.linestyle": ":",
            "grid.linewidth": 0.5,
            "axes.edgecolor": PAPER_PALETTE["axis"],
            "axes.linewidth": 0.6,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": PAPER_DPI,
            "axes.prop_cycle": plt.cycler("color", PAPER_COLOR_SEQUENCE),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            "patch.edgecolor": "white",
            "patch.linewidth": 0.8,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.borderaxespad": 0.4,
            "xtick.color": PAPER_PALETTE["text"],
            "ytick.color": PAPER_PALETTE["text"],
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        },
    )


def style_axis(ax, title=None, xlabel=None, ylabel=None):
    """Set common title/axis labels and final axis polish."""
    if title:
        ax.set_title(title, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", colors=PAPER_PALETTE["text"])
    ax.grid(True, alpha=0.65, linestyle=":", color=PAPER_PALETTE["light"], linewidth=0.5)
    return ax


def setup_plot(title=None, xlabel=None, ylabel=None, figsize=PAPER_FIGSIZE):
    """Create a paper-styled figure and single axis."""
    fig, ax = plt.subplots(figsize=figsize)
    style_axis(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    return fig, ax


def place_legend_outside(ax, *, ncol=1, anchor=(0.5, 1.02), location="lower center"):
    """Place a compact legend outside the plotting area to avoid overlap."""
    return ax.legend(
        loc=location,
        bbox_to_anchor=anchor,
        ncol=ncol,
        frameon=False,
        borderaxespad=0.0,
    )


def _relocate_axis_legend(ax):
    """Move an axis legend above the plot so it cannot obscure data marks."""
    legend = ax.get_legend()
    if legend is None:
        return False

    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return False

    column_count = min(len(handles), 3)
    legend.remove()
    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=column_count,
        frameon=False,
        borderaxespad=0.0,
    )
    return True


def save_paper_figure(fig, path: str | Path):
    """Persist a figure using the paper-wide export settings."""
    import warnings
    path = Path(path)
    has_outside_legend = False
    for ax in fig.axes:
        has_outside_legend = _relocate_axis_legend(ax) or has_outside_legend

    width_in, height_in = fig.get_size_inches()
    if has_outside_legend:
        # Add canvas above the axes for title + legend rather than shrinking the plot.
        fig.set_size_inches(width_in, height_in * 1.28, forward=True)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    else:
        fig.set_size_inches(width_in, height_in, forward=True)
        fig.tight_layout()
    fig.savefig(path, dpi=PAPER_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def compose_panel_grid(
    panel_paths: list[str | Path],
    output_path: str | Path,
    *,
    title: str,
    columns: int,
    panel_size: tuple[float, float] = (4.2, 3.2),
    legend_labels: list[str] | None = None,
    legend_colors: list[str] | None = None,
):
    """Build a documentation-friendly overview image from styled panel files."""
    paths = [Path(path) for path in panel_paths]
    rows = int(np.ceil(len(paths) / columns))
    legend_rows = 1 if legend_labels and legend_colors else 0
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * panel_size[0], rows * panel_size[1] + legend_rows * 0.55),
    )
    axes_array = np.asarray(axes, dtype=object).reshape(rows, columns)

    for ax in axes_array.flat:
        ax.axis("off")

    for ax, path in zip(axes_array.flat, paths, strict=False):
        ax.imshow(plt.imread(path))
        ax.axis("off")

    fig.suptitle(title)
    if legend_rows:
        handles = [
            Line2D(
                [0],
                [0],
                color=color,
                marker="o",
                linestyle="-",
                linewidth=1.2,
                markersize=4,
                label=label,
            )
            for label, color in zip(legend_labels, legend_colors, strict=False)
        ]
        fig.legend(
            handles=handles,
            labels=legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.968),
            ncol=len(legend_labels),
            frameon=False,
            fontsize=12,
            handlelength=2.2,
            handletextpad=0.8,
            columnspacing=2.2,
            borderaxespad=0.0,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    else:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    save_paper_figure(fig, output_path)
    plt.close(fig)
