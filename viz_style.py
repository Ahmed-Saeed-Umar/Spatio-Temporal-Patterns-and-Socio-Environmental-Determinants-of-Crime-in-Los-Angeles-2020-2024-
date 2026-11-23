"""Shared chart style so every figure in the notebook and the presentation looks the same.

Palette: a colour-blind-safe categorical order (blue, orange, aqua, violet, ...) and a single-hue
blue ramp for magnitude (heatmaps). Colour is used to carry meaning, never decoration:
blue = the main series, orange = the thing we want you to look at (e.g. arrests).
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BLUE, ORANGE, AQUA, VIOLET, YELLOW, MAGENTA = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6",
                                               "#eda100", "#e87ba4")
CATEGORICAL = [BLUE, ORANGE, AQUA, VIOLET, YELLOW, MAGENTA, "#008300", "#e34948"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f2f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "figure.facecolor": "white",
        "axes.facecolor": "white", "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "font.size": 10.5, "legend.frameon": False, "axes.prop_cycle":
            mpl.cycler(color=CATEGORICAL),
    })


def _thousands(fig):
    """Show 20000 as 20,000 on plain numeric axes (date and category axes are left alone)."""
    fmt = mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    for ax in fig.axes:
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            if isinstance(axis.get_major_formatter(), mpl.ticker.ScalarFormatter) \
                    and max(abs(lim[0]), abs(lim[1])) >= 10_000 and axis.get_scale() == "linear" \
                    and ax.get_label() != "<colorbar>":
                axis.set_major_formatter(fmt)


def savefig(fig, name):
    """Save into outputs/figures and return the path (figures are reused in the slide deck)."""
    _thousands(fig)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    return path
