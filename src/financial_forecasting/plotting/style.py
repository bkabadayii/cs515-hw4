"""Shared matplotlib style for all project figures.

Every plotting script must call :func:`apply_style` before creating any
figure.  This guarantees visual consistency across all publication-ready
outputs regardless of the local matplotlib rc configuration.

Design goals
------------
* Dark background with high-contrast foreground for screen and PDF.
* Inter / DejaVu Sans fallback typography.
* Consistent DPI, font sizes, line widths, and colour palette.
* No seaborn dependency -- plain matplotlib only.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
# Ordered so that the most visually distinct colours appear first.
PALETTE: list[str] = [
    "#1F77B4",  # professional blue  -- AAPL
    "#2CA02C",  # professional green -- MSFT
    "#FF7F0E",  # professional orange -- JPM
]

TICKER_COLOURS: dict[str, str] = {
    "AAPL": PALETTE[0],
    "MSFT": PALETTE[1],
    "JPM":  PALETTE[2],
}

# Split region colours (semi-transparent fills)
SPLIT_COLOURS: dict[str, str] = {
    "train":      "#1F77B4",  # blue
    "validation": "#9467BD",  # purple
    "test":       "#D62728",  # red
}
SPLIT_ALPHA: float = 0.08

# Background / foreground
BG_COLOUR   = "#FFFFFF"   # pure white
FG_COLOUR   = "#1A1A1A"   # near-black
GRID_COLOUR = "#E0E0E0"   # light grey grid

# ---------------------------------------------------------------------------
# Style application
# ---------------------------------------------------------------------------

def apply_style() -> None:
    """Apply the project-wide matplotlib style.

    Call once at the top of every plotting script, before any figure is
    created.  Modifies ``matplotlib.rcParams`` globally for the process.
    """
    mpl.rcParams.update(
        {
            # Background
            "figure.facecolor":  BG_COLOUR,
            "axes.facecolor":    BG_COLOUR,
            "savefig.facecolor": BG_COLOUR,
            # Text and tick colours
            "text.color":        FG_COLOUR,
            "axes.labelcolor":   FG_COLOUR,
            "xtick.color":       FG_COLOUR,
            "ytick.color":       FG_COLOUR,
            # Spine and grid
            "axes.edgecolor":    GRID_COLOUR,
            "grid.color":        GRID_COLOUR,
            "grid.linewidth":    0.6,
            "axes.grid":         True,
            "grid.alpha":        0.5,
            # Typography
            "font.family":       "DejaVu Sans",
            "font.size":         11,
            "axes.titlesize":    13,
            "axes.labelsize":    11,
            "xtick.labelsize":   9,
            "ytick.labelsize":   9,
            "legend.fontsize":   10,
            "figure.titlesize":  14,
            # Lines
            "lines.linewidth":   1.6,
            "axes.linewidth":    0.8,
            # Output quality
            "figure.dpi":        120,
            "savefig.dpi":       200,
            "savefig.bbox":      "tight",
            "savefig.pad_inches": 0.15,
        }
    )


def save_figure(fig: Figure, stem: str) -> tuple[str, str]:
    """Save *fig* as PNG and PDF under *stem* (without extension).

    Parameters
    ----------
    fig:
        The matplotlib Figure to save.
    stem:
        Full path without extension, e.g. ``"figures/dataset/fig01_timeline"``.

    Returns
    -------
    tuple[str, str]
        Absolute paths to the PNG and PDF files.
    """
    png_path = f"{stem}.png"
    pdf_path = f"{stem}.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path
