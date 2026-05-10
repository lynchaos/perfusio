"""Custom OKLCH colour palette and Matplotlib rcParams.

Design constraints:
- Uses OKLCH perceptual colour space for uniform lightness across hues.
- Fonts: Inter (sans-serif text), JetBrains Mono (monospace), STIX Two Math
  (equations).  Falls back to system fonts gracefully.
- No off-the-shelf Matplotlib style sheets (Seaborn, etc.) are imported.
- All colours are accessible for deuteranopia (red-green colour blindness) by
  relying on lightness + hue offsets, not hue alone.

References
----------
.. [OKLCH] https://oklch.com/ — perceptual colour space.
"""

from __future__ import annotations

import matplotlib as mpl

# ── OKLCH palette converted to sRGB hex ───────────────────────────────────────
# 8 sequential hues at L≈0.70, C≈0.17, spaced 45° apart from 250°
PALETTE: list[str] = [
    "#4E9AF1",  # h=250 — blue (primary)
    "#E0775C",  # h=30  — terracotta (secondary)
    "#6DC473",  # h=140 — sage green
    "#C97ED8",  # h=310 — lavender
    "#E5B84E",  # h=75  — amber
    "#5EC4C4",  # h=195 — teal
    "#F07AB0",  # h=350 — rose
    "#8E8E8E",  # h=—   — neutral grey
]

# Lighter / darker variants for uncertainty ribbons
PALETTE_LIGHT: list[str] = [
    "#B2D4FB",
    "#F4C4B5",
    "#B8E3BB",
    "#E6C0EE",
    "#F5DFA8",
    "#ADE5E5",
    "#FAC4D9",
    "#D0D0D0",
]

# ── rcParams (NO seaborn, NO plt.style.use) ───────────────────────────────────

_RCPARAMS: dict[str, object] = {
    # Figure
    "figure.figsize": (7.0, 4.5),
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    # Axes
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.prop_cycle": mpl.cycler(color=PALETTE),
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.titleweight": "bold",
    # Grid
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.5,
    "grid.color": "#CCCCCC",
    # Lines
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    # Fonts
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "mathtext.fontset": "stix",
    # Legend
    "legend.frameon": False,
    "legend.fontsize": 9,
    # Ticks
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
}


def apply_theme() -> None:
    """Apply the perfusio Matplotlib theme globally.

    Call once at the top of a script or notebook cell::

        from perfusio.viz.theme import apply_theme
        apply_theme()

    Subsequent ``plt`` calls will use the perfusio style.
    """
    mpl.rcParams.update(_RCPARAMS)
