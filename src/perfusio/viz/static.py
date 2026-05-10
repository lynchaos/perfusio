"""Static Matplotlib reproductions of Gadiyar et al. (2026) Figures 4, 6, 7, 8.

Each function returns a :class:`matplotlib.figure.Figure` and includes an
``alt_text`` parameter to generate an accessibility string for the figure.

Figure map
----------
- Fig. 4: Training experiment trajectories (Box-Behnken 24 runs, Clone X + Y).
- Fig. 6: 3-step-ahead hybrid model predictions vs. observations (all species).
- Fig. 7: Multi-objective Pareto front (titer vs. VCV at day 14).
- Fig. 8: Closed-loop control performance — VCD, Glc, Titer over 28 days.

References
----------
.. [Gadiyar2026] Gadiyar et al. (2026), Figures 4, 6, 7, 8.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _require_mpl() -> Any:
    import matplotlib.pyplot as plt

    return plt


# ── Figure 4 ──────────────────────────────────────────────────────────────────


def fig4_training_trajectories(
    runs: list[dict[str, Any]],
    species: str = "VCD",
    clone_labels: list[str] | None = None,
    alt_text: bool = False,
) -> Any:
    """Reproduce Gadiyar Fig. 4 — training experiment trajectories.

    Parameters
    ----------
    runs:
        List of run dicts as returned by
        :meth:`~perfusio.simulator.cho_perfusion.CHOSimulator.generate_box_behnken_experiment`.
    species:
        Species to plot on the y-axis.  Default ``"VCD"``.
    clone_labels:
        Optional run-level labels for the legend.
    alt_text:
        If True, prints an accessibility alt-text description.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from perfusio.viz.theme import PALETTE, apply_theme

    apply_theme()
    plt = _require_mpl()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    species_idx = {
        "VCD": 0,
        "Via": 1,
        "Glc": 2,
        "Gln": 3,
        "Glu": 4,
        "Lac": 5,
        "Amm": 6,
        "Pyr": 7,
        "Titer": 8,
    }
    k = species_idx.get(species, 0)

    for ax_idx, ax in enumerate(axes):
        # Split runs by clone label: left panel = CloneX (or first half), right = CloneY
        if clone_labels is not None:
            subset = [r for r, lbl in zip(runs, clone_labels) if (ax_idx == 0) == ("X" in lbl or "x" in lbl)]
        else:
            # No labels provided: split evenly by index
            mid = len(runs) // 2
            subset = runs[:mid] if ax_idx == 0 else runs[mid:]
        for i, run in enumerate(subset):
            traj = run["trajectory"]  # (n_days+1, n_species)
            days = np.arange(traj.shape[0])
            ax.plot(
                days,
                traj[:, k],
                color=PALETTE[i % len(PALETTE)],
                lw=1.2,
                alpha=0.8,
                label=f"Run {run['run_id']}" if i < 5 else None,
            )
        ax.set_xlabel("Culture day")
        ax.set_ylabel(species)
        ax.set_title("Clone X runs" if ax_idx == 0 else "Clone Y runs")
    axes[0].legend(ncol=2, fontsize=7)

    fig.suptitle(
        f"Fig. 4 — Box-Behnken training trajectories ({species})",
        fontweight="bold",
    )
    fig.tight_layout()

    if alt_text:
        print(
            f"Alt text: Line chart of {len(runs)} perfusion bioreactor runs "
            f"showing {species} over culture days. Each coloured line is one "
            "Box-Behnken run. Clone X (left panel) and Clone Y (right panel) "
            "are shown separately."
        )
    return fig


# ── Figure 6 ──────────────────────────────────────────────────────────────────


def fig6_model_predictions(
    true_traj: Any,
    pred_mean: Any,
    pred_q10: Any,
    pred_q90: Any,
    species_names: list[str],
    horizon: int = 3,
    alt_text: bool = False,
) -> Any:
    """Reproduce Gadiyar Fig. 6 — 3-step hybrid model predictions.

    Parameters
    ----------
    true_traj:
        Tensor/array, shape ``(T, n_species)``.
    pred_mean, pred_q10, pred_q90:
        Predictions, shape ``(T, n_species)``.
    species_names:
        List of species labels.
    horizon:
        Prediction horizon (drawn as shaded region).
    alt_text:
        If True, prints an accessibility alt-text description.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from perfusio.viz.theme import PALETTE, PALETTE_LIGHT, apply_theme

    apply_theme()
    plt = _require_mpl()

    n_spc = len(species_names)
    ncols = min(3, n_spc)
    nrows = (n_spc + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes_flat = np.array(axes).flatten()

    T = true_traj.shape[0] if hasattr(true_traj, "shape") else len(true_traj)
    days = np.arange(T)

    def _np(x: Any) -> np.ndarray:
        if hasattr(x, "numpy"):
            return x.detach().numpy()
        return np.asarray(x)

    for i, name in enumerate(species_names):
        ax = axes_flat[i]
        ax.plot(days, _np(true_traj)[:, i], "o", color=PALETTE[0], ms=4, label="Observed")
        ax.plot(days, _np(pred_mean)[:, i], "-", color=PALETTE[1], lw=1.5, label="Mean")
        ax.fill_between(
            days,
            _np(pred_q10)[:, i],
            _np(pred_q90)[:, i],
            color=PALETTE_LIGHT[1],
            alpha=0.5,
            label="80% PI",
        )
        ax.set_title(name)
        ax.set_xlabel("Day")
        if i == 0:
            ax.legend(fontsize=7)

    # Hide unused axes
    for j in range(n_spc, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Fig. 6 — Hybrid model 3-step predictions vs. observations", fontweight="bold")
    fig.tight_layout()

    if alt_text:
        print(
            f"Alt text: Grid of {n_spc} subplots showing hybrid model "
            "predictions (orange line with 80% interval shading) vs. "
            "observed daily offline samples (blue circles) for each "
            "measured species."
        )
    return fig


# ── Figure 7 ──────────────────────────────────────────────────────────────────


def fig7_pareto_front(
    pareto_titer: Any,
    pareto_vcv: Any,
    feasible_titer: Any | None = None,
    feasible_vcv: Any | None = None,
    alt_text: bool = False,
) -> Any:
    """Reproduce Gadiyar Fig. 7 — Pareto front: titer vs. VCV.

    Parameters
    ----------
    pareto_titer:
        Titer values on the Pareto front.
    pareto_vcv:
        VCV (or VCD) values on the Pareto front.
    feasible_titer, feasible_vcv:
        All feasible evaluated points (background scatter).
    alt_text:
        If True, prints an accessibility alt-text description.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from perfusio.viz.theme import PALETTE, PALETTE_LIGHT, apply_theme

    apply_theme()
    plt = _require_mpl()

    fig, ax = plt.subplots(figsize=(6, 5))

    if feasible_titer is not None and feasible_vcv is not None:
        ax.scatter(
            np.asarray(feasible_titer),
            np.asarray(feasible_vcv),
            color=PALETTE_LIGHT[0],
            s=20,
            alpha=0.5,
            label="Feasible",
            zorder=2,
        )

    order = np.argsort(np.asarray(pareto_titer))
    ax.plot(
        np.asarray(pareto_titer)[order],
        np.asarray(pareto_vcv)[order],
        "o-",
        color=PALETTE[0],
        lw=2,
        ms=7,
        label="Pareto front",
        zorder=3,
    )

    ax.set_xlabel("Titer (mg/L)")
    ax.set_ylabel("VCV (10⁶ cells/mL · viability)")
    ax.set_title("Fig. 7 — Multi-objective Pareto front (Day 14)", fontweight="bold")
    ax.legend()
    fig.tight_layout()

    if alt_text:
        print(
            "Alt text: Scatter plot showing the Pareto-optimal trade-off "
            "between titer (x-axis, mg/L) and viable cell volume (y-axis) "
            "at culture day 14. Blue circles connected by a line form the "
            "Pareto front; grey dots show all evaluated candidate points."
        )
    return fig


# ── Figure 8 ──────────────────────────────────────────────────────────────────


def fig8_closed_loop_performance(
    days: Any,
    vcd: Any,
    vcd_target: float,
    glc: Any,
    glc_target: float,
    titer: Any,
    titer_target: float,
    alt_text: bool = False,
) -> Any:
    """Reproduce Gadiyar Fig. 8 — closed-loop control performance.

    Parameters
    ----------
    days:
        Culture days (x-axis).
    vcd, glc, titer:
        Measured trajectories.
    vcd_target, glc_target, titer_target:
        Setpoint targets drawn as horizontal dashed lines.
    alt_text:
        If True, prints an accessibility alt-text description.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from perfusio.viz.theme import PALETTE, apply_theme

    apply_theme()
    plt = _require_mpl()

    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    specs = [
        (vcd, vcd_target, "VCD (10⁶ cells/mL)", PALETTE[0]),
        (glc, glc_target, "Glucose (g/L)", PALETTE[1]),
        (titer, titer_target, "Titer (mg/L)", PALETTE[2]),
    ]

    for ax, (traj, target, ylabel, color) in zip(axes, specs, strict=False):
        ax.plot(np.asarray(days), np.asarray(traj), "o-", color=color, lw=1.8, ms=5)
        ax.axhline(target, ls="--", lw=1.2, color=color, alpha=0.6, label=f"Target {target}")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Culture day")
    fig.suptitle("Fig. 8 — Closed-loop self-driving control performance", fontweight="bold")
    fig.tight_layout()

    if alt_text:
        print(
            "Alt text: Three-panel time series plot over 28 culture days. "
            "Top panel: VCD (blue) converging to the target (dashed). "
            "Middle panel: Glucose (orange). Bottom: Titer (green). "
            "Dashed horizontal lines indicate BED-selected setpoint targets."
        )
    return fig
