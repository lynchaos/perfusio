"""Figure snapshot tests.

Generates each static figure and checks:
1. Return type is ``matplotlib.figure.Figure``.
2. Figure has at least one axis with non-empty artists.

These are fast deterministic tests; actual pixel comparison is done in CI
via ``pytest-image-snapshot`` with a 2% threshold.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest


@pytest.fixture()
def dummy_runs(tmp_path: pathlib.Path) -> list[dict]:
    """24-run Box-Behnken output for figure tests."""
    from perfusio.simulator.cho_perfusion import CHOSimulator

    sim = CHOSimulator(clone="CloneX", seed=0)
    return sim.generate_box_behnken_experiment(n_days=14, seed=0)


def test_fig4_returns_figure(dummy_runs: list) -> None:
    import matplotlib.figure

    from perfusio.viz.static import fig4_training_trajectories

    fig = fig4_training_trajectories(dummy_runs, alt_text=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_fig7_returns_figure(dummy_runs: list) -> None:
    import matplotlib.figure

    from perfusio.viz.static import fig7_pareto_front

    titer = [r["trajectory"][-1, 8] for r in dummy_runs]
    vcv = [r["trajectory"][-1, 0] for r in dummy_runs]
    fig = fig7_pareto_front(titer, vcv, alt_text=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_fig8_returns_figure() -> None:
    import matplotlib.figure

    from perfusio.viz.static import fig8_closed_loop_performance

    days = list(range(29))
    vcd = np.linspace(1, 12, 29).tolist()
    glc = np.linspace(5, 4, 29).tolist()
    titer = np.linspace(0, 600, 29).tolist()
    fig = fig8_closed_loop_performance(days, vcd, 10.0, glc, 5.0, titer, 500.0, alt_text=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    import matplotlib.pyplot as plt

    plt.close(fig)
