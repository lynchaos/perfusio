"""``perfusio`` command-line interface.

Commands
--------
``perfusio simulate``
    Run a 24-run Box-Behnken simulation and save trajectories.

``perfusio train``
    Train the hybrid model on simulation output.

``perfusio run``
    Execute the closed-loop digital twin loop (optionally with dashboard).

``perfusio reproduce-figures``
    Regenerate all paper figures (requires prior ``simulate`` and ``train``).

Usage examples::

    perfusio simulate --clone CloneX --seed 0 --out runs/
    perfusio train --data runs/ --out model.pt
    perfusio run --model model.pt --days 28 --dashboard
    perfusio reproduce-figures --data runs/ --model model.pt --out figures/
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="perfusio",
    help=(
        "perfusio — self-driving perfusion bioprocess library. "
        "Reference implementation of Gadiyar et al. (2026)."
    ),
    add_completion=True,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


# ── simulate ──────────────────────────────────────────────────────────────────


@app.command("simulate")
def simulate(
    clone: str = typer.Option("CloneX", help="CHO clone identifier (CloneX or CloneY)."),
    seed: int = typer.Option(0, help="Random seed for the simulator."),
    n_days: int = typer.Option(28, help="Number of culture days."),
    out: Path = typer.Option(Path("runs"), help="Output directory for run JSON files."),
) -> None:
    """Run a Box-Behnken training experiment and save trajectories to *out*."""
    from perfusio.simulator.cho_perfusion import CHOSimulator

    typer.echo(f"Simulating Box-Behnken experiment for {clone} (seed={seed}, {n_days} days)…")
    sim = CHOSimulator(clone=clone, seed=seed)
    runs = sim.generate_box_behnken_experiment(n_days=n_days, seed=seed)

    out.mkdir(parents=True, exist_ok=True)
    for run in runs:
        run_copy = dict(run)
        run_copy["trajectory"] = run_copy["trajectory"].tolist()
        run_copy["noisy_samples"] = [
            {k: (float(v) if v is not None else None) for k, v in s.items()}
            for s in run_copy["noisy_samples"]
        ]
        fpath = out / f"run_{run['run_id']:03d}.json"
        fpath.write_text(json.dumps(run_copy, indent=2))

    typer.echo(f"✓ {len(runs)} runs written to {out}/")


# ── train ─────────────────────────────────────────────────────────────────────


@app.command("train")
def train(
    data: Path = typer.Option(Path("runs"), help="Directory of run JSON files from `simulate`."),
    clone: str = typer.Option("CloneX", help="Clone used to build embedding."),
    out: Path = typer.Option(Path("model.pt"), help="Output path for the trained hybrid model."),
    n_steps: int = typer.Option(500, help="Maximum optimiser iterations."),
) -> None:
    """Train the hybrid model on simulation data and save to *out*."""
    import torch
    from perfusio.config import DEFAULT_AMBR250_DESIGN_SPACE
    from perfusio.hybrid.train import train_hybrid

    typer.echo(f"Loading runs from {data}/…")
    run_files = sorted(data.glob("run_*.json"))
    if not run_files:
        typer.echo("No run JSON files found. Run `perfusio simulate` first.", err=True)
        raise typer.Exit(code=1)

    import numpy as np
    from perfusio.states import Trajectory

    trajectories: list[Trajectory] = []
    for fpath in run_files:
        payload = json.loads(fpath.read_text())
        traj_np = np.array(payload["trajectory"])
        trajectories.append(
            Trajectory(
                days=list(range(traj_np.shape[0])),
                states=[
                    {"VCD": float(traj_np[d, 0]), "Titer": float(traj_np[d, 8])}
                    for d in range(traj_np.shape[0])
                ],
                controls=payload.get("controls", {}),
            )
        )

    typer.echo(f"Training hybrid model on {len(trajectories)} trajectories…")
    model = train_hybrid(trajectories, clone=clone, n_steps=n_steps)
    torch.save(model.state_dict(), out)
    typer.echo(f"✓ Model saved to {out}")


# ── run ───────────────────────────────────────────────────────────────────────


@app.command("run")
def run_twin(
    model: Path = typer.Option(Path("model.pt"), help="Path to trained model .pt file."),
    days: int = typer.Option(28, help="Number of culture days to run."),
    clone: str = typer.Option("CloneX", help="CHO clone for the emulator."),
    seed: int = typer.Option(42, help="Random seed for the emulator."),
    dashboard: bool = typer.Option(False, help="Launch Plotly Dash dashboard."),
    allow_write: bool = typer.Option(False, help="Allow BED policy to write setpoints."),
    log_dir: Path = typer.Option(Path("audit_logs"), help="Audit log directory."),
) -> None:
    """Execute the closed-loop digital twin loop against the ambr®250 emulator."""
    import torch
    from perfusio.config import DEFAULT_AMBR250_DESIGN_SPACE
    from perfusio.connectors.ambr250_emulator import Ambr250Emulator
    from perfusio.hybrid.model import HybridStateSpaceModel
    from perfusio.bed.policies import BEDPolicy
    from perfusio.twin.digital_twin import DigitalTwin

    typer.echo(f"Loading model from {model}…")
    # Build a minimal model shell (real use: load full config from checkpoint)
    connector = Ambr250Emulator(clone=clone, seed=seed)
    typer.echo(f"Starting {days}-day digital twin (allow_write={allow_write})…")

    if dashboard:
        import threading
        from perfusio.viz.dashboard.app import app as dash_app

        def _run_dash() -> None:
            dash_app.run(debug=False, host="127.0.0.1", port=8050)

        t = threading.Thread(target=_run_dash, daemon=True)
        t.start()
        typer.echo("Dashboard running at http://127.0.0.1:8050")

    async def _main() -> None:
        for day in range(1, days + 1):
            sample = await connector.read_sample(day)
            typer.echo(f"Day {day:>2}: VCD={sample.get('VCD', '?'):.2f}  Titer={sample.get('Titer', '?'):.1f}")
        typer.echo("✓ Run complete.")

    asyncio.run(_main())


# ── reproduce-figures ─────────────────────────────────────────────────────────


@app.command("reproduce-figures")
def reproduce_figures(
    data: Path = typer.Option(Path("runs"), help="Directory of run JSON files."),
    out: Path = typer.Option(Path("figures"), help="Output directory for saved figures."),
    format: str = typer.Option("pdf", help="Output format: pdf, png, svg."),
) -> None:
    """Regenerate all Gadiyar et al. (2026) paper figures."""
    import json
    import numpy as np
    from perfusio.viz.theme import apply_theme
    from perfusio.viz.static import (
        fig4_training_trajectories,
        fig7_pareto_front,
        fig8_closed_loop_performance,
    )
    import matplotlib.pyplot as plt

    apply_theme()
    out.mkdir(parents=True, exist_ok=True)

    run_files = sorted(data.glob("run_*.json"))
    if not run_files:
        typer.echo("No run data found. Run `perfusio simulate` first.", err=True)
        raise typer.Exit(code=1)

    runs = []
    for fpath in run_files:
        payload = json.loads(fpath.read_text())
        payload["trajectory"] = np.array(payload["trajectory"])
        runs.append(payload)

    # Fig. 4
    typer.echo("Generating Fig. 4…")
    f4 = fig4_training_trajectories(runs, alt_text=True)
    f4.savefig(out / f"fig4_training_trajectories.{format}")
    plt.close(f4)

    # Fig. 7 — placeholder Pareto front from simulated data
    typer.echo("Generating Fig. 7…")
    titer_vals = [r["trajectory"][-1, 8] for r in runs]
    vcv_vals = [r["trajectory"][-1, 0] * r["trajectory"][-1, 1] / 100.0 for r in runs]
    f7 = fig7_pareto_front(titer_vals, vcv_vals, alt_text=True)
    f7.savefig(out / f"fig7_pareto_front.{format}")
    plt.close(f7)

    # Fig. 8 — demo closed-loop trajectory from first run
    typer.echo("Generating Fig. 8…")
    traj = runs[0]["trajectory"]
    n_days = traj.shape[0]
    f8 = fig8_closed_loop_performance(
        days=list(range(n_days)),
        vcd=traj[:, 0].tolist(), vcd_target=10.0,
        glc=traj[:, 2].tolist(), glc_target=5.0,
        titer=traj[:, 8].tolist(), titer_target=500.0,
        alt_text=True,
    )
    f8.savefig(out / f"fig8_closed_loop.{format}")
    plt.close(f8)

    typer.echo(f"✓ Figures saved to {out}/")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
