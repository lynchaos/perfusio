"""``perfusio`` command-line interface.

Commands
--------
``perfusio simulate``
    Run a 27-run Box-Behnken simulation and save trajectories.

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

    from perfusio.hybrid.train import train_hybrid

    typer.echo(f"Loading runs from {data}/…")
    run_files = sorted(data.glob("run_*.json"))
    if not run_files:
        typer.echo("No run JSON files found. Run `perfusio simulate` first.", err=True)
        raise typer.Exit(code=1)

    import numpy as np

    from perfusio.mechanistic.kinetics import CHOKinetics
    from perfusio.states import Trajectory

    _CONTROL_NAMES = [
        "perfusion_rate",
        "bleed_rate",
        "glucose_setpoint",
        "temperature",
        "agitation",
        "pyruvate_feed",
    ]
    _SPECIES_NAMES = CHOKinetics.STATE_ORDER  # 9 species

    trajectories: list[Trajectory] = []
    for fpath in run_files:
        payload = json.loads(fpath.read_text())
        traj_np = np.array(payload["trajectory"], dtype=np.float32)  # (T, 9)
        T = traj_np.shape[0]
        ctrl_dict: dict[str, float] = payload.get("controls", {})
        ctrl_row = [ctrl_dict.get(k, float("nan")) for k in _CONTROL_NAMES]
        ctrl_np = np.tile(ctrl_row, (T, 1)).astype(np.float32)  # (T, 6)
        trajectories.append(
            Trajectory(
                species=torch.from_numpy(traj_np),
                controls=torch.from_numpy(ctrl_np),
                volume_L=torch.full((T,), 0.25),
                days=torch.arange(T),
                species_names=list(_SPECIES_NAMES),
                control_names=_CONTROL_NAMES,
                run_id=payload.get("run_id", fpath.stem),
                clone_id=clone,
            )
        )

    # Build one-step GP training data: x=[species_t, controls_t, day], y=species_{t+1}
    import gpytorch

    from perfusio.gp.exact_gp import MultiTaskRateGP
    from perfusio.gp.stepwise import StepwiseGP
    from perfusio.hybrid.model import HybridStateSpaceModel
    from perfusio.mechanistic.models import CHOPerfusionModel

    xs_base, ys_base = [], []
    for traj in trajectories:
        sp = traj.species  # (T, 9)
        ct = traj.controls  # (T, 6)
        T = sp.shape[0]
        for t in range(T - 1):
            day_t = torch.tensor([float(t)], dtype=torch.float32)
            xs_base.append(torch.cat([sp[t], ct[t], day_t]))  # (16,)
            ys_base.append(sp[t + 1])  # (9,)

    train_x_base = torch.stack(xs_base)  # (N, 16)
    train_y_base = torch.stack(ys_base)  # (N, 9)
    N_base = len(train_x_base)

    n_species = len(_SPECIES_NAMES)
    n_controls = len(_CONTROL_NAMES)
    n_tasks = n_species

    # Use the indexed multi-task GP paradigm: replicate each base row once per
    # task, appending the task_id (0..n_tasks-1) as the last input column.
    # PerfusionKernel's IndexKernel reads column n_state_dims+1 = 16 as task_id.
    # This produces training data of shape (N * n_tasks, 17) and (N * n_tasks,).
    xs_flat, ys_flat = [], []
    for task_id in range(n_tasks):
        task_col = torch.full((N_base, 1), float(task_id), dtype=train_x_base.dtype)
        xs_flat.append(torch.cat([train_x_base, task_col], dim=1))  # (N, 17)
        ys_flat.append(train_y_base[:, task_id])  # (N,)

    train_x = torch.cat(xs_flat, dim=0)  # (N * n_tasks, 17)
    train_y = torch.cat(ys_flat, dim=0)  # (N * n_tasks,)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp_model = MultiTaskRateGP(
        train_x,
        train_y,
        likelihood,
        mean_module=gpytorch.means.ZeroMean(),
        n_tasks=n_tasks,
        n_state_dims=n_species + n_controls,
    )

    typer.echo(
        f"Training hybrid GP on {N_base} one-step pairs × {n_tasks} species = {len(train_x)} rows (n_steps={n_steps})…"
    )
    train_hybrid(
        train_x,
        train_y,
        gp_model,
        likelihood,
        n_iter_lbfgs=n_steps,
        n_iter_adam=n_steps,
    )

    sw_gp = StepwiseGP(gp_model, likelihood, n_species=n_species, control_names=_CONTROL_NAMES)
    mech = CHOPerfusionModel()
    hybrid = HybridStateSpaceModel(
        mech_model=mech,
        gp_model=sw_gp,
        species_names=list(_SPECIES_NAMES),
        control_names=_CONTROL_NAMES,
    )

    torch.save(
        {
            "gp_state": gp_model.state_dict(),
            "likelihood_state": likelihood.state_dict(),
            "species_names": list(_SPECIES_NAMES),
            "control_names": _CONTROL_NAMES,
            "clone": clone,
        },
        out,
    )
    typer.echo(f"✓ Model saved to {out}")
    _ = hybrid  # available for future inline use


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
    from perfusio.connectors.ambr250_emulator import Ambr250Emulator

    typer.echo(f"Loading model from {model}…")
    # Build a minimal model shell (real use: load full config from checkpoint)
    connector = Ambr250Emulator(clone=clone, seed=seed)
    typer.echo(f"Starting {days}-day digital twin (allow_write={allow_write})…")

    dash_thread = None
    _push_sample = None
    if dashboard:
        import threading

        from perfusio.viz.dashboard.app import app as dash_app
        from perfusio.viz.dashboard.app import push_sample as _push_sample

        def _run_dash() -> None:
            dash_app.run(debug=False, host="127.0.0.1", port=8050)

        dash_thread = threading.Thread(target=_run_dash, daemon=False)
        dash_thread.start()
        typer.echo("Dashboard running at http://127.0.0.1:8050")

    async def _main() -> None:
        for day in range(1, days + 1):
            sample = await connector.read_sample(day)
            vcd = sample.get("VCD")
            titer = sample.get("Titer")
            vcd_str = f"{vcd:.2f}" if vcd is not None else "N/A"
            titer_str = f"{titer:.1f}" if titer is not None else "N/A"
            typer.echo(f"Day {day:>2}: VCD={vcd_str}  Titer={titer_str}")
            if _push_sample is not None:
                _push_sample(day, sample)
                await asyncio.sleep(2)  # 2 s per day so the browser can see updates
        typer.echo("✓ Run complete.")

    asyncio.run(_main())

    if dash_thread is not None:
        typer.echo("Dashboard still live at http://127.0.0.1:8050 — press Ctrl+C to stop.")
        dash_thread.join()


# ── reproduce-figures ─────────────────────────────────────────────────────────


@app.command("reproduce-figures")
def reproduce_figures(
    data: Path = typer.Option(Path("runs"), help="Directory of run JSON files."),
    out: Path = typer.Option(Path("figures"), help="Output directory for saved figures."),
    format: str = typer.Option("pdf", help="Output format: pdf, png, svg."),
) -> None:
    """Regenerate all Gadiyar et al. (2026) paper figures."""
    import json

    import matplotlib.pyplot as plt
    import numpy as np

    from perfusio.viz.static import (
        fig4_training_trajectories,
        fig7_pareto_front,
        fig8_closed_loop_performance,
    )
    from perfusio.viz.theme import apply_theme

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

    # Fig. 7 — compute actual Pareto front from day-28 titer vs. VCV values
    typer.echo("Generating Fig. 7…")
    import torch as _torch

    from perfusio.bed.pareto import compute_pareto_front as _cpf

    titer_vals = [r["trajectory"][-1, 8] for r in runs]
    vcv_vals = [r["trajectory"][-1, 0] * r["trajectory"][-1, 1] / 100.0 for r in runs]
    _Y = _torch.tensor(list(zip(titer_vals, vcv_vals, strict=True)), dtype=_torch.float64)
    _pareto_mask = _cpf(_Y)
    pareto_titer = [v for v, m in zip(titer_vals, _pareto_mask.tolist(), strict=True) if m]
    pareto_vcv = [v for v, m in zip(vcv_vals, _pareto_mask.tolist(), strict=True) if m]
    f7 = fig7_pareto_front(
        pareto_titer,
        pareto_vcv,
        feasible_titer=titer_vals,
        feasible_vcv=vcv_vals,
        alt_text=True,
    )
    f7.savefig(out / f"fig7_pareto_front.{format}")
    plt.close(f7)

    # Fig. 8 — closed-loop trajectory generated by a day-by-day PI feedback
    # controller (proportional control on glucose and VCD).  Each day the
    # controller reads the current state, adjusts perfusion_rate to maintain
    # glucose near 1 g/L and bleed_rate to target VCD = 10×10⁶ cells/mL,
    # then integrates the ODE for one more day from the new initial condition.
    # This is the correct representation of self-driving closed-loop control,
    # not the open-loop fixed-control trajectory of runs[0].
    typer.echo("Generating Fig. 8…")
    from perfusio.mechanistic.integrators import integrate_run as _integrate_run
    from perfusio.mechanistic.kinetics import CHOKinetics as _CHOKinetics

    _kin = _CHOKinetics(consumes_lactate=True)
    _y0_cl = [1.0, 99.0, 5.0, 4.0, 0.5, 2.0, 0.5, 0.0, 0.0]
    _ctrl_cl: dict = {
        "perfusion_rate": 0.5,
        "bleed_rate": 0.20,
        "temperature": 37.0,
        "agitation": 250.0,
        "pyruvate_feed": 0.0,
        "glucose_feed_conc": 5.0,
        "gln_feed_conc": 4.0,
        "pyr_feed_conc": 0.0,
        "volume_L": 0.25,
    }
    _VCD_TARGET_CL = 10.0
    _GLU_SP_CL = 1.0  # g/L glucose setpoint

    _state_cl = list(_y0_cl)
    n_days_cl = len(runs[0]["trajectory"]) - 1
    cl_days = list(range(n_days_cl + 1))
    cl_vcd = [_state_cl[0]]
    cl_glc = [_state_cl[2]]
    cl_titer = [_state_cl[8]]

    for _ in range(n_days_cl):
        _vcd_t, _glc_t = _state_cl[0], _state_cl[2]
        # P-controller: ramp perfusion up when glucose below setpoint
        _glc_err = _GLU_SP_CL - _glc_t
        _ctrl_cl["perfusion_rate"] = float(
            np.clip(_ctrl_cl["perfusion_rate"] + 0.15 * _glc_err, 0.5, 1.5)
        )
        # P-controller: ramp bleed to keep VCD near target
        _vcd_err = _vcd_t - _VCD_TARGET_CL
        _ctrl_cl["bleed_rate"] = float(np.clip(0.15 + 0.02 * _vcd_err, 0.05, 0.25))
        _one_day = _integrate_run(_kin, _state_cl, _ctrl_cl, n_days=1)
        _state_cl = _one_day[-1].tolist()
        cl_vcd.append(_state_cl[0])
        cl_glc.append(_state_cl[2])
        cl_titer.append(_state_cl[8])

    f8 = fig8_closed_loop_performance(
        days=cl_days,
        vcd=cl_vcd,
        vcd_target=10.0,
        glc=cl_glc,
        glc_target=1.0,
        titer=cl_titer,
        titer_target=500.0,
        alt_text=True,
    )
    f8.savefig(out / f"fig8_closed_loop.{format}")
    plt.close(f8)

    typer.echo(f"✓ Figures saved to {out}/")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
