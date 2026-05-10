"""Example 05 — Entity-embedding transfer across CHO cell lines.

Trains on Clone X data, transfers to Clone Y with 6 fine-tuning runs.

Run::

    python examples/05_entity_embedding_transfer.py
"""

from __future__ import annotations

from pathlib import Path
import json

import torch

from perfusio.embedding.clones import CloneRegistry
from perfusio.embedding.transfer import TransferLearner
from perfusio.simulator.cho_perfusion import CHOSimulator
from perfusio.config import DEFAULT_AMBR250_DESIGN_SPACE

REGISTRY = CloneRegistry.default()
DS = DEFAULT_AMBR250_DESIGN_SPACE

# ── Source domain: Clone X (10 runs) ─────────────────────────────────────────
sim_x = CHOSimulator(clone="CloneX", seed=0)
runs_x = sim_x.generate_box_behnken_experiment(n_days=14, seed=0)[:10]
trajs_x = [torch.tensor(r["trajectory"], dtype=torch.float64) for r in runs_x]

# ── Target domain: Clone Y (only 6 fine-tuning runs) ─────────────────────────
sim_y = CHOSimulator(clone="CloneY", seed=1)
runs_y = sim_y.generate_box_behnken_experiment(n_days=14, seed=1)[:6]
trajs_y = [torch.tensor(r["trajectory"], dtype=torch.float64) for r in runs_y]

# ── Transfer learning ─────────────────────────────────────────────────────────
learner = TransferLearner(clone_registry=REGISTRY, design_space=DS)
learner.warm_start(trajs_x, clone_id=0)          # Clone X id=0
learner.joint_finetune(trajs_y, clone_id=1)       # Clone Y id=1

torch.save(learner.state_dict(), "transfer_model.pt")
print("Transfer model saved to transfer_model.pt")
print(f"Embedding weights:\n{learner.embedding.weight.data}")
