# T05 — Transfer Learning Across Cell Lines

**Goal:** Warm-start a model on CloneX data and fine-tune it on just 6
CloneY runs using entity-embedding transfer learning (Hutter 2021).

**Script:** `examples/05_entity_embedding_transfer.py`

## What the script does

1. Simulates 10 CloneX training runs (14-day).
2. Simulates 6 CloneY fine-tuning runs (14-day).
3. Creates a `TransferLearner` with `EntityEmbedding(n_clones=2, embed_dim=4)`.
4. Calls `learner.warm_start(trajs_x, clone_id=0)` — fits on CloneX.
5. Calls `learner.joint_finetune(trajs_y, clone_id=1)` — fine-tunes on CloneY
   with a lower learning rate for shared weights.
6. Saves `transfer_model.pt`.

## Running

```bash
python examples/05_entity_embedding_transfer.py
# → transfer_model.pt
```

## Key code

```python
from perfusio.embedding.transfer import TransferLearner
from perfusio.embedding.clones import CloneRegistry

registry = CloneRegistry.default()  # CloneX (id=0), CloneY (id=1)
learner  = TransferLearner(clone_registry=registry, design_space=DS)

learner.warm_start(source_trajectories, clone_id=0)
learner.joint_finetune(target_trajectories, clone_id=1)
```

## Architecture

```
clone_id ──► EntityEmbedding (4-dim) ──► concat ──► MultiTaskRateGP
                                              ▲
                                     [state, controls, day]
```

Each `clone_id` maps to a learned 4-dimensional embedding vector.
The embedding is concatenated to the kinetic input features before the GP
kernel, giving the model a differentiable representation of clone identity.

## Clone registry

```python
from perfusio.embedding.clones import CloneRegistry, CloneParams

registry = CloneRegistry(clones={
    0: CloneParams(name="CloneX", consumes_lactate=True),
    1: CloneParams(name="CloneY", consumes_lactate=False),
})
```

## When to use transfer learning

- You have ≥ 10 runs from a well-characterised source clone.
- The target clone has only 3–8 runs available.
- The clones share similar metabolic phenotypes (same CHO host, similar medium).

## Next step

Proceed to [T06 — Online Digital Twin (Filesystem)](T06-digital-twin.md).
