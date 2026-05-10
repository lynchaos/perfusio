# Choosing an Acquisition Function

This guide helps you select the right acquisition function for your
experimental campaign. All are available via `perfusio.bed.acquisitions.build_acquisition`.

## Decision Tree

```
Are you optimising more than one objective?
├── Yes → use qNEHVI (default) or qNParEGO (fast alternative)
└── No → continue
    │
    ├── Batch size q > 1?
    │   ├── Yes → qLogEI (recommended) or qEI
    │   └── No → continue
    │
    ├── Noisy observations?
    │   ├── Yes → LogEI (numerically robust)
    │   └── No → EI or PI
    │
    └── Exploration emphasis?
        ├── Yes → UCB with high β
        └── No → EI (balanced explore/exploit)
```

## Quick Reference

| Acquisition | When to use | Limitations |
|-------------|-------------|-------------|
| `EI` | Default single-obj | Can underperform near optimum |
| `LogEI` | Noisy / near-optimal | Slightly more compute |
| `PI` | Conservative campaigns | Greedy; can stall |
| `UCB` | Heavy exploration | β requires tuning |
| `qEI` | Parallel (q>1) batches | Slow for large q |
| `qLogEI` | Parallel + noisy | Best default for q>1 |
| `qUCB` | Parallel exploration | β tuning needed |
| `qEHVI` | Multi-obj noiseless | Slow for m>3 |
| `qNEHVI` | Multi-obj noisy | Slower than qEHVI |
| `qNParEGO` | Multi-obj, fast | Scalarisation bias |

## Recommended Defaults

- **Single-run selection**: `LogEI`
- **Batch (q=4)**: `qLogEI`
- **Two objectives**: `qNEHVI`
- **Three or more objectives**: `qNParEGO` (HV computation becomes expensive)
