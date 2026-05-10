# Comparison with DataHowLab / Datahow

> **Disclaimer** — This comparison is based solely on publicly available
> information and the methodology described in Gadiyar et al. (2026).
> Datahow is a commercial product; features and algorithms may change.
> No claim is made about the quality or suitability of either tool
> for any specific regulatory or manufacturing purpose.

## High-Level Summary

| Feature | `perfusio` | DataHowLab |
|---------|-----------|------------|
| License | Apache-2.0 (open source) | Commercial SaaS |
| Model type | Hybrid mechanistic + SW-GP | Hybrid (proprietary) |
| Transfer learning | Entity embeddings (Hutter 2021) | Unknown |
| BED acquisitions | 11 (PI, EI, LogEI, UCB, qEI, …) | Unknown |
| OPC UA connector | Yes (`asyncua`) | Yes (vendor) |
| GMP compliance | Out of scope (see LIMITATIONS.md) | Vendor-validated |
| Language | Python 3.11+ | Web interface + Python SDK |
| Figure reproducibility | Yes (paper figures reproducible) | N/A |

## Design Philosophy

`perfusio` prioritises **transparency and reproducibility**:
every equation, parameter, and figure is traceable to the published literature.
DataHowLab prioritises **industrial deployment** with a validated SaaS platform.

These are complementary, not competing, goals.

## When to use `perfusio`

- Academic research and peer review.
- Reproducing or extending Gadiyar et al. (2026).
- Benchmarking novel acquisition functions or GP kernels.
- Teaching bioprocess digital twins.

## When to use DataHowLab (or similar)

- GMP-regulated manufacturing environments.
- Integration with commercial LIMS / MES systems.
- Long-term vendor support and validation packages.
