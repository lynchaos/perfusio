# Limitations

This document explicitly lists what `perfusio` does **not** model or support.

## Out-of-Scope Biology

- **Critical Quality Attributes (CQAs)** — No modelling of glycoforms,
  charge variants, aggregates, HCP, or other product quality attributes.
  The model tracks titer (mg/L) only.
- **Medium composition** — The medium formulation is hard-coded to a generic
  CHO basal medium. Proprietary feed and basal media are not supported.
- **Cell line specifics** — Only two virtual cell lines (CloneX, CloneY) are
  provided. Real cell-line parameters require calibration.
- **3D fluid dynamics** — The model is zero-dimensional (perfectly mixed).
  Spatial gradients in ambr®250 microbioreactors are not captured.
- **Viral clearance** — No modelling of virus removal steps or biosafety
  considerations.

## Out-of-Scope Engineering

- **Dissolved oxygen control** — DO is tracked as a state variable but
  control (sparge rate, agitation) is not modelled.
- **pH control** — pH is not a state variable in this implementation.
- **Temperature ramping** — Temperature is a fixed parameter; dynamic
  temperature shift protocols are not supported.
- **Fed-batch mode** — The model assumes continuous perfusion; fed-batch
  and perfusion-start transitions are not implemented.

## Regulatory / Quality

- **GMP validation** — `perfusio` is not validated under 21 CFR Part 11,
  EU Annex 11, or any GxP framework. See
  `docs/source/regulatory-considerations.md`.
- **Electronic signatures** — `AuditLogger` produces audit records but not
  compliant electronic signatures.
- **CSV injection** — The `FilesystemStore` does not sanitise CSV fields
  for spreadsheet-formula injection; do not open output CSVs in untrusted
  spreadsheet applications.

## Performance

- **Wall-clock speed** — Training 27 Box-Behnken runs × 28 days on CPU
  takes approximately 2–5 minutes. GPU acceleration is available but
  not CI-tested.
- **Scalability** — The SQL schema and SQLite backend are not intended for
  multi-site or high-throughput deployments (> 100 concurrent reactors).
