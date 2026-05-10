# Regulatory Considerations

> **This page is informational only and does not constitute legal,
> regulatory, or quality advice. Consult your regulatory affairs team
> before using any software in a GMP context.**

## Is `perfusio` GxP-validated?

**No.** `perfusio` is a research library released under the Apache-2.0
licence. It has not been validated under 21 CFR Part 11, Annex 11, or
any other GMP framework.

## Using `perfusio` in a Research / Development Context

`perfusio` is appropriate for:

- Pre-GMP process development and optimisation.
- Generating hypotheses and experimental designs.
- Academic publication and benchmarking.

## Path to GMP Deployment

If you wish to use the methodology in a GMP-regulated context, you will
typically need to:

1. **Source-code qualification** — Lock to a specific release tag; document
   all changes.
2. **Installation Qualification (IQ)** — Verify that the library installs
   correctly on your validated compute environment.
3. **Operational Qualification (OQ)** — Run the provided test suite;
   confirm 92% line / 85% branch coverage pass.
4. **Performance Qualification (PQ)** — Execute the paper-figure
   reproduction scripts and compare output to archived reference images.
5. **21 CFR Part 11 / Annex 11 controls** — Implement audit trail, access
   control, and electronic signature wrappers around `AuditLogger` and
   `SQLStore`.

## Audit Trail

`perfusio.twin.audit.AuditLogger` logs all sampling events and setpoint
changes with ISO 8601 timestamps, operator ID, and a SHA-256 hash of the
state dict. This provides a **technical foundation** for audit trails but
does not replace a validated audit trail system.

## Data Integrity

All OPC UA reads are validated against configured engineering limits before
being stored. Rejected reads are logged with reason codes.

## Contact

For GMP consulting around `perfusio`, contact the corresponding author of
Gadiyar et al. (2026) or open an issue on GitHub.
