# Support and feature tiers

This project is pre-alpha. Tiers describe evidence and maintenance priority,
not paid support or guaranteed response times.

## Claim labels

- **Tested — Tier 1:** Continuously or manually exercised on a documented
  NVIDIA/Linux configuration with project correctness checks. A claim applies
  only to the recorded versions and workload.
- **Best effort — Tier 2:** A plausible configuration accepted for community
  reports, but not verified by project CI. Fixes depend on maintainer capacity.
- **Planned:** Not implemented or not validated. Planned features must not be
  presented as available.

## Current support

| Area | Classification | Current boundary |
|---|---|---|
| NVIDIA GPU on Linux x86_64 | **Tested — Tier 1** | Baseline environment and pinned MAX/PyTorch versions only |
| Python 3.11–3.13 | **Tested — Tier 1** | Package metadata and current dependency constraints |
| Experimental ray inference | **Tested — Tier 1** | Exact contract in `docs/api-contract.md` |
| Native Mojo correctness checks | **Tested — Tier 1** | `forward`, `intersect`, and `render-ply` |
| Other MAX accelerators or Linux architectures | **Best effort — Tier 2** | Unverified; reports welcome with reproducible details |
| Windows and macOS | **Planned** | No current package or correctness claim |
| EWA parity, training, dynamic shapes, batching | **Planned** | Roadmap work, unavailable today |

## Getting help

Use a GitHub issue template for reproducible bugs, feature requests, or
documentation problems. Search existing issues first and include versions,
hardware, driver, commands, and complete error output. General usage support is
best effort; security reports must follow [`SECURITY.md`](SECURITY.md).

Promotion from Tier 2 to Tier 1 requires repeatable correctness evidence and an
ongoing CI or documented validation path. A successful one-off report is not
enough.
