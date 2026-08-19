# Roadmap

This roadmap is directional, not a delivery promise. Completed capability
claims are labeled **Tested — Tier 1**; unverified portability is
**Best effort — Tier 2**; future work is **Planned**.

## Phase 0 — trustworthy pre-alpha

- **Tested — Tier 1:** inference-only PyTorch contract and explicit
  experimental ray mode.
- **Tested — Tier 1:** CUDA-resident color/alpha custom-op smoke path.
- **Tested — Tier 1:** native projection, binning, radix-sort, synthetic, and
  sampled real-scene correctness checks.
- **Best effort — Tier 2:** other MAX accelerator backends.
- **Tested — Tier 1:** automated GPU smoke verification, clean wheel install,
  and schema-validated machine-readable ray self-baseline.
- **Planned:** resolve asset provenance and private security/conduct reporting
  before release.
- **Tested — Tier 1:** prepare the `v0.1.0a0` release candidate.
- **Planned:** publishing and tagging require explicit approval after the
  remaining release blockers are resolved.

## Phase 1 — compatible inference

- **Planned:** gsplat-compatible EWA default, full SH evaluation through the
  public API, parity fixtures, and same-semantics comparison with a pinned
  upstream gsplat release.

## Phase 2 — scale and portability

- **Planned:** dynamic image dimensions, camera batches, removal of fixed
  scan/capacity ceilings, edge-case coverage, and evidence-based promotion of
  additional accelerators.

## Phase 3 — performance

- **Planned:** profile-gated fusion, bounded mixed precision, runtime tile
  dispatch, memory reduction, and continuous regression thresholds.

## Phase 4 — research inference

- **Planned:** Mip-Splatting, 2DGS as a distinct mode, quantized SH, and
  additional camera models.

## Phase 5 — optional training

- **Planned:** backward kernels, numerical gradient checks, PyTorch autograd
  integration, and training-framework integration.
