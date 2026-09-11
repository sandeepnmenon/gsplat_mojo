# Changelog

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and intends to use [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Tested — Tier 1:** inference-only `gsplat_mojo.rasterization` contract with
  an explicitly selected experimental ray renderer.
- **Tested — Tier 1:** PyTorch/MAX custom-op path returning CUDA color and alpha
  tensors.
- **Tested — Tier 1:** input validation, package source inclusion, and GPU smoke
  coverage.
- **Tested — Tier 1:** clean uv wheel installation with the pinned Modular
  nightly dependency set and non-default CUDA stream ordering coverage.
- **Tested — Tier 1:** LSD radix sort as the active rendering sort, with
  independent bitonic and host-reference checks.
- Public installation, validation, support, contribution, conduct, security,
  citation, roadmap, and asset-provenance documentation.

### Known limitations

- **Planned:** EWA compatibility, autograd, dynamic resolution, camera
  batching, and stable release support.
- **Best effort — Tier 2:** accelerators outside the documented NVIDIA/Linux
  baseline.
- Release redistribution is blocked by unresolved sample-asset provenance and
  missing private security/conduct intake policy.

## [0.1.0a0] — Unreleased

**Planned:** Pre-alpha candidate version. No tag or package release has been
published by this documentation change.
