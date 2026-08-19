# Validated radix-renderer baseline

All positive correctness statements in this document are
**Tested — Tier 1** and apply only to the recorded environment and workloads.
Other MAX accelerators are **Best effort — Tier 2**. Capabilities named under
"Limits of this baseline" are **Planned**, not available claims.

This focused Phase 0 evidence was captured on 2026-08-18 from a dirty working
tree based on commit `ed20784` that also contained the uncommitted Phase 0
changes described below. The results are not attributable to immutable commit
`ed20784` and should not be treated as a commit-reproducible baseline. They
record correctness evidence for that working-tree state, not a portability or
performance claim.

## Environment

- GPU: NVIDIA RTX 2000 Ada Generation Laptop GPU, 8188 MiB
- NVIDIA driver: 580.159.04
- Mojo: `1.1.0.dev2026081813` (`8cd05901`)
- pixi: `0.50.2`
- OS/architecture: Linux x86_64

This is a tested Tier 1 NVIDIA configuration. Other MAX accelerators remain
best-effort and unverified by this baseline.

## Focused checks

Run from `gsplat/`:

```bash
pixi run intersect
pixi run forward
pixi run render-ply
```

All three commands passed.

### Intersection and sort

- 9,354 tile/Gaussian intersections across five radix blocks
- 11 radix passes
- 0 of 9,354 slots differed from the independent bitonic sort
- host-reference count: 9,354
- tile-offset, membership, and ordering mismatches: 0

The renderer production path is LSD radix sort. Bitonic is test-only and is
retained to cross-check the same unsorted key/value stream.

### Synthetic forward renderer

- phase 1 closed-form maximum errors: color `1.4901161e-07`, alpha
  `1.7881393e-07`
- phase 2 scalar-reference maximum errors: color `1.3411045e-06`, alpha
  `2.2649765e-06`
- phase 3 used 1,367 intersections instead of 73,728 brute-force visits
- phase 3 versus phase 2: exact, with 0 differing pixels

### Real PLY scene

- input: `assets/christmas_tree.ply`, 329,004 Gaussians, SH degree 0
- output: 1024x768, 2,258,276 tile intersections, 11 radix passes over 1,103
  blocks
- sampled reference: 4,096 pixels, 3,725 with coverage
- maximum GPU versus float64 error: `0.0162144`
- maximum host-float32 versus float64 error: `0.017449826`
- mean GPU versus float64 error: `3.285998e-05`

The real-scene gate passed because the GPU maximum error remained within 1.5x
the independently measured float32 numerical floor. This is not a claim of
pixel equivalence with upstream gsplat: the current renderer uses experimental
ray/Gaussian shading rather than gsplat-compatible EWA shading.

## Release-candidate integration verification

On 2026-08-18, a `0.1.0a0` candidate wheel built from that same dirty working
tree was installed into a newly created uv environment with Python 3.13 using
only the documented Modular index command. Import and kernel-source discovery
succeeded from outside the checkout, `driver.py` rendered on CUDA, and all 11
Python tests passed, including the GPU smoke test. These results remain
working-tree evidence rather than verification of an immutable release
candidate.

The GPU smoke test dispatches from a non-default PyTorch CUDA stream, records a
CUDA event after the call, and consumes the CUDA outputs from the default
stream only after an event dependency. This verifies device residency and
usable cross-stream ordering through the public API rather than inferring them
from pointer equality. It does not claim fully asynchronous execution: input
validation performs scalar checks, and the current custom operation contains a
documented device-to-host synchronization to size its intersection sort.

## Limits of this baseline

These checks do not establish backward/autograd support, EWA parity, dynamic
image/camera batching, non-NVIDIA support, or a speed advantage over upstream
gsplat. Those remain later roadmap gates.
