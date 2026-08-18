# gsplat_mojo

3D Gaussian Splatting rendering kernels implemented in [Mojo](https://www.modular.com/mojo), targeting GPU acceleration via Modular's MAX platform.

> **Status:** Built against current Mojo (`1.1.0.dev`). The forward pass runs
> end to end on GPU — PLY load, projection, tile binning, depth sort, and
> rasterization — and renders `assets/christmas_tree.ply` (329k gaussians).
> Verified pixel-exactly against independent references on synthetic scenes,
> and against a float64 reference on the real one. The MAX custom-op path
> (`render.mojo`) has not been ported, and there is no backward pass.

## Project Structure

```
gsplat_mojo/
├── assets/
│   └── christmas_tree.ply          # Sample PLY model for testing
├── gsplat/
│   ├── mojoproject.toml            # Mojo project config (read by pixi)
│   ├── pixi.lock                   # Dependency lock file
│   ├── config.mojo                 # Shared sizes, constants and tensor layouts
│   ├── scene.mojo                  # Deterministic test scenes
│   ├── ply.mojo                    # INRIA-format 3DGS PLY loader
│   ├── refmath.mojo                # float32/float64 host reference for the intersection
│   ├── vec.mojo                    # Vec3/Vec4 math primitives
│   ├── utils_t.mojo                # Mat3/Mat2, rotation and quaternion helpers
│   └── operations/
│       ├── intersect.mojo          # Tile binning + GPU bitonic depth sort
│       ├── intersect_test.mojo     # Exact check of the binning stage
│       ├── gsplat_forward.mojo     # Rasterizer + 3-phase self-check (has main())
│       ├── render_ply.mojo         # Renders a real PLY and verifies it
│       └── render.mojo             # MAX custom op — NOT ported, does not compile
└── README.md
```

### Key Files

| File | Description |
|------|-------------|
| `config.mojo` | Scene/image sizes, compositing cutoffs, and every tensor layout, shared so the stages agree |
| `scene.mojo` | Deterministic test scenes — pure functions of the gaussian index, so device staging and host reference never drift |
| `vec.mojo` | SIMD-backed `Vec3` / `Vec4` with arithmetic, dot/cross product, normalization |
| `utils_t.mojo` | `Mat3`/`Mat2` value types plus `rotation_matrix_to_quaternion`, `quat_to_rotmat`, `transpose`, `matmul3x3`, `g_scalar` |
| `operations/intersect.mojo` | Projection, tile-footprint bounding, prefix sum, key emission, GPU bitonic sort, tile-offset extraction |
| `operations/gsplat_forward.mojo` | Ray/gaussian rasterization kernel and the self-checking render |
| `ply.mojo` | Parses binary 3DGS PLY files and undoes the trainer's parameterization (sigmoid opacity, exp scale, SH colour, quaternion reorder) |
| `operations/render_ply.mojo` | Loads a PLY, renders it, writes a PPM, and verifies sampled pixels |

## Running the checks

```bash
cd gsplat
pixi run forward     # rasterizer, 3 phases
pixi run intersect   # binning + depth sort, against a serial reference
pixi run render-ply  # render assets/christmas_tree.ply -> render.ppm
```

`render.ppm` is a plain binary PPM; convert it with
`pnmtopng render.ppm > render.png` or `magick render.ppm render.png`.

## Prerequisites

- **Linux x86_64** (only platform currently supported)
- **GPU** with driver support for Modular MAX
- [pixi](https://pixi.sh) — manages the Mojo/MAX toolchain and environment

Install pixi:
```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

> The `magic` CLI this project originally used has been retired by Modular.
> `pixi` reads the same `mojoproject.toml`, so no manifest change was needed.

## Setup

Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-user>/gsplat_mojo.git
cd gsplat_mojo/gsplat
pixi install
```

This reads `mojoproject.toml`, resolves dependencies (including the MAX/Mojo nightly toolchain), and creates the environment under `.pixi/`.

## Running

All commands should be run from the `gsplat/` directory.

### Run the forward rasterization kernel

The main entry point is in `operations/gsplat_forward.mojo`. Project tasks in
`mojoproject.toml` handle include paths; the manual form is:

```bash
pixi run mojo run -I . operations/gsplat_forward.mojo
```

### Format code

```bash
pixi run mojo format .
```

## Current State

The whole forward pipeline runs on GPU:

```
load PLY  ->  project & count  ->  two-level prefix sum  ->  emit (tile, depth) keys
          ->  bitonic sort  ->  tile offsets  ->  rasterize
```

### Rendering a real scene

`pixi run render-ply` loads all 329,004 gaussians of
`assets/christmas_tree.ply`, renders at 1024x768, and writes `render.ppm`:

![christmas tree render](assets/render_preview.png)

```
gaussians: 329004
tile intersections: 2258276
sorted 4194304 slots in 253 bitonic passes
coverage: 723850 of 786432 px lit ( 92 % ) | mean alpha 0.606 | max alpha 0.9999
max |GPU      - float64 truth| 0.0162   mean 3.29e-05
max |host f32 - float64 truth| 0.0174   mean 3.11e-05   <- float32 noise floor
PASS: PLY render is within the float32 noise floor of the independent reference
```

A whole-image host reference is not affordable at this scale, so 4096 spread
pixels are recomputed on the host from the tile lists the GPU actually
produced. **The bar is not a fixed tolerance.** These gaussians are ~1e-3
across and ~5 units away, so the intersection ends in `p = og + t*·dg`, a
near-total cancellation; evaluating that in float32 is genuinely uncertain.
The float32 host reference misses the float64 truth by *more* than the GPU
does (0.0174 vs 0.0162), so the test asserts the GPU is no worse than an
independent float32 evaluation rather than demanding better-than-float32.
Typical error is 3e-05; the millipixel-scale outliers sit on chains up to 196
gaussians deep.

### Synthetic self-checks

`pixi run forward` renders 24 gaussians into 1024x768 and checks **every**
pixel three ways:

| Phase | Scene | Checked against | Max error |
|-------|-------|-----------------|-----------|
| 1 | isotropic, on optical axis, identity camera | closed form `rho^2 = z^2 w / ((1+w) s^2)` | 1.5e-07 |
| 2 | rotated, anisotropic, off-axis; yawed + translated camera | `_ref_rho2`, a scalar reference written from the definitions | 1.3e-06 |
| 3 | same as 2, binned and depth-sorted by the real intersection stage | the phase 2 image (brute force) | **0.0** |

Phase 3 is exact: real binning visits 1367 tile/gaussian pairs instead of
73728, a **54x** reduction, and reproduces the brute-force image bit for bit.
`pixi run intersect` separately checks the binning against a serial host
implementation: total count, every `tile_offsets` entry, the set of gaussians
per tile, and depth ordering within each run.

All of it is mutation-tested rather than merely green:

| Mutation | Caught by |
|----------|-----------|
| `transpose(rotation)` -> `rotation` in `S^-1 R^T` | phase 2 (50k px); phase 1 still passes |
| skip the bitonic passes | `intersect`: 2111 bad offsets, 902 bad tile sets |
| shrink the footprint bound to 0.6x | phase 3: 32657 px differing; `intersect` still passes |

Still outstanding:

- **The bitonic sort is `O(n log^2 n)`** — 253 kernel launches for the tree.
  gsplat uses a radix sort here; this is correct but the main performance gap.
- **The prefix sum is two-level**, capping input at `SCAN_BLOCK * SCAN_WIDTH`
  = 1,048,576 gaussians. Fine for this scene, but not unbounded.
- **No spherical harmonics** — only the order-0 (view-independent) colour is
  used, so the render has no view-dependent shading.
- **`operations/render.mojo` does not compile.** It targets the old custom-op
  API (`@compiler.register`, `tensor.InputTensor`, `runtime.asyncrt`); these
  are now `extensibility.register` / `extensibility.InputTensor`, and
  `DeviceContextPtr` no longer exists.
- **No backward pass**, so this renders but cannot train.

## Architecture

The rendering pipeline follows the standard 3D Gaussian Splatting approach:

1. **World-space gaussians** are defined by their means (3D position), quaternion rotation, scales, colors (RGB), and opacities
2. **Camera projection** uses SE3 transforms (rotation + translation) and intrinsic matrices (focal length, principal point)
3. **Tile-based rasterization** divides the image into 16x16 pixel tiles, each processed by a GPU thread block
4. **Shared memory** is used to batch gaussian data per tile for efficient parallel accumulation
5. **Alpha compositing** accumulates color contributions front-to-back with transparency

Default render parameters: 1024x768 image, 16x16 tiles, single camera, RGB output.

## License

MIT
