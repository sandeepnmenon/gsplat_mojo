# gsplat_mojo

3D Gaussian Splatting rendering kernels implemented in [Mojo](https://www.modular.com/mojo), targeting GPU acceleration via Modular's MAX platform.

> **Status:** Work-in-progress. Built against current Mojo (`1.1.0.dev`). The
> forward rasterization kernel compiles, launches, and writes every pixel; the
> per-gaussian ray/gaussian evaluation inside the batch loop is still a stub.

## Project Structure

```
gsplat_mojo/
├── assets/
│   └── christmas_tree.ply          # Sample PLY model for testing
├── gsplat/
│   ├── mojoproject.toml            # Mojo/Magic project config
│   ├── magic.lock                  # Dependency lock file
│   ├── vec.mojo                    # Vec3/Vec4 math primitives
│   ├── utils_t.mojo                # Matrix/quaternion utilities (rotation, transpose, matmul)
│   └── operations/
│       ├── render.mojo             # High-level render dispatcher (registers "render" custom op)
│       └── gsplat_forward.mojo     # GPU kernel: tile-based gaussian rasterization (has main())
└── README.md
```

### Key Files

| File | Description |
|------|-------------|
| `vec.mojo` | SIMD-backed `Vec3` / `Vec4` with arithmetic, dot/cross product, normalization |
| `utils_t.mojo` | `Mat3`/`Mat2` value types plus `rotation_matrix_to_quaternion`, `quat_to_rotmat`, `transpose`, `matmul3x3`, `g_scalar` |
| `operations/gsplat_forward.mojo` | Core GPU rasterization kernel with `SE3` transforms, shared memory tiling, and a `main()` entry point that allocates buffers and launches the kernel |
| `operations/render.mojo` | `@compiler.register("render")` custom op that dispatches to the GPU kernel |

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

The main entry point is in `operations/gsplat_forward.mojo`. It allocates GPU buffers, constructs tensors, and launches the rasterization kernel:

```bash
cd gsplat
pixi run forward
```

This is a project task defined in `mojoproject.toml` that handles include paths automatically. You can also run it manually:

```bash
pixi run mojo run -I . operations/gsplat_forward.mojo
```

### Format code

```bash
pixi run mojo format .
```

## Current State

`pixi run forward` builds and runs a self-checking smoke test of the forward
pass. It stages 4 gaussians and an intersection set where every tile references
every gaussian, launches the kernel over a 64x48 tile grid, then verifies on the
host that all 786,432 pixels were written and that every tile walked its full
gaussian range:

```
launching: 1024 x 768 | tiles 64 x 48 | gaussians 4 | isects 12288
mismatches - color: 0 | alpha: 0 | last_ids: 0
PASS: every pixel written, and every tile streamed all 4 gaussians through shared memory
```

What is live: tile range resolution, shared-memory batching with barriers,
per-gaussian inverse-scale-rotation setup, background compositing, and
write-back to `render_colors` / `render_alphas` / `last_ids`.

Still outstanding:

- The per-gaussian ray/gaussian evaluation inside the batch loop is a stub, so
  transmittance stays 1 and a pixel resolves to just the background. This is the
  marked `TODO` in `gsplat_forward.mojo`.
- `operations/render.mojo` does **not** compile. It targets the old custom-op
  API (`@compiler.register`, `tensor.InputTensor`, `runtime.asyncrt`), which has
  been replaced by `extensibility.register` / `extensibility.InputTensor`.
- The PLY loader for `assets/christmas_tree.ply` is not yet implemented.

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
