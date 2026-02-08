# gsplat_mojo

3D Gaussian Splatting rendering kernels implemented in [Mojo](https://www.modular.com/mojo), targeting GPU acceleration via Modular's MAX platform.

> **Status:** Work-in-progress. The core rasterization kernel and math utilities are partially implemented. Several functions are incomplete and the code has known compilation errors (see [Current State](#current-state) below).

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
| `vec.mojo` | `Vec3` and `Vec4` structs with arithmetic, dot/cross product, normalization |
| `utils_t.mojo` | `rotation_matrix_to_quaternion`, `quat_to_rotmat`, `transpose`, `matmul3x3`, `g_scalar` |
| `operations/gsplat_forward.mojo` | Core GPU rasterization kernel with `SE3` transforms, shared memory tiling, and a `main()` entry point that allocates buffers and launches the kernel |
| `operations/render.mojo` | `@compiler.register("render")` custom op that dispatches to the GPU kernel |

## Prerequisites

- **Linux x86_64** (only platform currently supported)
- **GPU** with driver support for Modular MAX
- [Modular Magic CLI](https://docs.modular.com/magic/) (manages the Mojo toolchain and environment)

Install Magic:
```bash
curl -ssL https://magic.modular.com | bash
```

## Setup

Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-user>/gsplat_mojo.git
cd gsplat_mojo/gsplat
magic install
```

This reads `mojoproject.toml`, resolves dependencies (including the MAX/Mojo nightly toolchain), and creates the environment under `.magic/`.

## Running

All commands should be run from the `gsplat/` directory.

### Run the forward rasterization kernel

The main entry point is in `operations/gsplat_forward.mojo`. It allocates GPU buffers, constructs tensors, and launches the rasterization kernel:

```bash
cd gsplat
magic run mojo run operations/gsplat_forward.mojo
```

### Run individual modules (type-check)

To verify that a module parses/compiles without running it:

```bash
magic run mojo build vec.mojo
magic run mojo build utils_t.mojo
```

### Format code

```bash
magic run mblack .
```

## Current State

The project is in early development. Known issues that prevent a clean build:

1. **Module resolution** -- `gsplat_forward.mojo` imports `from vec import ...` and `from utils_t import ...`, but these modules live in the parent directory (`gsplat/`). Running from `operations/` can't resolve them. A package `__init__.mojo` or `-I` include path is needed.
2. **Undeclared variables** in `gsplat_forward.mojo` -- `xyz_ptr`, `iscl_rot`, `xyz_opac` are used before declaration in the kernel's inner loop (lines ~196-226).
3. **`RollingShutterParameters`** -- references `self.t_start` before it's assigned in the `__init__` (the field ordering issue on line 79).
4. **`gaussian_2d()`** in `render.mojo` -- declared but has no body.
5. **`render.mojo`** kernel signature -- the `rasterize_to_pixels_from_world_3dgs_fwd` in `render.mojo` has a different (simplified) signature than the one in `gsplat_forward.mojo`.
6. **`vec.mojo`** -- uses `random_Float32` which is not a standard Mojo function; needs to use `random.random_float64` or similar.

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
