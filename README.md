# gsplat_mojo

3D Gaussian Splatting rendering kernels implemented in [Mojo](https://www.modular.com/mojo), targeting GPU acceleration via Modular's MAX platform.

> **Status:** Work-in-progress. The core rasterization kernel and math utilities are partially implemented. The code compiles and the GPU kernel launches successfully.

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
magic run forward
```

This is a project task defined in `mojoproject.toml` that handles include paths automatically. You can also run it manually:

```bash
magic run mojo run -I . operations/gsplat_forward.mojo
```

### Format code

```bash
magic run mblack .
```

## Current State

The project is in early development. The kernel compiles and launches on GPU, but the rasterization loop is not yet fully connected:

- `render.mojo` contains a simplified version of the kernel with a different signature (used as a `@compiler.register("render")` custom op). Its `gaussian_2d()` function is stubbed out.
- The inner accumulation loop in `gsplat_forward.mojo` reads gaussian data per-tile but does not yet compute the 2D projection or alpha-composite colors.
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
