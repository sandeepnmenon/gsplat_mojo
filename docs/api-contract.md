# Phase 0 inference API contract

## Claim classification

Unless a paragraph says otherwise, implemented ray-mode behavior in this
document is **Tested — Tier 1** on the environment in
[`validation-baseline.md`](validation-baseline.md). `renderer="ewa"` and other
future behavior are **Planned**. Other MAX accelerators are
**Best effort — Tier 2** and are not implied by this contract.

This document is the implementation contract for the first `gsplat-mojo`
PyTorch package slice. It deliberately describes less than upstream
[`gsplat.rasterization`](https://docs.gsplat.studio/main/apis/rasterization.html):
only the inference inputs that the current renderer can implement honestly are
accepted. Training, EWA rendering, and dynamic dimensions are not implied.

## Public entry point

The distribution is `gsplat-mojo`, the import package is `gsplat_mojo`, and the
public function is:

```python
from typing import Literal

def rasterization(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sh_degree: int | None = None,
    packed: bool = True,
    tile_size: int | None = None,
    backgrounds: torch.Tensor | None = None,
    render_mode: Literal["RGB"] = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    rasterize_mode: Literal["classic"] = "classic",
    channel_chunk: int = 32,
    distributed: bool = False,
    camera_model: Literal["pinhole"] = "pinhole",
    *,
    renderer: Literal["ewa", "ray"] = "ewa",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    ...
```

`renderer` is the only gsplat-mojo-specific argument. All other named
arguments above correspond to documented upstream arguments. Unknown keyword
arguments must be rejected rather than ignored.

## Renderer modes

- **Planned:** `renderer="ewa"` is the stable, future-compatible default. In
  Phase 0 it
  raises `NotImplementedError("renderer='ewa' is not available in Phase 0")`.
  It must never silently select the ray renderer.
- **Tested — Tier 1:** `renderer="ray"` explicitly opts into the existing experimental
  ray/Gaussian renderer. It uses EWA projection only to conservatively choose
  tiles; pixel shading is a 3D ray/Gaussian evaluation and is not numerically
  equivalent to upstream gsplat EWA rasterization.
- Upstream `render_mode` controls output channels and is separate from
  `renderer`. Phase 0 accepts only `"RGB"`.
- Upstream `rasterize_mode` controls classic versus antialiased EWA behavior
  and is also separate from `renderer`. Phase 0 accepts only `"classic"`.

This separation lets Phase 1 implement the default EWA path without changing
the call shape or reinterpreting an existing option.

## Accepted tensors

Phase 0 accepts one unbatched Gaussian set and one camera:

| Input | Shape | Meaning |
|---|---|---|
| `means` | `[N, 3]` | finite world-space centers |
| `quats` | `[N, 4]` | finite rotations in upstream **wxyz** order |
| `scales` | `[N, 3]` | finite, strictly positive, activated linear scales |
| `opacities` | `[N]` | finite, activated values in `[0, 1]` |
| `colors` | `[N, 3]` | finite, post-activation linear RGB |
| `viewmats` | `[1, 4, 4]` | world-to-camera transforms |
| `Ks` | `[1, 3, 3]` | pinhole intrinsics |

All tensors must:

- have dtype `torch.float32`;
- be contiguous in row-major order;
- be CUDA tensors on the same device;
- have `requires_grad == False`; and
- remain on that device through the custom-op boundary.

No implicit dtype conversion, `.contiguous()`, CPU staging, or device transfer
is part of the API. `1 <= N <= 400_000`; leading batch dimensions, multiple
cameras, and broadcasted inputs are rejected in Phase 0.

Quaternions intentionally follow upstream gsplat's wxyz convention and need
not arrive normalized. The Python/custom-op boundary must reject zero-length
or non-finite quaternions, normalize each quaternion, then reorder it to the
current Mojo kernel's internal xyzw convention. The internal convention must
not leak into the public API.

`viewmats` use `x_camera = R @ x_world + t` and must have final row
`[0, 0, 0, 1]`. `Ks` must be the canonical pinhole matrix
`[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]` with positive `fx` and `fy`.

Colors and opacities are already activated. The package does not apply sigmoid,
SH activation, or exponential scale activation. RGB values are not clamped by
the renderer; display conversion may clamp them afterward.

## Phase 0 ray option values

A successful experimental ray call additionally requires:

- `width == 1024` and `height == 768`;
- `near_plane == 0.2` (the current compiled kernel constant);
- `far_plane == 1e10` and `radius_clip == 0.0`;
- `eps2d == 0.3` (used for the conservative tile bound, not shading);
- `sh_degree is None`, `packed is True`;
- `tile_size in (None, 16)`, where `None` resolves to `16`;
- `backgrounds is None` (black);
- `render_mode == "RGB"`, `rasterize_mode == "classic"`;
- `channel_chunk == 32`, `distributed is False`;
- `camera_model == "pinhole"`; and
- `sparse_grad is False`, `absgrad is False`.

The upstream default `near_plane=0.01` is retained in the signature for API
recognition, but the experimental Phase 0 ray renderer must reject it and name
the supported value. Callers therefore opt into both semantics explicitly:

```python
render_colors, render_alphas, meta = rasterization(
    means,
    quats,
    scales,
    opacities,
    colors,
    viewmats,
    Ks,
    1024,
    768,
    near_plane=0.2,
    renderer="ray",
)
```

The documented upstream options not present in the signature—depth modes,
covariances, distortion, rolling shutter, alternate cameras, distributed
rendering, extra signals, normals, UT/eval3d, and training controls—are not
Phase 0 features. If supplied through a compatibility wrapper, they must raise
`NotImplementedError` naming the first unsupported option.

## Outputs

The return structure mirrors upstream:

- `render_colors`: contiguous `float32`, shape `[1, 768, 1024, 3]`, on the
  input CUDA device;
- `render_alphas`: contiguous `float32`, shape `[1, 768, 1024, 1]`, on the
  input CUDA device; and
- `meta`: a dictionary containing at least `renderer`, `width`, `height`, and
  `tile_size`.

`meta` does not promise upstream training intermediates such as `means2d`,
`radii`, or gradient state in Phase 0. Output tensors are inference-only and
must not acquire an autograd function.

## Validation and errors

Validation happens before JIT compilation or dispatch where possible:

- non-tensor inputs or wrong Python scalar types: `TypeError`;
- shape, contiguity, finite-value, range, camera-matrix, or option violations:
  `ValueError` naming the argument and received value/shape;
- CPU tensors, mixed devices, or unavailable accelerator: `RuntimeError`;
- any `requires_grad=True` input: `RuntimeError` explaining that Phase 0 is
  inference-only;
- unavailable renderer/features: `NotImplementedError`; and
- configured capacity overflow: `RuntimeError` naming the relevant limit.

The implementation must not use assertions for user input validation and must
not silently coerce, detach, copy, clamp, or fall back to a different renderer.

## Handoff invariants

The custom-op agent must expose both color and alpha outputs, preserve
caller-owned CUDA storage, normalize/reorder quaternions at the boundary, and
keep the radix sort as the production sort. The package agent must implement
the validation above before loading/compiling the op and must keep EWA as an
explicitly unavailable default until a semantically compatible implementation
exists.
