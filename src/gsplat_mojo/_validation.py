from __future__ import annotations

from collections.abc import Sequence

import torch

_TENSOR_SPECS: tuple[tuple[str, tuple[int | str, ...]], ...] = (
    ("means", ("N", 3)),
    ("quats", ("N", 4)),
    ("scales", ("N", 3)),
    ("opacities", ("N",)),
    ("colors", ("N", 3)),
    ("viewmats", (1, 4, 4)),
    ("Ks", (1, 3, 3)),
)


def _check_exact_type(name: str, value: object, expected: type) -> None:
    if type(value) is not expected:
        raise TypeError(
            f"{name} must be {expected.__name__}, got {type(value).__name__}"
        )


def _check_tensor(
    name: str,
    value: object,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if value.dtype is not torch.float32:
        raise ValueError(f"{name} must have dtype torch.float32, got {value.dtype}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if value.requires_grad:
        raise RuntimeError(
            f"{name} requires gradients, but gsplat-mojo Phase 0 is inference-only"
        )
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor, got device {value.device}")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def validate_inputs(
    tensors: Sequence[object],
    *,
    width: object,
    height: object,
    near_plane: object,
    far_plane: object,
    radius_clip: object,
    eps2d: object,
    sh_degree: object,
    packed: object,
    tile_size: object,
    backgrounds: object,
    render_mode: object,
    sparse_grad: object,
    absgrad: object,
    rasterize_mode: object,
    channel_chunk: object,
    distributed: object,
    camera_model: object,
    renderer: object,
) -> tuple[torch.Tensor, ...]:
    if type(renderer) is not str:
        raise TypeError(f"renderer must be str, got {type(renderer).__name__}")
    if renderer == "ewa":
        raise NotImplementedError("renderer='ewa' is not available in Phase 0")
    if renderer != "ray":
        raise NotImplementedError(
            f"renderer={renderer!r} is not available; Phase 0 supports only 'ray'"
        )

    for name, value in (("width", width), ("height", height)):
        _check_exact_type(name, value, int)
    for name, value in (
        ("near_plane", near_plane),
        ("far_plane", far_plane),
        ("radius_clip", radius_clip),
        ("eps2d", eps2d),
    ):
        _check_exact_type(name, value, float)
    for name, value in (
        ("packed", packed),
        ("sparse_grad", sparse_grad),
        ("absgrad", absgrad),
        ("distributed", distributed),
    ):
        _check_exact_type(name, value, bool)
    for name, value in (
        ("render_mode", render_mode),
        ("rasterize_mode", rasterize_mode),
        ("camera_model", camera_model),
    ):
        _check_exact_type(name, value, str)
    _check_exact_type("channel_chunk", channel_chunk, int)
    if tile_size is not None:
        _check_exact_type("tile_size", tile_size, int)
    if sh_degree is not None:
        raise NotImplementedError(
            f"sh_degree={sh_degree!r} is not available in Phase 0"
        )
    if backgrounds is not None:
        raise NotImplementedError("backgrounds is not available in Phase 0")

    expected_options = (
        ("width", width, 1024),
        ("height", height, 768),
        ("near_plane", near_plane, 0.2),
        ("far_plane", far_plane, 1e10),
        ("radius_clip", radius_clip, 0.0),
        ("eps2d", eps2d, 0.3),
        ("packed", packed, True),
        ("render_mode", render_mode, "RGB"),
        ("sparse_grad", sparse_grad, False),
        ("absgrad", absgrad, False),
        ("rasterize_mode", rasterize_mode, "classic"),
        ("channel_chunk", channel_chunk, 32),
        ("distributed", distributed, False),
        ("camera_model", camera_model, "pinhole"),
    )
    for name, received, expected in expected_options:
        if received != expected:
            raise ValueError(
                f"{name} must be {expected!r} for renderer='ray', got {received!r}"
            )
    if tile_size not in (None, 16):
        raise ValueError(
            f"tile_size must be None or 16 for renderer='ray', got {tile_size!r}"
        )

    if len(tensors) != len(_TENSOR_SPECS):
        raise RuntimeError("internal error: unexpected tensor argument count")
    means_obj = tensors[0]
    if not isinstance(means_obj, torch.Tensor):
        raise TypeError(f"means must be a torch.Tensor, got {type(means_obj).__name__}")
    if means_obj.ndim != 2:
        raise ValueError(f"means must have shape (N, 3), got {tuple(means_obj.shape)}")
    n = means_obj.shape[0]
    if not 1 <= n <= 400_000:
        raise ValueError(f"means N must be in [1, 400000], got {n}")

    validated: list[torch.Tensor] = []
    for (name, symbolic_shape), value in zip(_TENSOR_SPECS, tensors, strict=True):
        shape = tuple(n if dim == "N" else dim for dim in symbolic_shape)
        validated.append(_check_tensor(name, value, shape))

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA accelerator is available")
    devices = {tensor.device for tensor in validated}
    if len(devices) != 1:
        detail = ", ".join(
            f"{name}={tensor.device}"
            for (name, _), tensor in zip(_TENSOR_SPECS, validated, strict=True)
        )
        raise RuntimeError(f"all tensors must be on the same CUDA device; got {detail}")

    means, quats, scales, opacities, colors, viewmats, ks = validated
    del means, colors
    if not bool((scales > 0).all().item()):
        raise ValueError("scales must contain strictly positive activated values")
    if not bool(((opacities >= 0) & (opacities <= 1)).all().item()):
        raise ValueError("opacities must contain activated values in [0, 1]")
    quat_norms = torch.linalg.vector_norm(quats, dim=-1)
    if not bool((quat_norms > 0).all().item()):
        raise ValueError("quats must not contain zero-length quaternions")

    expected_view_row = viewmats.new_tensor([0.0, 0.0, 0.0, 1.0])
    if not bool(torch.equal(viewmats[0, 3], expected_view_row)):
        raise ValueError(
            "viewmats must be world-to-camera transforms with final row [0, 0, 0, 1]"
        )
    expected_k_row = ks.new_tensor([0.0, 0.0, 1.0])
    if not bool(torch.equal(ks[0, 2], expected_k_row)):
        raise ValueError("Ks must have final row [0, 0, 1]")
    if bool((ks[0, 0, 1] != 0).item()) or bool((ks[0, 1, 0] != 0).item()):
        raise ValueError("Ks must be canonical pinhole matrices with zero skew")
    if not bool(((ks[0, 0, 0] > 0) & (ks[0, 1, 1] > 0)).item()):
        raise ValueError("Ks focal lengths fx and fy must be positive")

    return tuple(validated)
