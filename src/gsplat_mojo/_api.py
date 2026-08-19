from __future__ import annotations

from typing import Literal

import torch

from ._kernels import custom_ops
from ._validation import validate_inputs


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
    """Render one camera with the inference-only Phase 0 API.

    ``renderer="ewa"`` intentionally remains unavailable. Callers must opt
    into the experimental ray renderer and its fixed Phase 0 configuration.
    """

    (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
    ) = validate_inputs(
        (means, quats, scales, opacities, colors, viewmats, Ks),
        width=width,
        height=height,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        eps2d=eps2d,
        sh_degree=sh_degree,
        packed=packed,
        tile_size=tile_size,
        backgrounds=backgrounds,
        render_mode=render_mode,
        sparse_grad=sparse_grad,
        absgrad=absgrad,
        rasterize_mode=rasterize_mode,
        channel_chunk=channel_chunk,
        distributed=distributed,
        camera_model=camera_model,
        renderer=renderer,
    )

    # Public gsplat quaternions are wxyz. The Mojo kernel consumes normalized
    # xyzw values. Both operations stay on the caller's CUDA device.
    quat_norms = torch.linalg.vector_norm(quats, dim=-1, keepdim=True)
    quats_xyzw = (quats / quat_norms)[:, (1, 2, 3, 0)]

    with torch.inference_mode():
        render_colors = means.new_empty((1, 768, 1024, 3))
        render_alphas = means.new_empty((1, 768, 1024, 1))
        custom_ops().render(
            render_colors,
            render_alphas,
            means,
            colors,
            opacities,
            scales,
            quats_xyzw,
            viewmats,
            Ks,
        )

    meta: dict[str, object] = {
        "renderer": "ray",
        "width": width,
        "height": height,
        "tile_size": 16,
    }
    return render_colors, render_alphas, meta
