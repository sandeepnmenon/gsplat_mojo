from __future__ import annotations

import pytest
import torch

from gsplat_mojo import rasterization

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_ray_renderer_preserves_cuda_residency_and_returns_alpha() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        means = torch.tensor([[0.0, 0.0, 2.0]], device=device)
        # Deliberately non-unit wxyz input exercises normalization and reordering.
        quats = torch.tensor([[2.0, 0.0, 0.0, 0.0]], device=device)
        scales = torch.tensor([[0.1, 0.1, 0.1]], device=device)
        opacities = torch.tensor([0.75], device=device)
        colors = torch.tensor([[0.25, 0.5, 0.75]], device=device)
        viewmats = torch.eye(4, device=device).unsqueeze(0)
        ks = torch.tensor(
            [[[600.0, 0.0, 512.0], [0.0, 600.0, 384.0], [0.0, 0.0, 1.0]]],
            device=device,
        )

        render_colors, render_alphas, meta = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            ks,
            1024,
            768,
            near_plane=0.2,
            renderer="ray",
        )
        completion = torch.cuda.Event()
        completion.record()

    # Consume the outputs from the default stream through an explicit event
    # dependency. This exercises non-default-stream dispatch and ordering
    # without relying on pointer equality or a device-wide synchronize.
    torch.cuda.current_stream(device).wait_event(completion)

    assert render_colors.shape == (1, 768, 1024, 3)
    assert render_alphas.shape == (1, 768, 1024, 1)
    assert render_colors.device == device
    assert render_alphas.device == device
    assert render_colors.dtype is torch.float32
    assert render_alphas.dtype is torch.float32
    assert render_colors.is_contiguous()
    assert render_alphas.is_contiguous()
    assert not render_colors.requires_grad
    assert not render_alphas.requires_grad
    assert bool(torch.isfinite(render_colors).all().item())
    assert bool(torch.isfinite(render_alphas).all().item())
    assert bool((render_alphas > 0).any().item())
    assert meta == {
        "renderer": "ray",
        "width": 1024,
        "height": 768,
        "tile_size": 16,
    }
