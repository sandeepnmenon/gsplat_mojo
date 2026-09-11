"""Package-facing smoke example for the experimental Phase 0 ray renderer.

Install the project first, then run ``python driver.py`` from the repository
root. The public API keeps all inputs and outputs on the active CUDA device.
"""

from __future__ import annotations

import torch

from gsplat_mojo import rasterization


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("gsplat-mojo Phase 0 requires a CUDA accelerator")

    device = torch.device("cuda", torch.cuda.current_device())
    means = torch.tensor([[0.0, 0.0, 2.0]], device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)  # wxyz
    scales = torch.tensor([[0.1, 0.1, 0.1]], device=device)
    opacities = torch.tensor([0.75], device=device)
    colors = torch.tensor([[0.25, 0.5, 0.75]], device=device)
    viewmats = torch.eye(4, device=device).unsqueeze(0)
    Ks = torch.tensor(
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
        Ks,
        1024,
        768,
        near_plane=0.2,
        renderer="ray",
    )
    torch.cuda.synchronize(device)
    print(
        f"colors={tuple(render_colors.shape)} {render_colors.device}; "
        f"alphas={tuple(render_alphas.shape)}; "
        f"max_alpha={render_alphas.max().item():.6f}; meta={meta}"
    )


if __name__ == "__main__":
    main()
