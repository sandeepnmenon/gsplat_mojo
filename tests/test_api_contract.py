from __future__ import annotations

import pytest
import torch

from gsplat_mojo import rasterization
from gsplat_mojo._kernels import _kernel_library_path


def _cpu_inputs(n: int = 1) -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros((n, 3), dtype=torch.float32),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).expand(n, -1),
        torch.ones((n, 3), dtype=torch.float32),
        torch.ones((n,), dtype=torch.float32),
        torch.ones((n, 3), dtype=torch.float32),
        torch.eye(4, dtype=torch.float32).unsqueeze(0),
        torch.tensor(
            [[[600.0, 0.0, 512.0], [0.0, 600.0, 384.0], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
        ),
    )


def _ray_call(*inputs: torch.Tensor, **kwargs: object) -> object:
    return rasterization(
        *inputs,
        1024,
        768,
        near_plane=0.2,
        renderer="ray",
        **kwargs,
    )


def test_ewa_default_is_explicitly_unavailable() -> None:
    with pytest.raises(
        NotImplementedError, match="renderer='ewa' is not available in Phase 0"
    ):
        rasterization(*_cpu_inputs(), 1024, 768)


def test_ray_rejects_upstream_near_plane_default() -> None:
    with pytest.raises(ValueError, match="near_plane must be 0.2"):
        rasterization(*_cpu_inputs(), 1024, 768, renderer="ray")


def test_ray_rejects_cpu_before_loading_kernels() -> None:
    with pytest.raises(RuntimeError, match="means must be a CUDA tensor"):
        _ray_call(*_cpu_inputs())


def test_inference_only_rejects_gradients() -> None:
    inputs = list(_cpu_inputs())
    inputs[0] = inputs[0].requires_grad_()
    with pytest.raises(RuntimeError, match="Phase 0 is inference-only"):
        _ray_call(*inputs)


def test_unsupported_features_are_not_ignored() -> None:
    with pytest.raises(NotImplementedError, match="backgrounds"):
        _ray_call(*_cpu_inputs(), backgrounds=torch.zeros((1, 3)))


def test_kernel_sources_are_packaged() -> None:
    path = _kernel_library_path()
    assert path.is_dir()
    assert (path / "__init__.mojo").is_file()
    assert (path / "render.mojo").is_file()
