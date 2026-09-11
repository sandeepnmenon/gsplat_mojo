from __future__ import annotations

import os
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

_KERNEL_OVERRIDE = "GSPLAT_MOJO_KERNEL_LIBRARY"


def _kernel_library_path() -> Path:
    override = os.environ.get(_KERNEL_OVERRIDE)
    if override:
        path = Path(override).expanduser().resolve()
        if not path.exists():
            raise RuntimeError(f"{_KERNEL_OVERRIDE} does not exist: {path}")
        return path

    packaged = Path(str(files("gsplat_mojo").joinpath("kernels/gsplat_kernels")))
    if packaged.is_dir():
        return packaged

    # Source-tree fallback for contributors. Wheels always use the packaged
    # path populated by pyproject.toml's force-include rule.
    checkout = Path(__file__).resolve().parents[2] / "gsplat" / "gsplat_kernels"
    if checkout.is_dir():
        return checkout
    raise RuntimeError(
        "gsplat-mojo kernel sources are missing; reinstall the package from "
        "an official wheel or sdist"
    )


@lru_cache(maxsize=1)
def custom_ops() -> Any:
    try:
        from max.experimental.torch import CustomOpLibrary
    except ImportError as error:
        raise RuntimeError(
            "MAX's PyTorch bridge is unavailable; install the pinned "
            "'max' and 'torch' requirements from their configured indexes"
        ) from error
    return CustomOpLibrary(_kernel_library_path())
