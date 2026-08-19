# Installing the Phase 0 package

Unless marked otherwise, workflows in this document are
**Tested — Tier 1** on the NVIDIA/Linux baseline. Other MAX accelerators are
**Best effort — Tier 2**. Published-index installation is **Planned** because
`0.1.0a0` has not been released.

The 2026-08-18 release-candidate check resolved the earlier fresh-install
failure by allowing transitive MAX packages to resolve from the configured
Modular nightly index. Both `uv sync` resolution and installation of the built
wheel into a new Python 3.13 environment completed with the pinned MAX build.

`gsplat-mojo` is a Linux GPU package. The Phase 0 dependency set is pinned to
the MAX nightly used by the validated renderer baseline. PyTorch is constrained
to the matching supported minor release.

## Five-minute checkout with uv

The repository's `pyproject.toml` and `uv.lock` declare the resolved dependency
set and Modular index. Install a checkout with:

```bash
uv sync --extra test
uv run python driver.py
```

To install the built wheel in a clean uv environment:

```bash
uv venv --python 3.13 .venv
uv pip install \
  --python .venv/bin/python \
  --index https://whl.modular.com/nightly/simple/ \
  --prerelease allow \
  dist/gsplat_mojo-0.1.0a0-py3-none-any.whl
.venv/bin/python driver.py
```

## pip after a release

**Planned:** After a package is published, pip will need the official Modular
index and prerelease opt-in:

```bash
python -m pip install \
  --extra-index-url https://whl.modular.com/nightly/simple/ \
  --pre \
  gsplat-mojo
```

This command is not expected to succeed before publication. It is not a
current installation claim.

## Kernel loading and cache

The wheel contains the Mojo source package. On the first ray-renderer call,
MAX `CustomOpLibrary` JIT-compiles and loads the registered `render` operation;
the direct PyTorch bridge uses DLPack-compatible tensor exchange and does not
use `Tensor.data_ptr()`. PyTorch allocates color and alpha outputs on the input
CUDA device before dispatch.

MAX owns its compilation cache and reuses it on subsequent compatible runs.
Cache location and invalidation follow the installed MAX release; they are not
part of the `gsplat-mojo` API. Changing the MAX build, kernel sources, target
GPU, or compilation configuration can trigger recompilation. For contributor
diagnostics only, `GSPLAT_MOJO_KERNEL_LIBRARY` may point to a Mojo package
directory or precompiled kernel library.

pixi remains the contributor workflow for native Mojo checks:

```bash
cd gsplat
pixi run package
pixi run forward
pixi run intersect
```

End users do not need pixi and do not manually run `mojo precompile`.

## Phase 0 constraints

The default `renderer="ewa"` deliberately raises `NotImplementedError`.
Successful calls must explicitly use `renderer="ray"` and satisfy
`docs/api-contract.md`, including fixed 1024×768 output, one camera, float32
contiguous CUDA tensors, activated scales/opacities/colors, and no gradients.
