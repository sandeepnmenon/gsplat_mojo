# gsplat-mojo

Inference-first PyTorch bindings for experimental 3D Gaussian rendering kernels
written in [Mojo](https://www.modular.com/mojo) and loaded through MAX.

> **Pre-alpha (`0.1.0a0` candidate; not released).** The labels below are part
> of every capability claim:
>
> - **Tested — Tier 1:** exercised on the NVIDIA/Linux configuration recorded
>   in the [validation baseline](docs/validation-baseline.md).
> - **Best effort — Tier 2:** expected to use MAX portability, but not verified
>   by this project and not a release guarantee.
> - **Planned:** unavailable today; see the [roadmap](ROADMAP.md).

## Five-minute PyTorch quickstart

**Tested — Tier 1 renderer path:** From a checkout on Linux x86_64 with a
supported NVIDIA driver, [uv](https://docs.astral.sh/uv/), Python 3.11–3.13,
and access to the pinned Modular nightly:

```bash
git clone https://github.com/sandeepnmenon/gsplat_mojo.git
cd gsplat_mojo
uv sync --extra test
uv run python driver.py
```

The first call JIT-compiles the Mojo custom operation and can take longer than
later calls. A successful run prints CUDA-resident color and alpha shapes,
maximum alpha, and renderer metadata. The smoke program uses one Gaussian and
the only currently available mode:

```python
render_colors, render_alphas, meta = rasterization(
    means,
    quats,       # wxyz
    scales,      # activated, positive
    opacities,   # activated, [0, 1]
    colors,      # activated linear RGB
    viewmats,
    Ks,
    1024,
    768,
    near_plane=0.2,
    renderer="ray",
)
```

**Tested — Tier 1:** Inputs are contiguous `float32` CUDA tensors on one device;
outputs are contiguous `float32` CUDA tensors with shapes
`[1, 768, 1024, 3]` and `[1, 768, 1024, 1]`. Gradients, CPU fallback, multiple
cameras, dynamic resolution, and implicit conversion are rejected. See the
complete [API contract](docs/api-contract.md) and
[installation guide](docs/installation.md).

## What exists today

| Capability | Status | Scope |
|---|---|---|
| Experimental ray/Gaussian forward renderer | **Tested — Tier 1** | One camera, 1024×768, RGB, `float32`, CUDA, inference only |
| PyTorch → MAX custom-op path | **Tested — Tier 1** | Caller-owned CUDA inputs; CUDA color and alpha outputs |
| Projection, tile binning, LSD radix depth sort, compositing | **Tested — Tier 1** | Radix is the active path; bitonic is a test cross-check |
| Native SH evaluation, degrees 0–3 | **Tested — Tier 1** | Native Mojo check; public PyTorch call accepts activated RGB only |
| Other MAX accelerator backends | **Best effort — Tier 2** | Unverified; no compatibility or correctness guarantee |
| gsplat-compatible EWA renderer | **Planned** | The default `renderer="ewa"` currently raises `NotImplementedError` |
| Autograd/training | **Planned** | Inputs with `requires_grad=True` are rejected |
| Dynamic sizes, batches, and broader camera models | **Planned** | Fixed Phase 0 contract only |
| Upstream performance comparison | **Planned** | Blocked until semantically equivalent EWA output exists |

This package is not a drop-in replacement for upstream
[gsplat](https://github.com/nerfstudio-project/gsplat). The public signature
recognizes a deliberately small inference subset, while the available
`renderer="ray"` mode uses different shading semantics.

## Rendered example

**Tested — Tier 1:** The native real-scene check rendered 329,004 Gaussians at
1024×768 and passed the sampled float64-reference gate documented in the
[validation baseline](docs/validation-baseline.md).

![Experimental Christmas tree scene render](assets/render_preview.png)

**Release blocker:** the original source, creator, and license of
`assets/christmas_tree.ply` have not been established. The preview is a
derivative render and therefore has the same unresolved redistribution risk.
Do not treat either asset as MIT-licensed. See
[asset provenance and remediation](assets/README.md).

## Architecture

**Tested — Tier 1:** The current forward path is:

```text
PyTorch CUDA tensors
  → MAX CustomOpLibrary / DLPack-compatible exchange
  → projection and tile counting
  → two-level prefix scan
  → tile/depth key emission
  → 4-bit LSD radix sort
  → tile offsets
  → experimental ray/Gaussian rasterization
  → PyTorch CUDA color + alpha tensors
```

PyTorch allocates the outputs before dispatch. The implementation does not use
a raw `Tensor.data_ptr()` bridge. The current scan and renderer have fixed
capacity/configuration limits described by the [API contract](docs/api-contract.md).

## Reproduce validation

**Tested — Tier 1:** The focused native correctness baseline was produced with:

```bash
cd gsplat
pixi run intersect
pixi run forward
pixi run render-ply
```

Those checks cover radix-versus-bitonic ordering, synthetic scalar/closed-form
references, and a sampled real-scene float64 comparison. Exact hardware,
toolchain, counts, and errors are in
[`docs/validation-baseline.md`](docs/validation-baseline.md).

**Tested — Tier 1:** The machine-readable ray self-baseline harness and schema
are checked in under [`benchmarks/`](benchmarks/). Machine-specific result
artifacts are not part of this pre-release source tree. Produce and validate a
local artifact with:

```bash
uv run python benchmarks/phase0_ray.py \
  --gaussians 1024 \
  --warmups 5 \
  --iterations 20 \
  --output phase0-ray.json
uv run python benchmarks/phase0_ray.py --validate phase0-ray.json
```

The artifact records cold first-call/JIT estimates, synchronized warm
end-to-end latency, device-memory scopes, workload counts, and environment
metadata; unavailable internal stage/intersection data remains explicitly
unavailable. See [`benchmarks/README.md`](benchmarks/README.md). Phase 0 will
report only a ray self-baseline; fair cross-library comparisons are
**Planned** after EWA parity.

## Install and develop

- **Tested — Tier 1:** End-user checkout and wheel workflows are documented in
  [`docs/installation.md`](docs/installation.md).
- **Tested — Tier 1:** Contributors use pixi for native Mojo checks and uv for
  the Python package/tests; see [`CONTRIBUTING.md`](CONTRIBUTING.md).
- **Best effort — Tier 2:** Other MAX targets may work, but reports must include
  full environment details and are handled without a response-time guarantee.
- **Planned:** Published package installation begins only after the pre-alpha
  release candidate passes its gates and is explicitly approved.

## Project policies

- [Support and feature tiers](SUPPORT.md)
- [Roadmap](ROADMAP.md)
- [Contributing](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security policy](SECURITY.md)
- [Changelog](CHANGELOG.md)
- [Citation metadata](CITATION.cff)
- [Acknowledgements](ACKNOWLEDGEMENTS.md)
- [Asset provenance](assets/README.md)

## License

Project-authored code and documentation are available under the
[MIT License](LICENSE). That license does **not** establish rights to
third-party or provenance-unknown assets. In particular,
`assets/christmas_tree.ply` and `assets/render_preview.png` remain blocked from
release redistribution until the remediation in
[`assets/README.md`](assets/README.md) is complete.
