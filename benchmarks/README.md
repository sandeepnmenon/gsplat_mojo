# Phase 0 ray self-baseline

This directory measures the experimental `gsplat_mojo` ray renderer against
itself. It does not install, import, or compare against upstream gsplat. The
artifact format is defined by `phase0-ray-baseline.schema.json`.

## Reproduce

Run with the Python environment in which `gsplat-mojo`, its pinned MAX build,
and PyTorch are installed:

```bash
python benchmarks/phase0_ray.py \
  --gaussians 1024 \
  --warmups 5 \
  --iterations 20 \
  --output phase0-ray.json
python benchmarks/phase0_ray.py --validate phase0-ray.json
```

The workload is generated from a CPU `torch.Generator` with a recorded seed,
then copied to the selected CUDA device before measurement. It uses one
identity camera, 1024x768 output, activated float32 values, and the finalized
public call:

```python
gsplat_mojo.rasterization(..., near_plane=0.2, renderer="ray")
```

Pass `--help` for workload and device controls. `--print-schema` emits the
checked-in schema exactly. JSON is serialized with sorted keys and a trailing
newline so equal values produce byte-stable files.

## Timing policy

The top-level command starts a fresh Python worker with empty, isolated
`MODULAR_CACHE_DIR` and `MODULAR_MAX_CACHE_DIR` directories. The first
synchronized public API call records the cold-cache first-call time. Because
MAX compiles and dispatches through that same call, the artifact reports both
the measured total and an explicitly labeled JIT estimate:

`cold first call - median synchronized warm call`.

Warmup calls run before measured calls. Every measured latency starts after
`torch.cuda.synchronize()` and ends after another synchronization, so it is
host-observed end-to-end latency rather than asynchronous launch time.

The harness samples system-wide GPU memory through `nvidia-smi` every 20 ms and
also records PyTorch allocator peaks. The former includes MAX scratch but is
MiB-granular and can include unrelated processes; the latter is precise for
PyTorch-owned allocations but can omit MAX-owned scratch. Both scopes are
identified in the artifact.

## Honest unavailable fields

The finalized package API dispatches projection, binning, prefix scan, radix
sort, and rasterization as one atomic custom op. It does not return the actual
tile-intersection count or internal event timings. The stable schema therefore
contains:

- the exact input Gaussian count;
- an unavailable intersection count with a reason; and
- unavailable per-stage timings with a reason.

These fields must not be populated by estimates. Exposing them requires a
future package/kernel instrumentation contract; this benchmark intentionally
does not bypass or modify the public API.

## Interpreting results

This is a reproducibility and regression artifact for one experimental
renderer, not a performance claim. Compare artifacts only when workload,
schema, synchronization policy, MAX/Mojo/PyTorch versions, GPU, driver, and
source revision are compatible. The source record includes dirty-tree state
because development baselines may be captured before a commit.
