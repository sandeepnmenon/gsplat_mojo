"""Reproducible Phase 0 self-baseline for the experimental ray renderer."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from typing import Any, Self

SCHEMA_VERSION = "1.0.0"
SCHEMA_PATH = pathlib.Path(__file__).with_name("phase0-ray-baseline.schema.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the gsplat_mojo Phase 0 ray renderer against itself. "
            "No upstream gsplat code is imported or compared."
        )
    )
    parser.add_argument("--output", default="-", help="JSON path, or '-' for stdout")
    parser.add_argument("--gaussians", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--cold-cache-dir",
        type=pathlib.Path,
        help="empty directory to retain and use for the isolated MAX cache",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--print-schema", action="store_true", help="print the JSON Schema"
    )
    modes.add_argument(
        "--validate", type=pathlib.Path, metavar="JSON", help="validate an artifact"
    )
    parser.add_argument("--_worker-result", type=pathlib.Path, help=argparse.SUPPRESS)
    return parser


def _validate_cli(args: argparse.Namespace) -> None:
    if not 1 <= args.gaussians <= 400_000:
        raise SystemExit("--gaussians must be in [1, 400000]")
    if args.warmups < 0:
        raise SystemExit("--warmups must be non-negative")
    if args.iterations < 1:
        raise SystemExit("--iterations must be positive")
    if args.device < 0:
        raise SystemExit("--device must be non-negative")


def _canonical_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json(value: object, destination: str | pathlib.Path) -> None:
    text = _canonical_json(value)
    if str(destination) == "-":
        sys.stdout.write(text)
        return
    path = pathlib.Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _validate_value(
    value: object, schema: dict[str, Any], path: str, errors: list[str]
) -> None:
    expected = schema.get("type")
    expected_types = [expected] if isinstance(expected, str) else expected
    if expected_types:
        predicates = {
            "array": lambda item: isinstance(item, list),
            "boolean": lambda item: isinstance(item, bool),
            "integer": lambda item: (
                isinstance(item, int) and not isinstance(item, bool)
            ),
            "null": lambda item: item is None,
            "number": lambda item: (
                isinstance(item, (int, float)) and not isinstance(item, bool)
            ),
            "object": lambda item: isinstance(item, dict),
            "string": lambda item: isinstance(item, str),
        }
        if not any(predicates[kind](value) for kind in expected_types):
            errors.append(f"{path} must have type {expected_types}")
            return
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path} must equal {schema['const']!r}")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}.{name} is required")
        if schema.get("additionalProperties") is False:
            for name in sorted(value.keys() - properties.keys()):
                errors.append(f"{path}.{name} is not allowed")
        for name, child_schema in properties.items():
            if name in value:
                _validate_value(value[name], child_schema, f"{path}.{name}", errors)
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            _validate_value(item, schema["items"], f"{path}[{index}]", errors)
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and "minimum" in schema
        and value < schema["minimum"]
    ):
        errors.append(f"{path} must be >= {schema['minimum']}")
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and "maximum" in schema
        and value > schema["maximum"]
    ):
        errors.append(f"{path} must be <= {schema['maximum']}")


def _validate_artifact(data: object) -> list[str]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors: list[str] = []
    _validate_value(data, schema, "$", errors)
    return errors


def _command_output(command: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _git_metadata() -> dict[str, object]:
    root = pathlib.Path(__file__).resolve().parents[1]
    revision = _command_output(["git", "-C", str(root), "rev-parse", "HEAD"])
    status = _command_output(["git", "-C", str(root), "status", "--porcelain"])
    return {
        "revision": revision,
        "working_tree_dirty": bool(status) if status is not None else None,
    }


class _DeviceMemoryMonitor:
    """Sample system-wide device use without synchronizing the measured stream."""

    def __init__(self, device: int, interval_ms: int = 20) -> None:
        self._device = device
        self._interval_ms = interval_ms
        self._samples_mib: list[int] = []
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Self:
        command = [
            "nvidia-smi",
            f"--id={self._device}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-lms",
            str(self._interval_ms),
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except FileNotFoundError:
            return self

        def collect() -> None:
            assert self._process is not None
            assert self._process.stdout is not None
            for line in self._process.stdout:
                try:
                    self._samples_mib.append(int(line.strip()))
                except ValueError:
                    continue

        self._thread = threading.Thread(target=collect, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 2.0
        while not self._samples_mib and time.monotonic() < deadline:
            time.sleep(0.005)
        return self

    def __exit__(self, *_: object) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def result(self) -> dict[str, object]:
        if not self._samples_mib:
            return {
                "available": False,
                "reason": "nvidia-smi sampling produced no values",
            }
        baseline = self._samples_mib[0]
        peak = max(self._samples_mib)
        return {
            "available": True,
            "scope": "system-wide GPU memory used",
            "sampling_interval_ms": self._interval_ms,
            "sample_count": len(self._samples_mib),
            "baseline_bytes": baseline * 1024 * 1024,
            "peak_bytes": peak * 1024 * 1024,
            "peak_delta_bytes": (peak - baseline) * 1024 * 1024,
            "resolution_bytes": 1024 * 1024,
        }


def _summarize(samples_ms: list[float]) -> dict[str, object]:
    ordered = sorted(samples_ms)

    def percentile(fraction: float) -> float:
        index = (len(ordered) - 1) * fraction
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {
        "unit": "ms",
        "samples_ms": samples_ms,
        "count": len(samples_ms),
        "min_ms": min(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "p95_ms": percentile(0.95),
        "max_ms": max(samples_ms),
        "stdev_ms": statistics.pstdev(samples_ms),
    }


def _run_worker(args: argparse.Namespace) -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.device >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {args.device} is unavailable; "
            f"found {torch.cuda.device_count()} device(s)"
        )

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    n = args.gaussians
    means = torch.empty((n, 3), dtype=torch.float32)
    means[:, :2] = torch.rand((n, 2), generator=generator) * 2.0 - 1.0
    means[:, 2] = torch.rand((n,), generator=generator) * 4.0 + 2.0
    quats = torch.zeros((n, 4), dtype=torch.float32)
    quats[:, 0] = 1.0
    scales = torch.rand((n, 3), generator=generator) * 0.04 + 0.01
    opacities = torch.rand((n,), generator=generator) * 0.5 + 0.25
    colors = torch.rand((n, 3), generator=generator)
    viewmats = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    intrinsics = torch.tensor(
        [[[600.0, 0.0, 512.0], [0.0, 600.0, 384.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    tensors = [
        tensor.to(device)
        for tensor in (
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            intrinsics,
        )
    ]
    (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        intrinsics,
    ) = tensors

    from gsplat_mojo import __version__ as gsplat_mojo_version
    from gsplat_mojo import rasterization

    def render() -> tuple[Any, Any, dict[str, object]]:
        return rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            intrinsics,
            1024,
            768,
            near_plane=0.2,
            renderer="ray",
        )

    torch.cuda.synchronize(device)
    cold_torch_start = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    with _DeviceMemoryMonitor(args.device) as cold_monitor:
        cold_start = time.perf_counter_ns()
        render_colors, render_alphas, meta = render()
        torch.cuda.synchronize(device)
        cold_ms = (time.perf_counter_ns() - cold_start) / 1_000_000
    cold_torch_peak = torch.cuda.max_memory_allocated(device)

    for _ in range(args.warmups):
        render_colors, render_alphas, meta = render()
    torch.cuda.synchronize(device)

    warm_torch_start = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    warm_samples_ms: list[float] = []
    with _DeviceMemoryMonitor(args.device) as warm_monitor:
        for _ in range(args.iterations):
            torch.cuda.synchronize(device)
            start = time.perf_counter_ns()
            render_colors, render_alphas, meta = render()
            torch.cuda.synchronize(device)
            warm_samples_ms.append((time.perf_counter_ns() - start) / 1_000_000)
    warm_torch_peak = torch.cuda.max_memory_allocated(device)
    warm_summary = _summarize(warm_samples_ms)

    color_sum = float(render_colors.double().sum().item())
    alpha_sum = float(render_alphas.double().sum().item())
    max_alpha = float(render_alphas.max().item())
    output_digest = hashlib.sha256(
        (
            f"{color_sum:.12e}|{alpha_sum:.12e}|{max_alpha:.12e}|"
            f"{tuple(render_colors.shape)}|{tuple(render_alphas.shape)}"
        ).encode()
    ).hexdigest()

    properties = torch.cuda.get_device_properties(device)
    nvidia_query = _command_output(
        [
            "nvidia-smi",
            f"--id={args.device}",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    nvidia_fields = (
        [field.strip() for field in nvidia_query.split(",")]
        if nvidia_query is not None
        else []
    )
    mojo_executable = pathlib.Path(sys.executable).with_name("mojo")
    max_executable = pathlib.Path(sys.executable).with_name("max")
    mojo_version = _command_output([str(mojo_executable), "--version"])
    max_cli_version = _command_output([str(max_executable), "--version"])
    max_version = importlib.metadata.version("max")
    torch_version = torch.__version__
    warm_median = float(warm_summary["median_ms"])

    return {
        "$schema": "../phase0-ray-baseline.schema.json",
        "schema_version": SCHEMA_VERSION,
        "benchmark": {
            "name": "gsplat-mojo-phase0-ray-self-baseline",
            "renderer": "ray",
            "comparison": "self-baseline-only",
            "upstream_gsplat_used": False,
        },
        "captured_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "environment": {
            "system": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
            "gpu": {
                "device_index": args.device,
                "name": properties.name,
                "compute_capability": (f"{properties.major}.{properties.minor}"),
                "total_memory_bytes": properties.total_memory,
                "uuid": nvidia_fields[1] if len(nvidia_fields) == 4 else None,
                "driver_version": (
                    nvidia_fields[2] if len(nvidia_fields) == 4 else None
                ),
                "nvidia_smi_total_memory_mib": (
                    int(nvidia_fields[3]) if len(nvidia_fields) == 4 else None
                ),
            },
            "software": {
                "gsplat_mojo": gsplat_mojo_version,
                "torch": torch_version,
                "torch_cuda": torch.version.cuda,
                "max": max_version,
                "max_cli": max_cli_version,
                "mojo": mojo_version,
            },
            "source": _git_metadata(),
        },
        "workload": {
            "kind": "seeded-synthetic",
            "seed": args.seed,
            "width": 1024,
            "height": 768,
            "camera_count": 1,
            "dtype": "float32",
            "device": str(device),
            "camera": {
                "view_matrix": "identity world-to-camera",
                "fx": 600.0,
                "fy": 600.0,
                "cx": 512.0,
                "cy": 384.0,
                "near_plane": 0.2,
            },
            "distributions": {
                "means_xy": "uniform[-1, 1)",
                "means_z": "uniform[2, 6)",
                "quaternions_wxyz": "identity",
                "scales": "uniform[0.01, 0.05)",
                "opacities": "uniform[0.25, 0.75)",
                "colors_rgb": "uniform[0, 1)",
            },
        },
        "policy": {
            "cold": {
                "process": "fresh subprocess",
                "cache": "empty isolated MAX/Modular cache directories",
                "synchronization": "torch.cuda.synchronize before and after call",
            },
            "warmup_calls": args.warmups,
            "measured_calls": args.iterations,
            "warm_synchronization": (
                "torch.cuda.synchronize immediately before and after every call"
            ),
            "timer": "time.perf_counter_ns",
        },
        "counts": {
            "gaussians": n,
            "intersections": {
                "available": False,
                "count": None,
                "reason": (
                    "the finalized public API returns no tile-intersection "
                    "counter; the atomic custom op keeps it internal"
                ),
            },
        },
        "timing": {
            "cold_jit": {
                "measurement": "empty-cache first public API call",
                "total_first_call_ms": cold_ms,
                "estimated_jit_ms": max(0.0, cold_ms - warm_median),
                "estimate_method": "first-call time minus warm median",
                "includes_first_render": True,
                "cache_environment": {
                    "MODULAR_CACHE_DIR": os.environ.get("MODULAR_CACHE_DIR"),
                    "MODULAR_MAX_CACHE_DIR": os.environ.get("MODULAR_MAX_CACHE_DIR"),
                },
            },
            "warm_end_to_end": warm_summary,
            "per_stage": {
                "available": False,
                "stages": None,
                "reason": (
                    "the finalized public API dispatches projection, binning, "
                    "scan, sort, and rasterization as one atomic custom op"
                ),
            },
        },
        "memory": {
            "cold_device": cold_monitor.result(),
            "warm_device": warm_monitor.result(),
            "torch_allocator": {
                "scope": (
                    "PyTorch allocator only; MAX custom-op scratch may be "
                    "outside this allocator"
                ),
                "cold_start_allocated_bytes": cold_torch_start,
                "cold_peak_allocated_bytes": cold_torch_peak,
                "cold_peak_delta_bytes": cold_torch_peak - cold_torch_start,
                "warm_start_allocated_bytes": warm_torch_start,
                "warm_peak_allocated_bytes": warm_torch_peak,
                "warm_peak_delta_bytes": warm_torch_peak - warm_torch_start,
            },
        },
        "result": {
            "api_meta": meta,
            "render_colors_shape": list(render_colors.shape),
            "render_alphas_shape": list(render_alphas.shape),
            "color_sum_float64": color_sum,
            "alpha_sum_float64": alpha_sum,
            "max_alpha": max_alpha,
            "summary_digest_sha256": output_digest,
        },
        "limitations": [
            "This is a ray-renderer self-baseline, not an upstream comparison.",
            "Cold JIT is estimated because compilation and first dispatch are inseparable.",
            "Internal stage timings and intersection count are not public API outputs.",
            "nvidia-smi memory samples are system-wide, sampled, and MiB-granular.",
        ],
    }


def _run_parent(args: argparse.Namespace) -> dict[str, object]:
    supplied_cache = args.cold_cache_dir
    temporary_cache: tempfile.TemporaryDirectory[str] | None = None
    if supplied_cache is None:
        temporary_cache = tempfile.TemporaryDirectory(prefix="gsplat-mojo-cold-cache-")
        cache_root = pathlib.Path(temporary_cache.name)
    else:
        cache_root = supplied_cache.expanduser().resolve()
        if cache_root.exists() and any(cache_root.iterdir()):
            raise SystemExit("--cold-cache-dir must be absent or empty")
        cache_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="gsplat-mojo-result-") as result_dir:
        result_path = pathlib.Path(result_dir) / "worker.json"
        command = [
            sys.executable,
            str(pathlib.Path(__file__).resolve()),
            "--gaussians",
            str(args.gaussians),
            "--warmups",
            str(args.warmups),
            "--iterations",
            str(args.iterations),
            "--seed",
            str(args.seed),
            "--device",
            str(args.device),
            "--_worker-result",
            str(result_path),
        ]
        environment = os.environ.copy()
        environment["MODULAR_CACHE_DIR"] = str(cache_root / "modular")
        environment["MODULAR_MAX_CACHE_DIR"] = str(cache_root / "max")
        subprocess.run(command, check=True, env=environment)
        artifact = json.loads(result_path.read_text(encoding="utf-8"))

    if temporary_cache is not None:
        temporary_cache.cleanup()
    errors = _validate_artifact(artifact)
    if errors:
        raise RuntimeError("worker emitted an invalid artifact: " + "; ".join(errors))
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_cli(args)
    if args.print_schema:
        sys.stdout.write(SCHEMA_PATH.read_text(encoding="utf-8"))
        return 0
    if args.validate is not None:
        data = json.loads(args.validate.read_text(encoding="utf-8"))
        errors = _validate_artifact(data)
        if errors:
            for error in errors:
                print(f"invalid: {error}", file=sys.stderr)
            return 1
        print(f"valid: schema_version={SCHEMA_VERSION}")
        return 0
    if args._worker_result is not None:
        artifact = _run_worker(args)
        _write_json(artifact, args._worker_result)
        return 0
    artifact = _run_parent(args)
    _write_json(artifact, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
