from __future__ import annotations

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
HARNESS = ROOT / "benchmarks" / "phase0_ray.py"
SCHEMA = ROOT / "benchmarks" / "phase0-ray-baseline.schema.json"


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HARNESS), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def test_print_schema_is_exact_and_deterministic() -> None:
    first = _run("--print-schema")
    second = _run("--print-schema")

    assert first.returncode == 0, first.stderr
    assert first.stdout == second.stdout
    assert first.stdout == SCHEMA.read_text(encoding="utf-8")
    assert json.loads(first.stdout)["properties"]["schema_version"] == {
        "const": "1.0.0"
    }


def test_help_is_deterministic_and_names_self_baseline() -> None:
    first = _run("--help")
    second = _run("--help")
    normalized_help = " ".join(first.stdout.split())

    assert first.returncode == 0
    assert first.stdout == second.stdout
    assert "self" in normalized_help
    assert "upstream gsplat" in normalized_help


def test_cli_rejects_invalid_workload_before_cuda() -> None:
    completed = _run("--gaussians", "0")

    assert completed.returncode != 0
    assert "--gaussians must be in [1, 400000]" in completed.stderr


def test_validate_rejects_wrong_schema_version(tmp_path: pathlib.Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps({"schema_version": "future"}) + "\n", encoding="utf-8"
    )

    completed = _run("--validate", str(artifact))

    assert completed.returncode == 1
    assert "$.schema_version must equal '1.0.0'" in completed.stderr
