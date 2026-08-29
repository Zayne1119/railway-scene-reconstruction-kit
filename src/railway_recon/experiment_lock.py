from __future__ import annotations

import copy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .benchmark import validate_benchmark_file, validate_benchmark_root, validate_benchmark_value
from .io import load_json, sha256_file, write_json


def _latest_completed_freeze(root: Path) -> Path:
    candidates = sorted(root.glob("runs/*-benchmark-freeze/manifest.json"))
    for path in reversed(candidates):
        value = load_json(path)
        if value.get("schema_version") == "railway.run-manifest.v2" and value.get(
            "status"
        ) == "completed":
            return path
    raise FileNotFoundError(
        "No completed benchmark freeze was found; run benchmark-freeze first"
    )


def lock_benchmark_experiment(
    benchmark_root: str | Path,
    experiment_path: str | Path,
    bindings: list[tuple[str, str | Path]],
    output_path: str | Path | None = None,
    freeze_manifest_path: str | Path | None = None,
) -> Path:
    """Freeze one executable experiment before ground-truth inspection.

    Bindings are method/config/checkpoint artifacts whose content hashes define the
    implementation being evaluated. Dataset inputs remain bound through a completed
    benchmark-freeze run manifest.
    """

    root = Path(benchmark_root).resolve()
    values, root_errors = validate_benchmark_root(root)
    if root_errors:
        raise ValueError("Invalid benchmark root:\n- " + "\n- ".join(root_errors))

    source = Path(experiment_path).resolve()
    experiment, errors = validate_benchmark_file(source)
    if errors:
        raise ValueError("Invalid experiment:\n- " + "\n- ".join(errors))
    if experiment.get("schema_version") != "railway.benchmark.experiment.v1":
        raise ValueError("Experiment lock accepts railway.benchmark.experiment.v1 only")
    if experiment.get("status") != "planned":
        raise ValueError("Only a planned experiment can be locked")
    if not experiment.get("command"):
        raise ValueError("Experiment command is empty; an executable command must be frozen")
    if experiment.get("locked_before_test"):
        raise ValueError("Experiment already declares locked_before_test=true")

    expected = {
        "protocol_id": values["protocol"]["protocol_id"],
        "dataset_id": values["dataset"]["dataset_id"],
        "split_id": values["split"]["split_id"],
    }
    for key, expected_value in expected.items():
        if experiment.get(key) != expected_value:
            raise ValueError(
                f"Experiment {key}={experiment.get(key)!r} does not match "
                f"benchmark {expected_value!r}"
            )

    freeze_path = (
        Path(freeze_manifest_path).resolve()
        if freeze_manifest_path is not None
        else _latest_completed_freeze(root)
    )
    freeze = load_json(freeze_path)
    if freeze.get("schema_version") != "railway.run-manifest.v2":
        raise ValueError("Unsupported benchmark freeze manifest")
    if freeze.get("status") != "completed":
        raise ValueError("Benchmark freeze manifest did not complete")
    for key, expected_value in expected.items():
        if freeze.get(key) != expected_value:
            raise ValueError(f"Freeze manifest {key} does not match the experiment")

    binding_records: list[dict[str, Any]] = []
    seen_roles: set[str] = set()
    for role, reference in bindings:
        normalized_role = role.strip()
        if not normalized_role:
            raise ValueError("Binding role cannot be empty")
        if normalized_role in seen_roles:
            raise ValueError(f"Duplicate binding role: {normalized_role}")
        seen_roles.add(normalized_role)
        path = Path(reference).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        binding_records.append(
            {
                "role": normalized_role,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not binding_records:
        raise ValueError("At least one method/config binding is required")

    locked = copy.deepcopy(experiment)
    locked["status"] = "locked"
    locked["locked_before_test"] = True
    locked["lock"] = {
        "locked_at": datetime.now(UTC).isoformat(),
        "source_experiment_path": str(source),
        "source_experiment_sha256": sha256_file(source),
        "benchmark_freeze_manifest": str(freeze_path),
        "benchmark_freeze_sha256": sha256_file(freeze_path),
        "bindings": binding_records,
        "ground_truth_not_used_by_lock": True,
    }
    locked_errors = validate_benchmark_value(locked)
    if locked_errors:
        raise ValueError("Generated locked experiment is invalid:\n- " + "\n- ".join(locked_errors))

    output = (
        Path(output_path).resolve()
        if output_path is not None
        else root / "experiments" / "locked" / f"{experiment['experiment_id']}.locked.json"
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite locked experiment: {output}")
    write_json(output, locked)
    return output
