from __future__ import annotations

import csv
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import resources
from itertools import pairwise
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover - minimal offline installations
    Draft202012Validator = None  # type: ignore[assignment]

from .io import load_json, sha256_file, sha256_json, write_json

SCHEMA_RESOURCES = {
    "railway.benchmark.protocol.v1": "benchmark-protocol.schema.json",
    "railway.benchmark.dataset-manifest.v1": "benchmark-dataset-manifest.schema.json",
    "railway.benchmark.split-manifest.v1": "benchmark-split-manifest.schema.json",
    "railway.benchmark.ground-truth.v1": "benchmark-ground-truth.schema.json",
    "railway.benchmark.experiment.v1": "benchmark-experiment.schema.json",
    "railway.benchmark.metric-report.v1": "benchmark-metric-report.schema.json",
    "railway.benchmark.metric-report.v2": "benchmark-metric-report-v2.schema.json",
    "railway.benchmark.evaluation-input.v1": "benchmark-evaluation-input.schema.json",
    "railway.benchmark.evaluation-input.v2": "benchmark-evaluation-input-v2.schema.json",
    "railway.benchmark.failure-case.v1": "benchmark-failure-case.schema.json",
    "railway.run-manifest.v2": "run-manifest-v2.schema.json",
}


def _resource_json(name: str) -> dict[str, Any]:
    resource = resources.files("railway_recon.resources").joinpath(name)
    with resource.open("r", encoding="utf-8") as stream:
        import json

        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected an object in resource {name}")
    return value


def _now() -> str:
    return datetime.now(UTC).isoformat()


def validate_benchmark_value(value: dict[str, Any]) -> list[str]:
    version = value.get("schema_version")
    resource_name = SCHEMA_RESOURCES.get(str(version))
    if resource_name is None:
        return [
            "<root>: unsupported schema_version; expected one of "
            + ", ".join(sorted(SCHEMA_RESOURCES))
        ]

    errors: list[str] = []
    if Draft202012Validator is not None:
        validator = Draft202012Validator(_resource_json(resource_name))
        errors.extend(
            f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: "
            f"{error.message}"
            for error in sorted(validator.iter_errors(value), key=lambda item: list(item.path))
        )

    if version == "railway.benchmark.dataset-manifest.v1":
        scene_ids = [str(scene.get("scene_id")) for scene in value.get("scenes", [])]
        if len(scene_ids) != len(set(scene_ids)):
            errors.append("scenes: duplicate scene_id values")
        for scene in value.get("scenes", []):
            input_ids = [str(item.get("id")) for item in scene.get("inputs", [])]
            if len(input_ids) != len(set(input_ids)):
                errors.append(f"scenes.{scene.get('scene_id')}.inputs: duplicate input ids")

    if version == "railway.benchmark.split-manifest.v1":
        block_ids = [str(block.get("block_id")) for block in value.get("blocks", [])]
        if len(block_ids) != len(set(block_ids)):
            errors.append("blocks: duplicate block_id values")
        for block in value.get("blocks", []):
            start = block.get("core_start_m")
            end = block.get("core_end_m")
            if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end <= start:
                errors.append(f"blocks.{block.get('block_id')}: core_end_m must exceed core_start_m")

    if version == "railway.benchmark.ground-truth.v1":
        instance_ids = [str(item.get("id")) for item in value.get("instances", [])]
        if len(instance_ids) != len(set(instance_ids)):
            errors.append("instances: duplicate instance ids")
        known = set(instance_ids)
        for relation in value.get("relations", []):
            missing = [key for key in ("from", "to") if relation.get(key) not in known]
            if missing:
                errors.append(
                    f"relations.{relation.get('id')}: references unknown instance(s)"
                )

    if version == "railway.benchmark.evaluation-input.v2":
        ground_truth_ids = [str(item.get("id")) for item in value.get("ground_truth_assets", [])]
        prediction_ids = [str(item.get("id")) for item in value.get("predicted_assets", [])]
        if len(ground_truth_ids) != len(set(ground_truth_ids)):
            errors.append("ground_truth_assets: duplicate ground-truth asset ids")
        if len(prediction_ids) != len(set(prediction_ids)):
            errors.append("predicted_assets: duplicate predicted asset ids")
        known_ground_truth = set(ground_truth_ids)
        known_predictions = set(prediction_ids)
        for index, relation in enumerate(value.get("ground_truth_relations", [])):
            if relation.get("from") not in known_ground_truth or relation.get("to") not in known_ground_truth:
                errors.append(
                    f"ground_truth_relations.{index}: references unknown ground-truth asset"
                )
        for index, relation in enumerate(value.get("predicted_relations", [])):
            if relation.get("from") not in known_predictions or relation.get("to") not in known_predictions:
                errors.append(
                    f"predicted_relations.{index}: references unknown predicted asset"
                )
        unknown_ignored = set(value.get("ignored_prediction_ids", [])) - known_predictions
        if unknown_ignored:
            errors.append(
                "ignored_prediction_ids: unknown predictions " + ", ".join(sorted(unknown_ignored))
            )

    return errors


def validate_benchmark_file(path: str | Path) -> tuple[dict[str, Any], list[str]]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    value = load_json(source)
    return value, validate_benchmark_value(value)


def validate_benchmark_root(root: str | Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    benchmark_root = Path(root).resolve()
    paths = {
        "protocol": benchmark_root / "protocols" / "paper_v1.json",
        "dataset": benchmark_root / "manifests" / "dataset-manifest.json",
        "split": benchmark_root / "manifests" / "split-manifest.json",
    }
    values: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for name, path in paths.items():
        try:
            value, document_errors = validate_benchmark_file(path)
        except FileNotFoundError:
            errors.append(f"{name}: missing file {path}")
            continue
        values[name] = value
        errors.extend(f"{name}: {error}" for error in document_errors)
    if errors or len(values) != len(paths):
        return values, errors

    protocol = values["protocol"]
    dataset = values["dataset"]
    split = values["split"]
    if split["dataset_id"] != dataset["dataset_id"]:
        errors.append("split.dataset_id does not match dataset.dataset_id")
    if split["protocol_id"] != protocol["protocol_id"]:
        errors.append("split.protocol_id does not match protocol.protocol_id")

    scene_by_id = {scene["scene_id"]: scene for scene in dataset["scenes"]}
    blocks_by_scene: dict[str, list[dict[str, Any]]] = {}
    for block in split["blocks"]:
        scene_id = block["scene_id"]
        scene = scene_by_id.get(scene_id)
        if scene is None:
            errors.append(f"block {block['block_id']} references unknown scene {scene_id}")
            continue
        scene_start, scene_end = scene["chainage_range_m"]
        if block["core_start_m"] < scene_start or block["core_end_m"] > scene_end:
            errors.append(f"block {block['block_id']} core lies outside its scene range")
        block_length = block["core_end_m"] - block["core_start_m"]
        atomic_length = protocol["spatial_split"]["atomic_block_m"]
        if abs(block_length - atomic_length) > 1e-6:
            errors.append(
                f"block {block['block_id']} length {block_length} m does not match "
                f"atomic_block_m {atomic_length}"
            )
        blocks_by_scene.setdefault(scene_id, []).append(block)

    for scene_id, blocks in blocks_by_scene.items():
        ordered = sorted(blocks, key=lambda item: (item["core_start_m"], item["core_end_m"]))
        for previous, current in pairwise(ordered):
            if current["core_start_m"] < previous["core_end_m"]:
                errors.append(
                    f"scene {scene_id} has overlapping score cores: "
                    f"{previous['block_id']} and {current['block_id']}"
                )

    for scene in dataset["scenes"]:
        coordinate_reference = scene["coordinate_reference"]
        control = coordinate_reference.get("control_points")
        check = coordinate_reference.get("independent_check_points")
        if control and check and control == check:
            errors.append(
                f"scene {scene['scene_id']} reuses control points as independent check points"
            )

    if dataset["role"] == "prospective_blind_test" and not any(
        block["role"] == "test" for block in split["blocks"]
    ):
        errors.append("prospective_blind_test must contain at least one test block")
    return values, errors


def _write_review_templates(root: Path) -> None:
    with (root / "annotations" / "human_review_log.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "review_id",
                "reviewer_id",
                "scene_id",
                "block_id",
                "asset_or_candidate_id",
                "started_at",
                "finished_at",
                "duration_seconds",
                "decision",
                "evidence_used",
                "edit_action",
                "note",
            ]
        )
    with (root / "annotations" / "automation_boundary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "stage_id",
                "stage_name",
                "automation_level",
                "human_input",
                "machine_output",
                "acceptance_gate",
            ]
        )


def initialize_benchmark(target: str | Path, dataset_id: str, scene_id: str) -> Path:
    root = Path(target).resolve()
    if root.exists() and not root.is_dir():
        raise FileExistsError(f"Target exists and is not a directory: {root}")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Target directory is not empty: {root}")

    for relative in (
        "protocols",
        "manifests",
        "annotations",
        "experiments",
        "runs",
        "failures",
        "reports",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)

    protocol = {
        "schema_version": "railway.benchmark.protocol.v1",
        "protocol_id": "paper-v1",
        "title": "Evidence-aware railway reconstruction benchmark",
        "created_at": _now(),
        "spatial_split": {
            "assignment": "unique_chainage",
            "atomic_block_m": 25.0,
            "evaluation_core_m": 50.0,
            "context_m": 10.0,
            "score_context": False,
        },
        "evaluation_modes": {
            "frozen_generalization": {
                "allowed_adaptations": ["format", "units", "crs", "sensor_calibration"]
            },
            "calibration_assisted": {"pilot_length_m": [20.0, 50.0]},
        },
        "evidence_levels": [
            "observed",
            "photo_interpreted",
            "rule_inferred",
            "unsupported",
        ],
        "primary_metrics": [
            "geometry.p90_m",
            "instances.macro_f1",
            "topology.relation_f1",
            "evidence.unsupported_hallucination_rate",
            "confidence.ece",
            "human.minutes_per_100m",
        ],
        "statistics": {
            "primary_unit": "scene",
            "secondary_unit": "spatial_block",
            "confidence_interval": "cluster_bootstrap_95",
        },
    }
    dataset = {
        "schema_version": "railway.benchmark.dataset-manifest.v1",
        "dataset_id": dataset_id,
        "role": "development_case_study",
        "created_at": _now(),
        "scenes": [
            {
                "scene_id": scene_id,
                "site_id": scene_id,
                "split_role": "development",
                "chainage_range_m": [0.0, 200.0],
                "coordinate_reference": {
                    "units": "metre",
                    "axis": "Z-up",
                    "epsg": None,
                    "vertical_datum": None,
                },
                "inputs": [],
                "limitations": [
                    "Development scene; repeatedly inspected and not an independent blind test."
                ],
            }
        ],
    }
    split = {
        "schema_version": "railway.benchmark.split-manifest.v1",
        "split_id": f"{dataset_id}-development-v1",
        "dataset_id": dataset_id,
        "protocol_id": "paper-v1",
        "created_at": _now(),
        "blocks": [
            {
                "block_id": f"{scene_id}-000-025",
                "evaluation_group_id": f"{scene_id}-eval-000-050",
                "scene_id": scene_id,
                "role": "development",
                "core_start_m": 0.0,
                "core_end_m": 25.0,
                "context_before_m": 0.0,
                "context_after_m": 10.0,
            }
        ],
    }
    ground_truth = {
        "schema_version": "railway.benchmark.ground-truth.v1",
        "ground_truth_id": f"{dataset_id}-gt-draft",
        "dataset_id": dataset_id,
        "split_id": split["split_id"],
        "truth_level": "GT-L1_observed_consensus",
        "status": "draft",
        "instances": [],
        "relations": [],
        "exclusion_regions": [],
        "annotation": {
            "blind_to_final_model": True,
            "double_annotation_fraction": 0.2,
            "adjudication_required": True,
        },
    }
    experiment = {
        "schema_version": "railway.benchmark.experiment.v1",
        "experiment_id": f"{dataset_id}-full-auto-v1",
        "protocol_id": "paper-v1",
        "dataset_id": dataset_id,
        "split_id": split["split_id"],
        "ground_truth_id": ground_truth["ground_truth_id"],
        "method_id": "railway-recon",
        "variant_id": "full",
        "execution_mode": "auto",
        "status": "planned",
        "command": [],
        "parameters": {},
        "random_seeds": [20260826],
        "locked_before_test": False,
    }
    evaluation_input = {
        "schema_version": "railway.benchmark.evaluation-input.v1",
        "evaluation_id": f"{dataset_id}-evaluation-draft",
        "experiment_id": experiment["experiment_id"],
        "assets": [],
        "ground_truth_relations": [],
        "predicted_relations": [],
        "distance_sets": [],
        "human_review": {"corridor_length_m": 0.0, "events": []},
        "limitations": ["Draft input; populate from locked predictions and ground truth."],
    }
    failure = {
        "schema_version": "railway.benchmark.failure-case.v1",
        "failure_id": "FAILURE-EXAMPLE",
        "scene_id": scene_id,
        "block_id": f"{scene_id}-000-025",
        "category": "geometry",
        "severity": "major",
        "stage": "reconstruction",
        "status": "open",
        "description": "Replace this example with a real, pre-defined failure record.",
        "detected_by_automatic_qa": False,
        "references": [],
    }

    write_json(root / "protocols" / "paper_v1.json", protocol)
    write_json(root / "manifests" / "dataset-manifest.json", dataset)
    write_json(root / "manifests" / "split-manifest.json", split)
    write_json(root / "annotations" / "ground-truth.json", ground_truth)
    write_json(root / "experiments" / "full-auto.json", experiment)
    write_json(root / "experiments" / "evaluation-input.json", evaluation_input)
    write_json(root / "failures" / "failure-case.example.json", failure)
    _write_review_templates(root)
    return root


def _resolve_reference(root: Path, reference: str) -> Path:
    path = Path(reference)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _file_record(root: Path, item: dict[str, Any], full_hash: bool) -> dict[str, Any]:
    reference = str(item["path"])
    path = _resolve_reference(root, reference)
    record: dict[str, Any] = {
        "id": item["id"],
        "kind": item["kind"],
        "reference": reference,
        "required": item.get("required", True),
        "exists": path.is_file(),
    }
    if not path.is_file():
        return record
    stat = path.stat()
    record["bytes"] = stat.st_size
    declared = item.get("sha256")
    if full_hash or not declared:
        record["sha256"] = sha256_file(path)
        record["hash_source"] = "computed"
    else:
        record["sha256"] = declared
        record["hash_source"] = "declared"
    expected_bytes = item.get("bytes")
    if expected_bytes is not None:
        record["size_matches_declared"] = int(expected_bytes) == stat.st_size
    return record


def _git_commit(start: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def freeze_benchmark(root: str | Path, full_hash: bool = False) -> Path:
    benchmark_root = Path(root).resolve()
    values, errors = validate_benchmark_root(benchmark_root)
    if errors:
        raise ValueError("Invalid benchmark root:\n- " + "\n- ".join(errors))
    protocol = values["protocol"]
    dataset = values["dataset"]
    split = values["split"]
    records = [
        _file_record(benchmark_root, item, full_hash)
        for scene in dataset["scenes"]
        for item in scene["inputs"]
    ]
    missing_required = [
        record["id"] for record in records if record["required"] and not record["exists"]
    ]
    size_mismatches = [
        record["id"] for record in records if record.get("size_matches_declared") is False
    ]
    timestamp = datetime.now(UTC)
    run_id = timestamp.strftime("%Y%m%dT%H%M%S%fZ") + "-benchmark-freeze"
    result = {
        "schema_version": "railway.run-manifest.v2",
        "run_id": run_id,
        "protocol_id": protocol["protocol_id"],
        "dataset_id": dataset["dataset_id"],
        "split_id": split["split_id"],
        "method_id": "benchmark-freeze",
        "variant_id": "site-freeze",
        "execution_mode": "auto",
        "status": "completed" if not missing_required and not size_mismatches else "failed",
        "started_at": timestamp.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "command": ["benchmark-freeze", "--root", str(benchmark_root)],
        "config_hashes": {
            "protocols/paper_v1.json": sha256_json(protocol),
            "manifests/dataset-manifest.json": sha256_json(dataset),
            "manifests/split-manifest.json": sha256_json(split),
        },
        "inputs": records,
        "ground_truth": [],
        "outputs": [],
        "random_seeds": [],
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git_commit": _git_commit(benchmark_root),
        },
        "human_review": {"minutes": 0.0, "operations": 0},
        "warnings": [
            *(f"Missing required input: {item}" for item in missing_required),
            *(f"Declared byte size does not match: {item}" for item in size_mismatches),
        ],
        "limitations": [
            "A declared hash is trusted unless --full-hash is requested.",
            "This manifest freezes evidence; it does not establish world-coordinate accuracy.",
        ],
    }
    errors = validate_benchmark_value(result)
    if errors:
        raise ValueError("Generated run manifest is invalid:\n- " + "\n- ".join(errors))
    output = benchmark_root / "runs" / run_id / "manifest.json"
    write_json(output, result)
    return output
