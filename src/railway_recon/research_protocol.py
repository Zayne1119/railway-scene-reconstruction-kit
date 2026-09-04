from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .benchmark import freeze_benchmark, initialize_benchmark, validate_benchmark_root
from .io import load_json, sha256_file, sha256_json, write_json

BASELINE_SUFFIXES = {
    ".csv",
    ".json",
    ".lock",
    ".md",
    ".ps1",
    ".py",
    ".toml",
    ".yaml",
    ".yml",
}
DEFAULT_BASELINE_INCLUDES = (
    "pyproject.toml",
    "uv.lock",
    "src/railway_recon",
    "configs/templates",
    "tests",
    "benchmarks/protocols",
    "docs/PAPER_BENCHMARK_CN.md",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _relative_to(path: Path, root: Path, label: str) -> Path:
    try:
        return path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must remain inside repository root {root}: {path}") from error


def _git_state(root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *arguments],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    entries = [] if not status else status.splitlines()
    return {
        "commit": commit,
        "working_tree_dirty": bool(entries),
        "status_entry_count": len(entries),
        "status_sha256": sha256_json(entries),
        "note": (
            "File-level hashes below are authoritative for this freeze; a dirty working tree "
            "is allowed but must be disclosed."
        ),
    }


def _iter_baseline_files(root: Path, includes: Iterable[str | Path]) -> list[Path]:
    files: set[Path] = set()
    for reference in includes:
        candidate = Path(reference)
        resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
        _relative_to(resolved, root, "Baseline include")
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        if resolved.is_file():
            if resolved.suffix.lower() in BASELINE_SUFFIXES:
                files.add(resolved)
            continue
        for path in resolved.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in BASELINE_SUFFIXES:
                continue
            relative_parts = set(_relative_to(path.resolve(), root, "Baseline file").parts)
            if relative_parts & {".git", ".pytest_cache", ".ruff_cache", "__pycache__"}:
                continue
            files.add(path.resolve())
    return sorted(files, key=lambda item: str(item).lower())


def freeze_research_baseline(
    repository_root: str | Path,
    output_path: str | Path,
    freeze_id: str = "baseline-v1",
    includes: Iterable[str | Path] | None = None,
    benchmark_root: str | Path | None = None,
) -> Path:
    """Freeze research source/config/protocol state without copying or modifying it."""

    root = Path(repository_root).resolve()
    if not (root / "pyproject.toml").is_file():
        raise ValueError(f"Not a railway reconstruction repository: {root}")
    output = Path(output_path).resolve()
    _relative_to(output, root, "Baseline output")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite research baseline: {output}")

    selected = tuple(includes or DEFAULT_BASELINE_INCLUDES)
    files = [path for path in _iter_baseline_files(root, selected) if path != output]
    if not files:
        raise ValueError("Research baseline contains no files")
    records = [
        {
            "path": _relative_to(path, root, "Baseline file").as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]

    benchmark_binding: dict[str, Any] | None = None
    if benchmark_root is not None:
        local_root = Path(benchmark_root).resolve()
        _relative_to(local_root, root, "Benchmark root")
        candidates = sorted(local_root.glob("runs/*-benchmark-freeze/manifest.json"))
        completed = [path for path in candidates if load_json(path).get("status") == "completed"]
        if not completed:
            raise FileNotFoundError(
                f"No completed benchmark freeze manifest exists under {local_root}"
            )
        latest = completed[-1]
        benchmark_binding = {
            "path": _relative_to(latest, root, "Benchmark freeze").as_posix(),
            "sha256": sha256_file(latest),
        }

    manifest: dict[str, Any] = {
        "schema_version": "railway.research.baseline-freeze.v1",
        "freeze_id": freeze_id,
        "created_at": _now(),
        "status": "completed",
        "purpose": "Freeze the development-scene research state before prospective data.",
        "scope": "source_config_protocol_tests_and_selected_reports",
        "non_mutating": True,
        "production_algorithms_modified_by_freeze": False,
        "repository": _git_state(root),
        "included_roots": [Path(item).as_posix() for item in selected],
        "file_count": len(records),
        "files": records,
        "benchmark_input_freeze": benchmark_binding,
        "limitations": [
            "This manifest binds files but does not copy them.",
            "A dirty working tree is valid only because every included file is content-hashed.",
            "The development scene is not an independent generalization test.",
        ],
    }
    manifest["payload_sha256"] = sha256_json(manifest)
    write_json(output, manifest)
    return output


def verify_research_baseline(
    repository_root: str | Path, manifest_path: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    source = Path(manifest_path).resolve()
    _relative_to(source, root, "Baseline manifest")
    manifest = load_json(source)
    issues: list[str] = []
    if manifest.get("schema_version") != "railway.research.baseline-freeze.v1":
        issues.append("unsupported schema_version")
    expected_payload_hash = manifest.get("payload_sha256")
    payload = dict(manifest)
    payload.pop("payload_sha256", None)
    if expected_payload_hash != sha256_json(payload):
        issues.append("manifest payload_sha256 mismatch")
    for record in manifest.get("files", []):
        relative = Path(str(record.get("path", "")))
        path = (root / relative).resolve()
        try:
            _relative_to(path, root, "Frozen file")
        except ValueError as error:
            issues.append(str(error))
            continue
        if not path.is_file():
            issues.append(f"missing file: {relative.as_posix()}")
            continue
        if path.stat().st_size != record.get("bytes"):
            issues.append(f"byte-size mismatch: {relative.as_posix()}")
        if sha256_file(path) != record.get("sha256"):
            issues.append(f"SHA-256 mismatch: {relative.as_posix()}")
    return {
        "schema_version": "railway.research.baseline-verification.v1",
        "manifest": str(source),
        "status": "pass" if not issues else "fail",
        "checked_files": len(manifest.get("files", [])),
        "issues": issues,
    }


def _make_test_blocks(scene_id: str, end_m: float, atomic_m: float = 25.0) -> list[dict[str, Any]]:
    if end_m <= 0 or abs((end_m / atomic_m) - round(end_m / atomic_m)) > 1e-9:
        raise ValueError(f"chainage_end_m must be a positive multiple of {atomic_m:g} m")
    blocks: list[dict[str, Any]] = []
    start = 0.0
    while start < end_m:
        end = start + atomic_m
        group_start = int(start // 50.0) * 50
        group_end = min(group_start + 50, int(end_m))
        blocks.append(
            {
                "block_id": f"{scene_id}-{int(start):04d}-{int(end):04d}",
                "evaluation_group_id": f"{scene_id}-eval-{group_start:04d}-{group_end:04d}",
                "scene_id": scene_id,
                "role": "test",
                "core_start_m": start,
                "core_end_m": end,
                "context_before_m": 0.0 if start == 0.0 else 10.0,
                "context_after_m": 0.0 if end == end_m else 10.0,
            }
        )
        start = end
    return blocks


def initialize_prospective_benchmark(
    target: str | Path,
    dataset_id: str,
    scene_id: str,
    chainage_end_m: float = 200.0,
    baseline_manifest: str | Path | None = None,
) -> Path:
    """Create an isolated Site-B workspace without changing production defaults."""

    test_blocks = _make_test_blocks(scene_id, float(chainage_end_m))
    root = initialize_benchmark(target, dataset_id, scene_id)
    dataset_path = root / "manifests" / "dataset-manifest.json"
    split_path = root / "manifests" / "split-manifest.json"
    truth_path = root / "annotations" / "ground-truth.json"
    experiment_path = root / "experiments" / "full-auto.json"

    dataset = load_json(dataset_path)
    dataset["role"] = "prospective_blind_test"
    scene = dataset["scenes"][0]
    scene["split_role"] = "test"
    scene["chainage_range_m"] = [0.0, float(chainage_end_m)]
    scene["limitations"] = [
        "Prospective B0 scene; do not tune reconstruction parameters from test output.",
        "Absolute accuracy requires independent check points not used for registration.",
    ]

    split = load_json(split_path)
    split["split_id"] = f"{dataset_id}-prospective-b0-v1"
    split["blocks"] = test_blocks

    truth = load_json(truth_path)
    truth["ground_truth_id"] = f"{dataset_id}-prospective-gt-draft"
    truth["split_id"] = split["split_id"]
    truth["truth_level"] = "GT-L1S_observed_single_reviewer"
    truth["annotation"]["double_annotation_fraction"] = 0.0
    truth["annotation"]["adjudication_required"] = False
    scene["limitations"].extend(
        [
            "Single-owner review is allowed operationally and must be disclosed in the paper.",
            "Ground truth remains draft until sampled raw-evidence review is completed.",
        ]
    )

    experiment = load_json(experiment_path)
    experiment["experiment_id"] = f"{dataset_id}-b0-frozen-generalization-v1"
    experiment["split_id"] = split["split_id"]
    experiment["ground_truth_id"] = truth["ground_truth_id"]
    experiment["variant_id"] = "b0-frozen-generalization"
    experiment["parameters"] = {
        "adaptation_policy": "format_units_crs_sensor_calibration_only",
        "result_policy": "preserve_first_run_before_any_site_specific_tuning",
    }

    evaluation_path = root / "experiments" / "evaluation-input.json"
    evaluation = load_json(evaluation_path)
    evaluation["evaluation_id"] = f"{dataset_id}-b0-evaluation-draft"
    evaluation["experiment_id"] = experiment["experiment_id"]

    failure_path = root / "failures" / "failure-case.example.json"
    failure = load_json(failure_path)
    failure["block_id"] = split["blocks"][0]["block_id"]

    baseline_binding: dict[str, Any] | None = None
    if baseline_manifest is not None:
        binding = Path(baseline_manifest).resolve()
        if not binding.is_file():
            raise FileNotFoundError(binding)
        value = load_json(binding)
        if value.get("schema_version") != "railway.research.baseline-freeze.v1":
            raise ValueError("baseline_manifest is not a research baseline freeze")
        baseline_binding = {
            "path": str(binding),
            "sha256": sha256_file(binding),
            "freeze_id": value.get("freeze_id"),
        }

    input_contract = {
        "schema_version": "railway.research.input-contract.v1",
        "dataset_id": dataset_id,
        "scene_id": scene_id,
        "status": "awaiting_inputs",
        "required_inputs": ["point_cloud"],
        "optional_inputs": ["camera_poses", "panoramas", "survey", "asset_ledger"],
        "required_declarations": [
            "coordinate_mode",
            "units",
            "axis",
            "epsg_or_explicit_local_frame",
            "vertical_datum_or_unknown",
        ],
        "accepted_point_cloud_suffixes": [".las", ".laz", ".e57", ".pcd", ".ply"],
        "fixed_units": "metre",
        "fixed_axis": "Z-up",
        "hash_algorithm": "SHA-256",
        "notes": [
            "Do not reuse Site-A coordinate offsets or detector thresholds as sensor metadata.",
            "Register inputs with this module; do not hand-edit hashes.",
        ],
    }
    state = {
        "schema_version": "railway.research.blind-run-state.v1",
        "dataset_id": dataset_id,
        "scene_id": scene_id,
        "status": "awaiting_inputs",
        "created_at": _now(),
        "baseline_binding": baseline_binding,
        "production_defaults_modified": False,
        "allowed_before_b0": ["format", "units", "crs", "sensor_calibration"],
        "forbidden_before_b0": [
            "site_specific_threshold_tuning",
            "topology_rule_changes_from_test_output",
            "manual_mesh_repair_before_first_metrics",
            "discarding_failed_first_run",
        ],
        "output_policy": {
            "b0": "runs/b0_frozen_generalization",
            "b1": "runs/b1_calibration_assisted",
            "never_overwrite_b0": True,
        },
    }

    write_json(dataset_path, dataset)
    write_json(split_path, split)
    write_json(truth_path, truth)
    write_json(experiment_path, experiment)
    write_json(evaluation_path, evaluation)
    write_json(failure_path, failure)
    write_json(root / "manifests" / "input-contract.json", input_contract)
    write_json(root / "blind-run-state.json", state)

    _, errors = validate_benchmark_root(root)
    if errors:
        raise ValueError("Generated prospective benchmark is invalid:\n- " + "\n- ".join(errors))
    return root


def _input_record(identifier: str, kind: str, path: Path, required: bool) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() and not resolved.is_dir():
        raise FileNotFoundError(resolved)
    if resolved.is_dir():
        files = sorted(item for item in resolved.rglob("*") if item.is_file())
        if not files:
            raise ValueError(f"Input directory is empty: {resolved}")
        digest_payload = [
            {
                "path": item.relative_to(resolved).as_posix(),
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
            for item in files
        ]
        return {
            "id": identifier,
            "kind": kind,
            "path": str(resolved),
            "required": required,
            "bytes": sum(item["bytes"] for item in digest_payload),
            "sha256": sha256_json(digest_payload),
            "metadata": {"directory_file_count": len(digest_payload)},
        }
    return {
        "id": identifier,
        "kind": kind,
        "path": str(resolved),
        "required": required,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def register_prospective_inputs(
    benchmark_root: str | Path,
    point_cloud: str | Path,
    camera_poses: str | Path | None = None,
    panoramas: str | Path | None = None,
    coordinate_mode: str = "local",
    epsg: int | None = None,
    vertical_datum: str | None = None,
) -> Path:
    root = Path(benchmark_root).resolve()
    state_path = root / "blind-run-state.json"
    state = load_json(state_path)
    if state.get("status") != "awaiting_inputs":
        raise ValueError(f"Inputs cannot be registered from state {state.get('status')!r}")
    if coordinate_mode not in {"local", "epsg"}:
        raise ValueError("coordinate_mode must be 'local' or 'epsg'")
    if coordinate_mode == "epsg" and epsg is None:
        raise ValueError("epsg is required when coordinate_mode='epsg'")
    if coordinate_mode == "local" and epsg is not None:
        raise ValueError("Do not provide EPSG for an explicitly local coordinate frame")

    dataset_path = root / "manifests" / "dataset-manifest.json"
    dataset = load_json(dataset_path)
    if dataset.get("role") != "prospective_blind_test":
        raise ValueError("Input registration is limited to prospective_blind_test datasets")
    scene = dataset["scenes"][0]
    if scene.get("inputs"):
        raise FileExistsError("Inputs are already registered; create a new benchmark to replace them")

    records = [_input_record("point-cloud", "point_cloud", Path(point_cloud), True)]
    if camera_poses is not None:
        records.append(_input_record("camera-poses", "camera_poses", Path(camera_poses), False))
    if panoramas is not None:
        panorama_path = Path(panoramas)
        panorama_kind = "panorama_directory" if panorama_path.is_dir() else "panorama_archive"
        records.append(_input_record("panoramas", panorama_kind, panorama_path, False))
    scene["inputs"] = records
    scene["coordinate_reference"]["epsg"] = epsg
    scene["coordinate_reference"]["vertical_datum"] = vertical_datum
    scene["coordinate_reference"]["coordinate_mode"] = coordinate_mode
    if coordinate_mode == "local":
        scene["limitations"].append(
            "Coordinates use an explicit local frame; no absolute world-accuracy claim is allowed."
        )
    if vertical_datum is None:
        scene["limitations"].append(
            "Vertical datum is unknown; report only internal relative-height accuracy."
        )

    contract_path = root / "manifests" / "input-contract.json"
    contract = load_json(contract_path)
    contract["status"] = "registered_unfrozen"
    contract["registered_at"] = _now()
    contract["coordinate_mode"] = coordinate_mode
    contract["epsg"] = epsg
    contract["vertical_datum"] = vertical_datum
    contract["input_ids"] = [record["id"] for record in records]

    state["status"] = "inputs_registered_unfrozen"
    state["inputs_registered_at"] = _now()
    state["input_manifest_sha256"] = sha256_json(dataset)

    write_json(dataset_path, dataset)
    write_json(contract_path, contract)
    write_json(state_path, state)
    return dataset_path


def seal_prospective_intake(benchmark_root: str | Path) -> Path:
    root = Path(benchmark_root).resolve()
    state_path = root / "blind-run-state.json"
    state = load_json(state_path)
    if state.get("status") != "inputs_registered_unfrozen":
        raise ValueError(f"Intake cannot be sealed from state {state.get('status')!r}")
    dataset = load_json(root / "manifests" / "dataset-manifest.json")
    point_clouds = [
        item
        for scene in dataset.get("scenes", [])
        for item in scene.get("inputs", [])
        if item.get("kind") == "point_cloud" and item.get("required", True)
    ]
    if not point_clouds:
        raise ValueError("A required point_cloud input must be registered before sealing")

    freeze_path = freeze_benchmark(root, full_hash=True)
    freeze = load_json(freeze_path)
    if freeze.get("status") != "completed":
        raise ValueError("Full-hash intake freeze failed; inspect its warnings")
    state["status"] = "intake_frozen_ready_for_experiment_lock"
    state["sealed_at"] = _now()
    state["intake_freeze_manifest"] = str(freeze_path)
    state["intake_freeze_sha256"] = sha256_file(freeze_path)
    write_json(state_path, state)
    return freeze_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Isolated research baseline and prospective Site-B preparation"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze-baseline")
    freeze.add_argument("--repo-root", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--freeze-id", default="baseline-v1")
    freeze.add_argument("--benchmark-root")
    freeze.add_argument("--include", action="append", default=[])

    verify = commands.add_parser("verify-baseline")
    verify.add_argument("--repo-root", required=True)
    verify.add_argument("--manifest", required=True)

    init = commands.add_parser("init-blind-site")
    init.add_argument("--target", required=True)
    init.add_argument("--dataset-id", required=True)
    init.add_argument("--scene-id", required=True)
    init.add_argument("--chainage-end-m", type=float, default=200.0)
    init.add_argument("--baseline-manifest")

    register = commands.add_parser("register-inputs")
    register.add_argument("--root", required=True)
    register.add_argument("--point-cloud", required=True)
    register.add_argument("--camera-poses")
    register.add_argument("--panoramas")
    register.add_argument("--coordinate-mode", choices=["local", "epsg"], required=True)
    register.add_argument("--epsg", type=int)
    register.add_argument("--vertical-datum")

    seal = commands.add_parser("seal-intake")
    seal.add_argument("--root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze-baseline":
        includes = args.include or None
        path = freeze_research_baseline(
            args.repo_root,
            args.output,
            freeze_id=args.freeze_id,
            includes=includes,
            benchmark_root=args.benchmark_root,
        )
    elif args.command == "verify-baseline":
        report = verify_research_baseline(args.repo_root, args.manifest)
        print(json.dumps(report, ensure_ascii=False))
        return 0 if report["status"] == "pass" else 1
    elif args.command == "init-blind-site":
        path = initialize_prospective_benchmark(
            args.target,
            args.dataset_id,
            args.scene_id,
            args.chainage_end_m,
            args.baseline_manifest,
        )
    elif args.command == "register-inputs":
        path = register_prospective_inputs(
            args.root,
            args.point_cloud,
            args.camera_poses,
            args.panoramas,
            args.coordinate_mode,
            args.epsg,
            args.vertical_datum,
        )
    else:
        path = seal_prospective_intake(args.root)
    print(json.dumps({"status": "ok", "path": str(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
