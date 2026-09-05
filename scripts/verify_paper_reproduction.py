"""Verify immutable research artifacts against a source-snapshot replay.

This script executes no experiment, generates no layout, and edits no inputs.
Run it with the reproduction environment's Python to record its package versions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import numpy as np

SYNTHETIC_SCHEMA = "railway.synthetic-track-guarded-run-manifest.v1"
PUBLIC_SCHEMA = "railway.public-rail-spacing-manifest.v1"
SNAPSHOT_SCHEMA = "railway.paper-source-snapshot.v1"
PUBLIC_NONALGORITHM_SOURCE = "src/railway_recon/synthetic_track_formal_analysis.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checked_path(root: Path, relative: str) -> Path:
    if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
            or PurePosixPath(relative).is_absolute()
            or any(part in {"", ".", ".."} for part in relative.split("/"))):
        raise ValueError("Manifest paths must be canonical relative POSIX paths")
    path = root.joinpath(*PurePosixPath(relative).parts)
    if path.resolve() != root.resolve().joinpath(*PurePosixPath(relative).parts):
        raise ValueError("Manifest path is redirected or escapes its root")
    return path


def _json(path: Path) -> object:
    return json.loads(path.read_bytes())


def verify_inventory(root: Path, filename: str, schema: str, *, exact: bool = True) -> dict:
    """Check every listed file and, for runs, require the exact full inventory."""
    root = root.resolve()
    path = root / filename
    before = sha256_file(path)
    manifest = _json(path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != schema:
        raise ValueError("Unexpected manifest schema")
    if schema != SYNTHETIC_SCHEMA and manifest.get("status") != "completed":
        raise ValueError("Manifest is not completed")
    if schema == SYNTHETIC_SCHEMA and manifest.get("source_unchanged_during_run") is not True:
        raise ValueError("Synthetic run lacks source stability verification")
    if (root / "run_failure.json").exists():
        raise ValueError("Failed runs are ineligible for reproduction verification")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest requires a nonempty artifact list")
    if schema == SNAPSHOT_SCHEMA and manifest.get("file_count") != len(records):
        raise ValueError("Snapshot file_count disagrees with bindings")
    bindings = {}
    for record in records:
        relative = record["path"]
        candidate = checked_path(root, relative)
        if relative in bindings or relative == filename:
            raise ValueError("Duplicate/self-referential artifact binding")
        if type(record.get("bytes")) is not int or record["bytes"] < 0:
            raise ValueError("Artifact byte count is invalid")
        if not isinstance(record.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
            raise ValueError("Artifact SHA-256 is invalid")
        if candidate.stat().st_size != record["bytes"] or sha256_file(candidate) != record["sha256"]:
            raise ValueError(f"Artifact integrity mismatch: {relative}")
        bindings[relative] = record["sha256"]
    if exact:
        actual = {item.relative_to(root).as_posix() for item in root.rglob("*") if item.is_file()}
        if actual != set(bindings) | {filename}:
            raise ValueError("Run manifest must bind every file, without extra or missing artifacts")
    if sha256_file(path) != before:
        raise ValueError("Manifest changed during verification")
    return {"manifest": manifest, "manifest_sha256": before, "bindings": bindings,
            "files_verified": len(bindings), "inventory_exact": exact}


def verify_snapshot(root: Path) -> dict:
    """Verify copied source bytes; later environments/runs are not snapshot files."""
    verified = verify_inventory(root, "snapshot_manifest.json", SNAPSHOT_SCHEMA, exact=False)
    return {"status": "pass", "manifest_sha256": verified["manifest_sha256"],
            "files_verified": verified["files_verified"], "every_bound_file_verified": True,
            "additional_environment_and_run_files_attested_by_snapshot": False,
            "boundary": "Source snapshot bytes only; no publication or license clearance implied"}


def compare_sources(original: dict, replay: dict, kind: str) -> dict:
    left, right = original.get("source_sha256"), replay.get("source_sha256")
    if not isinstance(left, dict) or not left or not isinstance(right, dict) or not right:
        raise ValueError("Both runs require nonempty source hash inventories")
    differences = []
    for relative in sorted(set(left) | set(right)):
        checked_path(Path.cwd(), relative)  # Validate text; never emit absolute paths.
        for digest in (left.get(relative), right.get(relative)):
            if digest is not None and (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
                raise ValueError("Source hash inventory contains an invalid SHA-256")
        if left.get(relative) != right.get(relative):
            known = kind == "public" and relative == PUBLIC_NONALGORITHM_SOURCE
            differences.append({"path": relative, "original_sha256": left.get(relative),
                                "replay_sha256": right.get(relative),
                                "classification": ("synthetic_analysis_not_public_extraction_or_scoring"
                                                   if known else "unapproved_source_difference"),
                                "excluded_from_algorithm_identity_check": known})
    return {"all_recorded_sources_identical": not differences, "differences": differences,
            "algorithm_source_check_passed": all(item["excluded_from_algorithm_identity_check"]
                                                 for item in differences),
            "scope": "Recorded source inventories; not a claim of identical machine state"}


def _same_npz(first: Path, second: Path) -> bool:
    with np.load(first, allow_pickle=False) as left, np.load(second, allow_pickle=False) as right:
        if set(left.files) != set(right.files) or len(left.files) != len(set(left.files)):
            return False
        for name in left.files:
            a, b = left[name], right[name]
            if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
                return False
    return True


def _same_artifact(first: Path, second: Path) -> bool:
    if first.suffix == ".npz":
        return _same_npz(first, second)
    if first.suffix == ".json":
        return _json(first) == _json(second)
    return sha256_file(first) == sha256_file(second)


def _verify_public_lock(root: Path, verified: dict) -> None:
    lock = _json(root / "prediction_lock.json")
    predictions = {name: digest for name, digest in verified["bindings"].items()
                   if "predictions" in PurePosixPath(name).parts}
    if not predictions or lock.get("files") != predictions:
        raise ValueError("Public prediction lock must bind every prediction artifact exactly")
    if lock.get("status") != "all_samples_all_modes_saved_before_any_reference_evaluation":
        raise ValueError("Public pre-evaluation prediction lock status is invalid")
    for key in ("protocol_sha256", "input_sha256"):
        if lock.get(key) != verified["manifest"].get(key):
            raise ValueError("Public prediction lock disagrees with input/protocol bindings")


def compare_runs(original: Path, replay: Path, kind: str) -> dict:
    """Compare all result artifacts; disclose timing/source differences explicitly."""
    if kind not in {"synthetic", "public"}:
        raise ValueError("Run kind must be synthetic or public")
    if original.resolve() == replay.resolve():
        raise ValueError("Replay must be a distinct run, not the original directory itself")
    schema = SYNTHETIC_SCHEMA if kind == "synthetic" else PUBLIC_SCHEMA
    left = verify_inventory(original, "manifest.json", schema)
    right = verify_inventory(replay, "manifest.json", schema)
    source_comparison = compare_sources(left["manifest"], right["manifest"], kind)
    mismatches = []
    if not source_comparison["algorithm_source_check_passed"]:
        mismatches.append("algorithm_source_inventory")
    bindings = (left["bindings"], right["bindings"])
    if set(bindings[0]) != set(bindings[1]):
        mismatches.append("artifact_inventory")
    manifest_keys = (("data_fingerprint_sha256", "prediction_fingerprint_sha256",
                      "protocol_sha256", "policy_sha256", "split") if kind == "synthetic"
                     else ("protocol_sha256", "input_sha256", "receipt_sha256"))
    for key in manifest_keys:
        if key not in left["manifest"] or left["manifest"][key] != right["manifest"].get(key):
            mismatches.append(f"manifest.{key}")
    if kind == "public":
        _verify_public_lock(original, left)
        _verify_public_lock(replay, right)
    excluded = {"timings.json"}
    if kind == "public":
        # Lock bytes are each checked independently. NPZ container compression
        # may differ while typed array contents remain exactly equal.
        excluded.add("prediction_lock.json")
    compared, container_differences = [], []
    for relative in sorted((set(bindings[0]) & set(bindings[1])) - excluded):
        if not _same_artifact(original / relative, replay / relative):
            mismatches.append(relative)
        elif bindings[0][relative] != bindings[1][relative]:
            container_differences.append(relative)
        compared.append(relative)
    report_left, report_right = _json(original / "report.json"), _json(replay / "report.json")
    required = (("data_fingerprint_sha256", "prediction_fingerprint_sha256", "summary_by_mode",
                 "by_group", "by_condition", "case_count", "base_layout_count") if kind == "synthetic"
                else ("samples", "by_parent_cloud", "pooled_descriptive_evaluation", "sample_count",
                      "parent_cloud_count", "total_point_count"))
    for key in required:
        if key not in report_left or key not in report_right or report_left[key] != report_right[key]:
            mismatches.append(f"report.{key}")
    if kind == "synthetic":
        for key in ("data_fingerprint_sha256", "prediction_fingerprint_sha256"):
            if report_left[key] != left["manifest"][key] or report_right[key] != right["manifest"][key]:
                mismatches.append(f"report_manifest_consistency.{key}")
    return {
        "kind": kind, "status": "pass" if not mismatches else "fail",
        "original_manifest_sha256": left["manifest_sha256"],
        "replay_manifest_sha256": right["manifest_sha256"],
        "original_files_verified": left["files_verified"], "replay_files_verified": right["files_verified"],
        "all_artifacts_hash_verified_on_both_sides": True,
        "artifact_comparison_count": len(compared), "compared_artifacts": compared,
        "mismatches": sorted(set(mismatches)), "source_comparison": source_comparison,
        "equal_content_different_serialized_bytes": container_differences,
        "excluded_from_between_run_result_equality": sorted(excluded),
        "excluded_files_still_integrity_verified": True,
        "source_and_dependency_manifests_not_required_byte_identical": True,
        "manifest_metadata_not_compared_for_result_equality": [
            "environment", "freeze_sha256", "freeze_verification",
        ],
        "freeze_metadata_boundary": "Replay uses its own local freeze; paths are not scientific results",
        "original_run_environment": left["manifest"].get("environment"),
        "replay_run_environment": right["manifest"].get("environment"),
        "replay_is_additional_held_out_test": False,
        "boundary": "Re-execution consistency, not new scientific evidence or independent validation",
    }


def recorded_environment() -> dict:
    versions = {}
    for name in ("numpy", "scipy", "Pillow", "laspy", "lazrs", "jsonschema", "pytest", "ruff"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "system": platform.system(),
            "packages": versions, "isolated_interpreter": sys.prefix != sys.base_prefix,
            "interpreter_binary_sha256": sha256_file(Path(sys.executable)),
            "fresh_install_history_verified_by_this_script": False,
            "boundary": "Actual verifier interpreter; creation/install history needs separate receipts"}


def verify_reproduction(*, synthetic_original: Path, synthetic_replay: Path,
                        public_original: Path, public_replay: Path, snapshot: Path,
                        output: Path) -> dict:
    inputs = [path.resolve() for path in (synthetic_original, synthetic_replay,
                                          public_original, public_replay)]
    destination = output.resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Verification output already exists")
    if any(destination.is_relative_to(path) for path in inputs):
        raise ValueError("Verification output must be outside immutable run inputs")
    if destination.is_relative_to(snapshot.resolve()):
        raise ValueError("Verification output must be outside the source snapshot")
    snapshot_result = verify_snapshot(snapshot)
    comparisons = [compare_runs(synthetic_original, synthetic_replay, "synthetic"),
                   compare_runs(public_original, public_replay, "public")]
    report = {"schema_version": "railway.paper-reproduction-verification.v1",
              "status": "pass" if all(item["status"] == "pass" for item in comparisons) else "fail",
              "created_at_utc": datetime.now(UTC).isoformat(), "snapshot": snapshot_result,
              "comparisons": comparisons, "verifier_environment": recorded_environment(),
              "verifier_source_sha256": sha256_file(Path(__file__)),
              "input_locations_disclosed": False, "customer_data_loaded": False,
              "published": False, "replay_is_additional_held_out_test": False,
              "scope": "Only explicitly supplied synthetic/public experiment artifacts and source snapshot"}
    # Complete receipts must not combine changing inputs or a changing snapshot.
    if verify_snapshot(snapshot) != snapshot_result:
        raise ValueError("Snapshot changed during verification")
    for first, second, result in ((synthetic_original, synthetic_replay, comparisons[0]),
                                  (public_original, public_replay, comparisons[1])):
        schema = SYNTHETIC_SCHEMA if result["kind"] == "synthetic" else PUBLIC_SCHEMA
        for path, key in ((first, "original_manifest_sha256"), (second, "replay_manifest_sha256")):
            if verify_inventory(path, "manifest.json", schema)["manifest_sha256"] != result[key]:
                raise ValueError("Input run changed during verification")
    destination.mkdir(parents=True, exist_ok=False)
    report_path = destination / "report.json"
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    receipt = {"schema_version": "railway.paper-reproduction-verification-manifest.v1",
               "status": report["status"], "files": [{"path": "report.json",
                   "bytes": report_path.stat().st_size, "sha256": sha256_file(report_path)}],
               "input_content_addresses": {
                   "source_snapshot": "sha256:" + snapshot_result["manifest_sha256"],
                   **{f"{result['kind']}_{role}": "sha256:" + result[f"{role}_manifest_sha256"]
                      for result in comparisons for role in ("original", "replay")}},
               "manifest_self_hash_excluded": True}
    with (destination / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("synthetic-original", "synthetic-replay", "public-original", "public-replay",
                 "snapshot", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        report = verify_reproduction(**vars(args))
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps({"status": report["status"], "comparisons": [
        {"kind": item["kind"], "status": item["status"], "mismatches": item["mismatches"],
         "source_differences": item["source_comparison"]["differences"]}
        for item in report["comparisons"]]}, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
