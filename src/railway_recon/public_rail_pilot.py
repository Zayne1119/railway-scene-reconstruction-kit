"""Hash-pinned, label-blind extraction pilot on one public training sample.

This is a bounded offline development experiment, not a general dataset loader,
held-out benchmark, route-identity evaluator or production model promotion.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import time
from pathlib import Path

import laspy
import numpy as np

from .public_las_intake import _sha256, inspect_public_las
from .synthetic_track_pilot import _canonical_bytes, _write_json

PUBLIC_SAMPLE = {
    "filename": "cloud180_Seg1.laz",
    "sha256": "b81701df0f2f07dff14fe956f4b507c9ea719f416d99550da8e550b41ef36367",
    "point_count": 907816,
    "parent_cloud": "cloud180",
    "segment": 1,
    "split": "train",
    "selection": "first_complete_member_of_bounded_train_archive_prefix_not_random_sampling",
}
PUBLIC_METADATA_SHA256 = {
    "zenodo_15641832.json": "b321fd4d035a8819ceba95a1bca88e1bbdd75a654a5e4cd111d7150e0e08dd59",
    "upstream_README.md": "a5dd3032269fc15ea46dae46e8f0f746aeec48780be9d44e1404e21c8408804d",
    "train_clouds.txt": "b43bf0ba8826364de60ac3658338b6be94e5242c9c6c52de9230b12d4ae7826c",
    "val_clouds.txt": "c9e532f9836629a857836eb881c609d92f0f7d1ca75e379e3d0879ae04c1e1e6",
    "test_clouds.txt": "0e9910e18e649b13fbb13e4dfded4dec9ec163158fbdf07e58a9863677dd84b6",
    "removed_clouds.txt": "7aa4cc14fa941ff964112db4c8176331aac9aed0033c76a0d5fdf3e38fbd7975",
}
DATASET = {
    "name": "SemanticRail3D-V2",
    "doi": "10.5281/zenodo.15641832",
    "source_url": "https://zenodo.org/records/15641832",
    "documentation_commit": "ff7d48c7518646f5093b1de57ce89f8969f9d25b",
    "license_id": "cc-by-4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "attribution": "Ghasemlou, Soilan, Martinez-Sanchez, Arias, Lorenzo and Riveiro (2025)",
    "processing": "XYZ-only geometric candidate extraction; numeric semantic reference evaluation",
}


def _digest(value: dict) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def create_public_rail_protocol() -> dict:
    from .public_rail_candidates import DEFAULT_PUBLIC_RAIL_POLICY, PUBLIC_RAIL_CANDIDATE_MODES
    from .public_rail_reference import PUBLIC_RAIL_REFERENCE_CONTRACT

    value = {
        "schema_version": "railway.public-rail-pilot-protocol.v1",
        "dataset": copy.deepcopy(DATASET), "sample": copy.deepcopy(PUBLIC_SAMPLE),
        "metadata_sha256": copy.deepcopy(PUBLIC_METADATA_SHA256),
        "reference_contract": copy.deepcopy(PUBLIC_RAIL_REFERENCE_CONTRACT),
        "policy": copy.deepcopy(DEFAULT_PUBLIC_RAIL_POLICY),
        "modes": list(PUBLIC_RAIL_CANDIDATE_MODES),
        "role": "single_public_training_sample_exploratory_not_held_out_test",
        "candidate_input": ["xyz"],
        "reference_use": "evaluation_only_after_both_predictions_are_saved_and_locked",
        "comparison_unit": "one_parent_cloud_not_independent_points",
        "connection_truth_available": False,
        "absolute_accuracy_available": False,
        "maximum_points": 2000000,
        "maximum_display_points": 100000,
    }
    value["protocol_id"] = "public-rail-pilot-" + _digest(value)[:24]
    return value


def validate_public_rail_protocol(value: dict) -> None:
    if not isinstance(value, dict) or _digest(value) != _digest(create_public_rail_protocol()):
        raise ValueError("Public pilot protocol must match the declared sample, reference and fixed policy")


def _parse_split(text: str) -> set[tuple[str, int]]:
    entries = set()
    for line in text.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"(cloud\d+)_Seg\s+([1-5](?:\s*,\s*[1-5])*)", line.strip())
        if match is None:
            raise ValueError("Malformed public split declaration")
        for segment in match[2].split(","):
            key = (match[1], int(segment))
            if key in entries:
                raise ValueError("Duplicate public split entry")
            entries.add(key)
    return entries


def verify_public_metadata(directory: str | Path, protocol: dict) -> dict:
    """Check exact previously acquired public metadata, not private directories."""
    root = Path(directory).resolve()
    contents = {}
    for name, expected in protocol["metadata_sha256"].items():
        path = root / name
        if path.resolve().parent != root:
            raise ValueError("Metadata path escapes the explicit directory")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Public metadata hash mismatch: {name}")
        contents[name] = data.decode("utf-8-sig")
    record = json.loads(contents["zenodo_15641832.json"])
    if (record["id"] != 15641832 or record["doi"] != DATASET["doi"]
            or record["metadata"]["license"]["id"] != "cc-by-4.0"
            or record["metadata"]["access_right"] != "open"):
        raise ValueError("Pinned record does not match the declared open dataset")
    sample = protocol["sample"]
    key = (sample["parent_cloud"], sample["segment"])
    splits = {name: _parse_split(contents[name + "_clouds.txt"])
              for name in ("train", "val", "test", "removed")}
    if key not in splits["train"] or any(key in splits[name] for name in ("val", "test", "removed")):
        raise ValueError("Selected sample is not an unambiguous retained training member")
    if any(parent == key[0] for name in ("val", "test") for parent, _ in splits[name]):
        raise ValueError("Selected parent cloud overlaps a validation/test declaration")
    return {"status": "verified", "metadata_sha256": protocol["metadata_sha256"],
            "retained_train_member": True, "parent_cloud": key[0],
            "validation_or_test_points_loaded": False,
            "boundary": "Pinned public metadata and member hash; full archive hash was not verified"}


def load_pinned_public_arrays(source: Path, expected_sha256: str, maximum_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode XYZ and separate raw labels; the extractor never receives labels."""
    if type(maximum_points) is not int or maximum_points < 1:
        raise ValueError("maximum_points must be a positive integer")
    with source.open("rb") as stream:
        if _sha256(stream) != expected_sha256:
            raise ValueError("Public input hash mismatch before array decode")
        stream.seek(0)
        with laspy.open(stream, closefd=False, read_evlrs=False) as reader:
            count = int(reader.header.point_count)
            if not 0 < count <= maximum_points:
                raise ValueError("Public point count is empty or exceeds the declared bound")
            if "classification" not in set(reader.header.point_format.dimension_names):
                raise ValueError("Public reference classification is absent")
            xyz = np.empty((count, 3), dtype=np.float64)
            labels = np.empty(count, dtype=np.uint8)
            cursor = 0
            for chunk in reader.chunk_iterator(100000):
                stop = cursor + len(chunk)
                if stop > count:
                    raise ValueError("Array decode exceeds declared point count")
                xyz[cursor:stop] = np.column_stack((chunk.x, chunk.y, chunk.z))
                labels[cursor:stop] = np.asarray(chunk.classification)
                cursor = stop
            if cursor != count or not np.isfinite(xyz).all():
                raise ValueError("Public array decode incomplete or nonfinite")
        if _sha256(stream) != expected_sha256:
            raise ValueError("Public input changed during array decode")
    return xyz, labels


def _source_bindings() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = [*sorted((root / "src" / "railway_recon").rglob("*.py")),
             *sorted((root / "src" / "railway_recon" / "resources").glob("*.json")),
             root / "scripts" / "run_public_rail_pilot.py", root / "pyproject.toml", root / "uv.lock"]
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def _save_array(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _report_text(report: dict) -> str:
    lines = ["# Public rail candidate pilot", "",
             "One SemanticRail3D-V2 TRAIN sample; exploratory transfer of geometric heuristics, not held-out performance.", "",
             f"Input: {report['source']['filename']}; {report['point_count']} decoded points. Labels are used only for evaluation, after both XYZ-only predictions are saved and locked.", "",
             f"Reference target: **{report['reference_contract']['target_name']}**, numeric IDs {report['reference_contract']['target_class_ids']}. Its semantic mapping status is recorded in `reference_contract`; unverified numeric-class agreement must not be called rail segmentation accuracy.", "",
             "| Mode | View | TP | FP | FN | TN | Precision | Recall | F1 | IoU |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for mode, result in report["evaluation_by_mode"].items():
        for view in ("all_points", "annotated_only"):
            values = result[view]
            formatted = [str(values[key]) if values[key] is not None else "NA"
                         for key in ("tp", "fp", "fn", "tn", "precision", "recall", "f1", "iou")]
            lines.append(f"| {mode} | {view} | " + " | ".join(formatted) + " |")
    lines.extend(["", "The all-points view treats unclassified codes as non-target for numeric comparison only, not confirmed negatives. Annotated-only excludes the declared unclassified code and can be optimistic; both views and per-class selected counts are retained.", "",
                  "Candidate polylines and masks are geometric hypotheses, not corrected rail models or relationship truth. Internal fit residuals, coordinate storage steps and semantic-reference scores are not absolute measurement accuracy. One parent cloud provides no cross-site confidence interval or evidence of saved human review time.", "",
                  "All points are evaluated. The figure uses a deterministic display subset with full-cloud bounds; unclassified points stay visually distinct. No outcome-based threshold adjustment, crop selection or failed-candidate deletion is performed.", "",
                  "See candidate reports for empty/degenerate outcomes, both predictions for all selected points, and `error_indices.npz` for all TP/FP/FN/ignored-selected point indices in each scoring view. Public raw data is not reclassified or overwritten. No customer inputs, test generation, network upload or production gate change occurs.", "",
                  "Source: [SemanticRail3D-V2](https://zenodo.org/records/15641832), Ghasemlou et al. (2025), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: geometric extraction and evaluation. Original authors do not endorse this pilot.", ""])
    return "\n".join(lines)


def run_public_rail_pilot(input_path: str | Path, metadata_directory: str | Path,
                          protocol_path: str | Path, output: str | Path) -> dict:
    from .public_rail_candidates import extract_public_rail_candidates
    from .public_rail_figure import render_public_rail_comparison
    from .public_rail_reference import evaluate_public_rail_mask

    destination, source = Path(output).resolve(), Path(input_path).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Public pilot output already exists: {destination}")
    protocol_bytes = Path(protocol_path).read_bytes()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    validate_public_rail_protocol(protocol)
    provenance = verify_public_metadata(metadata_directory, protocol)
    sample = protocol["sample"]
    if source.name != sample["filename"]:
        raise ValueError("Public input filename differs from the pinned sample")
    sources = _source_bindings()
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        _write_json(destination / "protocol_snapshot.json", protocol)
        _write_json(destination / "provenance.json", provenance)
        intake = inspect_public_las(source, sample["sha256"], destination / "intake.json",
                                    max_points=protocol["maximum_points"])
        if intake["decoded"]["point_count"] != sample["point_count"]:
            raise ValueError("Decoded point count differs from pinned public sample")
        xyz, labels = load_pinned_public_arrays(source, sample["sha256"], protocol["maximum_points"])
        _save_array(destination / "inputs" / "xyz.npz", xyz=xyz)
        _save_array(destination / "ground_truth" / "classification.npz", classification=labels)
        predictions, candidates, timing = {}, {}, {}
        immutable_xyz = xyz.copy()
        immutable_xyz.flags.writeable = False
        for mode in protocol["modes"]:
            before = time.perf_counter()
            result = extract_public_rail_candidates(immutable_xyz, copy.deepcopy(protocol["policy"]), mode)
            timing[mode] = time.perf_counter() - before
            mask = result["point_mask"]
            if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != (len(xyz),):
                raise ValueError("Extractor must return one explicit boolean per input point")
            predictions[mode] = mask.copy()
            candidates[mode] = result["report"]
            _save_array(destination / "predictions" / mode / "mask.npz", point_mask=mask)
            _write_json(destination / "predictions" / mode / "candidates.json", result["report"])
        if not np.array_equal(xyz, immutable_xyz):
            raise ValueError("Extractor mutated its XYZ-only input")
        if (candidates["height_prominence"]["candidate_pool"] != candidates["paired_geometry"]["candidate_pool"]
                or np.any(predictions["paired_geometry"] & ~predictions["height_prominence"])):
            raise ValueError("Paired mode must only select from the unchanged local candidate pool")
        _write_json(destination / "prediction_lock.json", {
            "status": "saved_before_semantic_evaluation", "source_sha256": sample["sha256"],
            "candidate_input_fields": ["xyz"], "policy_sha256": _digest(protocol["policy"]),
            "files": {path.relative_to(destination).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in sorted((destination / "predictions").rglob("*")) if path.is_file()},
            "boundary": "Internal ordering record, not external blind preregistration",
        })
        reference = protocol["reference_contract"]
        target = tuple(reference["target_class_ids"])
        ignored = tuple(reference["ignored_class_ids"])
        evaluations = {}
        rows = []
        for mode, mask in predictions.items():
            evaluation = evaluate_public_rail_mask(labels, mask, target, ignored)
            evaluations[mode] = evaluation
            _write_json(destination / "evaluation" / f"{mode}.json", evaluation)
            positive, excluded = np.isin(labels, target), np.isin(labels, ignored)
            indices = {}
            for view, eligible in (("all_points", np.ones(len(labels), dtype=bool)),
                                   ("annotated_only", ~excluded)):
                indices.update({f"{view}_tp": np.flatnonzero(eligible & positive & mask),
                                f"{view}_fp": np.flatnonzero(eligible & ~positive & mask),
                                f"{view}_fn": np.flatnonzero(eligible & positive & ~mask)})
            indices["ignored_selected"] = np.flatnonzero(excluded & mask)
            _save_array(destination / "evaluation" / mode / "error_indices.npz", **indices)
            for view in ("all_points", "annotated_only"):
                rows.append({"mode": mode, "view": view, **{key: evaluation[view][key]
                             for key in ("tp", "fp", "fn", "tn", "precision", "recall", "f1", "iou")}})
        figure = render_public_rail_comparison(xyz, labels, predictions, evaluations,
                    destination / "comparison.png", target_class_ids=target, ignored_class_ids=ignored,
                    source_label=sample["filename"], max_display_points=protocol["maximum_display_points"],
                    target_label=reference["target_name"])
        report = {
            "schema_version": "railway.public-rail-pilot-report.v1", "status": "completed",
            "protocol_id": protocol["protocol_id"], "role": protocol["role"],
            "source": sample, "dataset": protocol["dataset"], "point_count": len(xyz),
            "parent_cloud_count": 1, "candidate_input_fields": ["xyz"],
            "reference_contract": reference, "evaluation_by_mode": evaluations,
            "candidate_summary": {mode: {"selected_point_count": int(mask.sum()),
                                         "status": candidates[mode]["status"]}
                                  for mode, mask in predictions.items()},
            "figure": figure, "customer_data_used": False, "held_out_test": False,
            "connection_accuracy": {"available": False, "reason": "no_independent_relation_truth"},
            "absolute_accuracy": {"available": False, "reason": "no_independent_survey_reference"},
        }
        _write_json(destination / "timings.json", {"detector_seconds": timing,
                    "elapsed_seconds": time.perf_counter() - started,
                    "scope": "single_run_wall_clock_not_statistical_speed_or_human_time_comparison"})
        with (destination / "metrics.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if sources != _source_bindings() or Path(protocol_path).read_bytes() != protocol_bytes:
            raise ValueError("Source code or protocol changed during the public pilot")
        with source.open("rb") as stream:
            if _sha256(stream) != sample["sha256"]:
                raise ValueError("Public input changed during extraction/evaluation")
        verify_public_metadata(metadata_directory, protocol)
        _write_json(destination / "report.json", report)
        with (destination / "report.md").open("x", encoding="utf-8") as stream:
            stream.write(_report_text(report))
        _write_json(destination / "manifest.json", {
            "schema_version": "railway.public-rail-pilot-manifest.v1", "status": "completed",
            "source_sha256": sources, "input_sha256": sample["sha256"],
            "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "environment": {"python": platform.python_version(), "system": platform.system(),
                "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "Pillow", "laspy")}},
            "files": [{"path": path.relative_to(destination).as_posix(), "bytes": path.stat().st_size,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                      for path in sorted(destination.rglob("*")) if path.is_file()],
            "manifest_self_hash_excluded": True,
        })
        return report
    except Exception as error:
        _write_json(destination / "run_failure.json", {"status": "failed", "exception_type": type(error).__name__})
        raise
