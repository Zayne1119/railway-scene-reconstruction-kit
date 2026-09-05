"""Additive, explicitly post-pilot TRAIN spacing-sensitivity experiment.

The broad interval is a fitted-line center-spacing hypothesis, not a physical
gauge measurement or an inferred railway standard. Numeric class 1 remains an
unverified semantic probe. All sample/mode predictions are locked before any
reference agreement is evaluated. No reserved synthetic layout is accessed.
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

import numpy as np

from .public_las_intake import _sha256, inspect_public_las
from .public_rail_candidates import DEFAULT_PUBLIC_RAIL_POLICY
from .public_rail_pilot import (
    DATASET,
    PUBLIC_METADATA_SHA256,
    PUBLIC_SAMPLE,
    _digest,
    _save_array,
    load_pinned_public_arrays,
    verify_public_metadata,
)
from .public_rail_pilot import (
    _source_bindings as _pilot_source_bindings,
)
from .public_rail_reference import PUBLIC_RAIL_REFERENCE_CONTRACT
from .synthetic_track_pilot import _write_json

SPACING_MODES = ("height_prominence", "paired_default", "paired_broad_spacing")
DEFAULT_RECEIPT = {
    "filename": "range_sample_receipt_v4.json",
    "sha256": "5ebf0484563a9051e2072bacd9bdbcc1d25082c631308da60c5d35e1de9e2349",
}
_METRICS = ("tp", "fp", "fn", "tn", "precision", "recall", "f1", "iou")
_VIEWS = ("all_points", "annotated_only")


def _sample_validation(samples: list[dict]) -> None:
    if not isinstance(samples, list) or not 1 <= len(samples) <= 8:
        raise ValueError("Supply one to eight explicitly pinned public TRAIN samples")
    if not isinstance(samples[0], dict):
        raise TypeError("Each sample must be a dictionary")
    base = {key: samples[0].get(key) for key in PUBLIC_SAMPLE}
    if base != PUBLIC_SAMPLE or samples[0].get("acquisition_receipt") != DEFAULT_RECEIPT:
        raise ValueError("The unchanged original public pilot sample must remain first")
    filenames, digests = set(), set()
    required = set(PUBLIC_SAMPLE) | {"acquisition_receipt"}
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != required:
            raise ValueError("Sample fields must match the declared pinned sample contract")
        name = sample["filename"]
        match = re.fullmatch(r"(cloud\d+)_Seg([1-5])\.laz", name) if isinstance(name, str) else None
        if match is None or sample["parent_cloud"] != match[1] or sample["segment"] != int(match[2]):
            raise ValueError("Sample filename, parent and segment must agree")
        if type(sample["segment"]) is not int or sample["split"] != "train":
            raise ValueError("Only retained TRAIN members are allowed")
        if type(sample["point_count"]) is not int or not 0 < sample["point_count"] <= 2_000_000:
            raise ValueError("Sample point count must be within the fixed resource bound")
        if not isinstance(sample["selection"], str) or not re.fullmatch(r"[a-z0-9_]{1,180}", sample["selection"]):
            raise ValueError("Selection must be an explicit safe descriptive identifier")
        digest = sample["sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Sample SHA256 must be 64 lowercase hexadecimal characters")
        if name in filenames or digest in digests:
            raise ValueError("Duplicate sample filenames or bytes are not independent samples")
        filenames.add(name)
        digests.add(digest)
        receipt = sample["acquisition_receipt"]
        if not isinstance(receipt, dict) or set(receipt) != {"filename", "sha256"}:
            raise ValueError("Each sample requires a hash-pinned acquisition receipt")
        if not isinstance(receipt["filename"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+\.json", receipt["filename"]):
            raise ValueError("Receipt filename must be a single safe JSON basename")
        if not isinstance(receipt["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", receipt["sha256"]):
            raise ValueError("Acquisition receipt SHA256 is invalid")


def create_public_spacing_protocol(samples: list[dict] | None = None) -> dict:
    selected = copy.deepcopy(samples if samples is not None else [
        {**PUBLIC_SAMPLE, "acquisition_receipt": DEFAULT_RECEIPT}
    ])
    _sample_validation(selected)
    policies = {mode: copy.deepcopy(DEFAULT_PUBLIC_RAIL_POLICY) for mode in SPACING_MODES}
    policies["paired_broad_spacing"]["rail_pair_maximum_m"] = 1.90
    value = {
        "schema_version": "railway.public-rail-spacing-protocol.v1",
        "dataset": copy.deepcopy(DATASET),
        "samples": selected,
        "metadata_sha256": copy.deepcopy(PUBLIC_METADATA_SHA256),
        "reference_contract": copy.deepcopy(PUBLIC_RAIL_REFERENCE_CONTRACT),
        "modes": list(SPACING_MODES),
        "policies": policies,
        "extractor_modes": {"height_prominence": "height_prominence",
                            "paired_default": "paired_geometry",
                            "paired_broad_spacing": "paired_geometry"},
        "role": "post_pilot_public_training_development_spacing_sensitivity_not_held_out",
        "development_exposure": {
            "original_pilot_already_viewed": True,
            "motivation": "original_candidate_intervals_about_1_75_and_1_78_m_exceeded_1_65_m",
            "broad_hypothesis": "fixed_fitted_center_spacing_1_35_to_1_90_m_not_physical_gauge",
            "new_semantic_mapping_inferred": False,
            "outcome_based_sample_or_mode_exclusion_allowed": False,
            "threshold_tuning_within_run_allowed": False,
        },
        "candidate_input": ["xyz"],
        "reference_use": "evaluation_only_after_all_samples_all_modes_saved_and_locked",
        "comparison_unit": "parent_cloud_descriptive_not_independent_points",
        "connection_truth_available": False,
        "absolute_accuracy_available": False,
        "maximum_points": 2_000_000,
        "maximum_samples": 8,
    }
    value["protocol_id"] = "public-rail-spacing-" + _digest(value)[:24]
    return value


def validate_public_spacing_protocol(value: dict) -> None:
    if not isinstance(value, dict):
        raise TypeError("Protocol must be a dictionary")
    _sample_validation(value.get("samples"))
    if _digest(value) != _digest(create_public_spacing_protocol(value["samples"])):
        raise ValueError("Spacing protocol differs from the declared fixed comparison contract")


def _file_in(directory: str | Path, name: str) -> Path:
    root = Path(directory).resolve()
    path = root / name
    if path.resolve().parent != root:
        raise ValueError("Explicit input/receipt path escapes its directory")
    return path


def _receipt_binds(value: object, filename: str, digest: str) -> bool:
    if isinstance(value, dict):
        if ((value.get("sample_file") == filename and value.get("sample_sha256") == digest)
                or (value.get("filename") == filename and value.get("sha256") == digest)):
            return True
        return any(_receipt_binds(item, filename, digest) for item in value.values())
    if isinstance(value, list):
        return any(_receipt_binds(item, filename, digest) for item in value)
    return False


def verify_spacing_receipts(directory: str | Path, samples: list[dict]) -> dict:
    bindings = {}
    for sample in samples:
        receipt = sample["acquisition_receipt"]
        data = _file_in(directory, receipt["filename"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != receipt["sha256"]:
            raise ValueError("Acquisition receipt hash mismatch")
        if not _receipt_binds(json.loads(data), sample["filename"], sample["sha256"]):
            raise ValueError("Acquisition receipt does not bind the declared sample filename and bytes")
        bindings[receipt["filename"]] = receipt["sha256"]
    return {"status": "hash_and_sample_binding_verified", "receipts_sha256": bindings,
            "boundary": "Local acquisition provenance, not independent full-archive verification"}


def _source_bindings() -> dict:
    sources = _pilot_source_bindings()
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "run_public_rail_spacing_study.py"
    sources[script.relative_to(root).as_posix()] = hashlib.sha256(script.read_bytes()).hexdigest()
    return sources


def _pooled_counts(evaluations: list[dict]) -> dict:
    value = {key: sum(item[key] for item in evaluations) for key in ("tp", "fp", "fn", "tn")}
    tp, fp, fn, tn = (value[key] for key in ("tp", "fp", "fn", "tn"))
    for name, numerator, denominator in (("precision", tp, tp + fp), ("recall", tp, tp + fn),
                                         ("f1", 2 * tp, 2 * tp + fp + fn), ("iou", tp, tp + fp + fn)):
        value[name] = numerator / denominator if denominator else None
    value["evaluated_points"] = tp + fp + fn + tn
    return value


def _evaluate_saved_sample(destination: Path, sample: dict, protocol: dict) -> tuple[dict, list[dict]]:
    from .public_rail_reference import evaluate_public_rail_mask

    sample_root = destination / "samples" / Path(sample["filename"]).stem
    with np.load(sample_root / "ground_truth" / "classification.npz", allow_pickle=False) as stored:
        labels = stored["classification"]
    contract = protocol["reference_contract"]
    target, ignored = tuple(contract["target_class_ids"]), tuple(contract["ignored_class_ids"])
    positive, excluded = np.isin(labels, target), np.isin(labels, ignored)
    evaluations, rows = {}, []
    for mode in protocol["modes"]:
        with np.load(sample_root / "predictions" / mode / "mask.npz", allow_pickle=False) as stored:
            mask = stored["point_mask"]
        evaluation = evaluate_public_rail_mask(labels, mask, target, ignored)
        evaluations[mode] = evaluation
        _write_json(sample_root / "evaluation" / f"{mode}.json", evaluation)
        indices = {}
        for view, eligible in (("all_points", np.ones(len(labels), dtype=bool)),
                               ("annotated_only", ~excluded)):
            for key, selected in (("tp", positive & mask), ("fp", ~positive & mask),
                                  ("fn", positive & ~mask), ("tn", ~positive & ~mask)):
                indices[f"{view}_{key}"] = np.flatnonzero(eligible & selected)
            rows.append({"filename": sample["filename"], "parent_cloud": sample["parent_cloud"],
                         "mode": mode, "view": view, **{key: evaluation[view][key] for key in _METRICS}})
        indices["ignored_selected"] = np.flatnonzero(excluded & mask)
        _save_array(sample_root / "evaluation" / mode / "error_indices.npz", **indices)
    return evaluations, rows


def _report_text(report: dict) -> str:
    lines = ["# Public TRAIN fitted-center-spacing sensitivity", "",
             "Post-pilot development adaptation, not held-out validation. All modes and samples remain in the report.", "",
             "Broad pairing uses a fixed 1.35–1.90 m fitted-line center-spacing hypothesis; this is not a physical gauge measurement or an inferred railway standard. Only the maximum pair spacing changes from the preserved default extractor.", "",
             "Reference: numeric_class_1; its rail semantic mapping is NOT VERIFIED. Scores below are numeric-label agreement, not rail segmentation, survey, or connection accuracy.", "",
             f"{report['sample_count']} segment(s), {report['parent_cloud_count']} parent cloud(s). Pooled values are descriptive; points and segments are not treated as independent statistical units.", "",
             "| Sample | Mode | View | TP | FP | FN | TN | Precision | Recall | F1 | IoU |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for sample in report["samples"]:
        for mode in SPACING_MODES:
            for view in _VIEWS:
                metrics = sample["evaluation_by_mode"][mode][view]
                values = ["NA" if metrics[key] is None else str(metrics[key]) for key in _METRICS]
                lines.append(f"| {sample['source']['filename']} | {mode} | {view} | " + " | ".join(values) + " |")
    lines.extend(["", "All-points reference disagreements include intentionally unclassified class 0, which is not confirmed non-rail. Annotated-only excludes 0 and can be optimistic; both views and per-class counts are retained.", "",
                  "Predictions for every sample and mode were saved before any label agreement was evaluated. This internal lock does not erase prior exposure to the original pilot and is not external blind preregistration. It does not verify the semantic meaning of class 1.", "",
                  "Parent clouds are grouping labels, not proof of site independence. No spatial bootstrap over millions of correlated points, speed superiority, or human-review savings are claimed. No customer data, production settings, or reserved synthetic test layouts are used.", "",
                  "Source: [SemanticRail3D-V2](https://zenodo.org/records/15641832), Ghasemlou et al. (2025), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: geometric candidate extraction and numeric-reference evaluation. Original authors do not endorse this experiment.", ""])
    return "\n".join(lines)


def run_public_spacing_study(input_directory: str | Path, metadata_directory: str | Path,
                             receipt_directory: str | Path, protocol_path: str | Path,
                             output: str | Path) -> dict:
    from .public_rail_candidates import extract_public_rail_candidates

    destination = Path(output).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Spacing study output already exists: {destination}")
    protocol_bytes = Path(protocol_path).read_bytes()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    validate_public_spacing_protocol(protocol)
    receipts = verify_spacing_receipts(receipt_directory, protocol["samples"])
    provenance = {sample["filename"]: verify_public_metadata(metadata_directory, {
        "metadata_sha256": protocol["metadata_sha256"], "sample": sample
    }) for sample in protocol["samples"]}
    sources = _source_bindings()
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        _write_json(destination / "protocol_snapshot.json", protocol)
        _write_json(destination / "provenance.json", {"metadata_by_sample": provenance, "acquisition": receipts})
        summaries, timing = {}, {}
        for sample in protocol["samples"]:
            source = _file_in(input_directory, sample["filename"])
            sample_root = destination / "samples" / source.stem
            intake = inspect_public_las(source, sample["sha256"], sample_root / "intake.json",
                                        max_points=protocol["maximum_points"])
            if intake["decoded"]["point_count"] != sample["point_count"]:
                raise ValueError("Decoded point count differs from pinned sample")
            xyz, labels = load_pinned_public_arrays(source, sample["sha256"], protocol["maximum_points"])
            _save_array(sample_root / "inputs" / "xyz.npz", xyz=xyz)
            _save_array(sample_root / "ground_truth" / "classification.npz", classification=labels)
            del labels
            unchanged_xyz = xyz.copy()
            xyz.flags.writeable = False
            predictions, candidates = {}, {}
            timing[source.name] = {}
            for mode in protocol["modes"]:
                before = time.perf_counter()
                result = extract_public_rail_candidates(xyz, copy.deepcopy(protocol["policies"][mode]),
                                                        protocol["extractor_modes"][mode])
                timing[source.name][mode] = time.perf_counter() - before
                mask = result["point_mask"]
                if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != (len(xyz),):
                    raise ValueError("Extractor must return one explicit boolean per point")
                predictions[mode], candidates[mode] = mask.copy(), result["report"]
                _save_array(sample_root / "predictions" / mode / "mask.npz", point_mask=mask)
                _write_json(sample_root / "predictions" / mode / "candidates.json", result["report"])
            if not np.array_equal(xyz, unchanged_xyz):
                raise ValueError("Extractor mutated XYZ input")
            for mode in SPACING_MODES[1:]:
                if (candidates[mode]["candidate_pool"] != candidates["height_prominence"]["candidate_pool"]
                        or np.any(predictions[mode] & ~predictions["height_prominence"])):
                    raise ValueError("Both pair hypotheses must select only from the unchanged candidate pool")
            summaries[source.name] = {mode: {
                "selected_point_count": int(predictions[mode].sum()),
                "selected_line_count": len(candidates[mode]["selected_candidate_ids"]),
                "selected_pair_count": len(candidates[mode]["rail_pairs"]),
                "candidate_pool_count": len(candidates[mode]["candidate_pool"]),
                "status": candidates[mode]["status"],
                "diagnostics": candidates[mode]["diagnostics"],
            } for mode in SPACING_MODES}
            del xyz, unchanged_xyz, predictions, candidates
        locked_paths = sorted(path for path in (destination / "samples").rglob("*")
                              if path.is_file() and "predictions" in path.relative_to(destination).parts)
        lock_files = {path.relative_to(destination).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in locked_paths}
        _write_json(destination / "prediction_lock.json", {
            "status": "all_samples_all_modes_saved_before_any_reference_evaluation",
            "candidate_input_fields": ["xyz"], "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "input_sha256": {sample["filename"]: sample["sha256"] for sample in protocol["samples"]},
            "files": lock_files, "boundary": "Internal ordering record; original pilot was already viewed",
        })
        sample_reports, rows = [], []
        for sample in protocol["samples"]:
            evaluations, sample_rows = _evaluate_saved_sample(destination, sample, protocol)
            sample_reports.append({"source": sample, "candidate_summary": summaries[sample["filename"]],
                                   "evaluation_by_mode": evaluations})
            rows.extend(sample_rows)
        parents = sorted({sample["parent_cloud"] for sample in protocol["samples"]})
        grouped = {}
        for parent in parents:
            entries = [item for item in sample_reports if item["source"]["parent_cloud"] == parent]
            grouped[parent] = {"segment_count": len(entries), "evaluation_by_mode": {
                mode: {view: _pooled_counts([item["evaluation_by_mode"][mode][view] for item in entries])
                       for view in _VIEWS} for mode in SPACING_MODES}}
        report = {
            "schema_version": "railway.public-rail-spacing-report.v1", "status": "completed",
            "protocol_id": protocol["protocol_id"], "role": protocol["role"],
            "dataset": protocol["dataset"], "reference_contract": protocol["reference_contract"],
            "development_exposure": protocol["development_exposure"],
            "sample_count": len(sample_reports), "parent_cloud_count": len(parents),
            "total_point_count": sum(sample["point_count"] for sample in protocol["samples"]),
            "samples": sample_reports, "by_parent_cloud": grouped,
            "pooled_descriptive_evaluation": {mode: {
                view: _pooled_counts([item["evaluation_by_mode"][mode][view] for item in sample_reports])
                for view in _VIEWS} for mode in SPACING_MODES},
            "inference": {"confidence_intervals": None, "points_assumed_independent": False,
                          "parent_labels_establish_site_independence": False,
                          "reason": "bounded_convenience_training_sample_not_population_validation"},
            "customer_data_used": False, "held_out_test": False,
            "connection_accuracy_available": False, "absolute_accuracy_available": False,
        }
        with (destination / "metrics.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        _write_json(destination / "timings.json", {"detector_seconds_by_sample": timing,
                    "elapsed_seconds": time.perf_counter() - started,
                    "scope": "single_run_wall_clock_not_statistical_speed_or_human_time_comparison"})
        if sources != _source_bindings() or Path(protocol_path).read_bytes() != protocol_bytes:
            raise ValueError("Source code or protocol changed during spacing study")
        verify_spacing_receipts(receipt_directory, protocol["samples"])
        for sample in protocol["samples"]:
            verify_public_metadata(metadata_directory, {"sample": sample, "metadata_sha256": protocol["metadata_sha256"]})
            with _file_in(input_directory, sample["filename"]).open("rb") as stream:
                if _sha256(stream) != sample["sha256"]:
                    raise ValueError("Public sample changed during spacing study")
        if any(hashlib.sha256((destination / relative).read_bytes()).hexdigest() != digest
               for relative, digest in lock_files.items()):
            raise ValueError("Prediction bytes changed after the pre-evaluation lock")
        _write_json(destination / "report.json", report)
        with (destination / "report.md").open("x", encoding="utf-8") as stream:
            stream.write(_report_text(report))
        _write_json(destination / "manifest.json", {
            "schema_version": "railway.public-rail-spacing-manifest.v1", "status": "completed",
            "source_sha256": sources, "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "input_sha256": {sample["filename"]: sample["sha256"] for sample in protocol["samples"]},
            "receipt_sha256": receipts["receipts_sha256"],
            "environment": {"python": platform.python_version(), "system": platform.system(),
                            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "laspy")}},
            "files": [{"path": path.relative_to(destination).as_posix(), "bytes": path.stat().st_size,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                      for path in sorted(destination.rglob("*")) if path.is_file()],
            "manifest_self_hash_excluded": True,
        })
        return report
    except Exception as error:
        _write_json(destination / "run_failure.json", {"status": "failed", "exception_type": type(error).__name__})
        raise
