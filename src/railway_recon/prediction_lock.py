from __future__ import annotations

import csv
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .io import load_json, sha256_file, write_json

SEGMENT_PATTERN = re.compile(r"^s(?P<start>\d+)_(?P<end>\d+)m$")
OUTPUT_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{2,63}$")
PREDICTION_COLUMNS = [
    "task_id",
    "segment_id",
    "source_candidate_id",
    "prediction_existence",
    "prediction_type",
    "confidence",
    "evidence_level",
    "legacy_evidence_source",
    "legacy_asset_id",
]


def _segment_start(segment_id: str) -> int:
    match = SEGMENT_PATTERN.fullmatch(segment_id)
    if not match:
        raise ValueError(f"Invalid segment id {segment_id!r}; expected s000_50m")
    return int(match.group("start"))


def _read_annotation_task_ids(package_root: Path) -> set[str]:
    manifest = load_json(package_root / "manifest.json")
    reviewer_files = manifest.get("reviewer_files", [])
    if not reviewer_files:
        raise ValueError(f"Annotation package has no reviewer files: {package_root}")
    with (package_root / reviewer_files[0]).open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        return {row["task_id"] for row in csv.DictReader(stream)}


def _source_predictions(
    segment_id: str, classification_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    start = _segment_start(segment_id)
    value = load_json(classification_path)
    redetection = value.get("continuity_redetection")
    if not isinstance(redetection, dict) or not isinstance(redetection.get("candidates"), list):
        raise TypeError(f"Classification has no continuity candidate list: {classification_path}")
    rows: list[dict[str, Any]] = []
    for candidate in redetection["candidates"]:
        semantic_class = str(candidate["semantic_class"])
        present = semantic_class != "false_positive"
        rows.append(
            {
                "task_id": f"VANN-{start:03d}-{int(candidate['id']):03d}",
                "segment_id": segment_id,
                "source_candidate_id": int(candidate["id"]),
                "prediction_existence": "present" if present else "absent",
                "prediction_type": semantic_class if present else "",
                "confidence": float(candidate["classification_confidence"]),
                "evidence_level": (
                    "photo_interpreted" if candidate.get("evidence_views") else "rule_inferred"
                ),
                "legacy_evidence_source": candidate.get("evidence_source", ""),
                "legacy_asset_id": candidate.get("asset_id", ""),
            }
        )
    declared_count = redetection.get("candidate_count")
    if declared_count is not None and int(declared_count) != len(rows):
        raise ValueError(f"Classification candidate count mismatch: {classification_path}")
    source_record = {
        "segment_id": segment_id,
        "classification_report": str(classification_path),
        "classification_report_sha256": sha256_file(classification_path),
        "prediction_count": len(rows),
    }
    return rows, source_record


def lock_vertical_predictions(
    benchmark_root: str | Path,
    annotation_package: str | Path,
    sources: list[tuple[str, str | Path]],
    output_name: str = "vertical_predictions_legacy_hitl_v1",
) -> Path:
    root = Path(benchmark_root).resolve()
    package_root = Path(annotation_package).resolve()
    if not OUTPUT_NAME_PATTERN.fullmatch(output_name):
        raise ValueError("output_name must use lowercase letters, digits, '-' or '_'")
    output_root = root / "locked_predictions" / output_name
    if output_root.exists():
        raise FileExistsError(f"Locked prediction directory already exists: {output_root}")

    predictions: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    seen_segments: set[str] = set()
    for segment_id, reference in sources:
        if segment_id in seen_segments:
            raise ValueError(f"Duplicate segment source: {segment_id}")
        seen_segments.add(segment_id)
        rows, source_record = _source_predictions(segment_id, Path(reference).resolve())
        predictions.extend(rows)
        source_records.append(source_record)

    prediction_ids = [row["task_id"] for row in predictions]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("Locked predictions contain duplicate task IDs")
    annotation_ids = _read_annotation_task_ids(package_root)
    missing = sorted(annotation_ids - set(prediction_ids))
    extra = sorted(set(prediction_ids) - annotation_ids)
    if missing or extra:
        raise ValueError(
            f"Prediction/annotation task mismatch: missing={len(missing)}, extra={len(extra)}"
        )

    output_root.mkdir(parents=True)
    prediction_path = output_root / "predictions.csv"
    with prediction_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerows(sorted(predictions, key=lambda item: item["task_id"]))
    distribution: dict[str, int] = {}
    for row in predictions:
        label = row["prediction_type"] or "absent_false_positive"
        distribution[label] = distribution.get(label, 0) + 1
    manifest = {
        "schema_version": "railway.locked-predictions.v1",
        "prediction_set_id": output_name,
        "locked_at": datetime.now(UTC).isoformat(),
        "execution_mode": "HITL",
        "eligible_as_fully_automatic_baseline": False,
        "method_note": (
            "Legacy production classification includes geometry, panorama interpretation, "
            "railway rules and prior human decisions."
        ),
        "annotation_package_id": load_json(package_root / "manifest.json")["package_id"],
        "annotation_package_manifest_sha256": sha256_file(package_root / "manifest.json"),
        "prediction_count": len(predictions),
        "class_distribution": dict(sorted(distribution.items())),
        "sources": source_records,
        "prediction_file": "predictions.csv",
        "prediction_file_sha256": sha256_file(prediction_path),
        "locked_before_ground_truth": True,
    }
    write_json(output_root / "manifest.json", manifest)
    return output_root
