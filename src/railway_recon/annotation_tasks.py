from __future__ import annotations

import csv
import random
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .io import load_json, sha256_file, write_json

SEGMENT_PATTERN = re.compile(r"^s(?P<start>\d+)_(?P<end>\d+)m$")
OUTPUT_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{2,63}$")

FEATURE_COLUMNS = [
    "task_id",
    "segment_id",
    "evaluation_group_id",
    "source_candidate_id",
    "photo_sheet",
    "photo_tile",
    "geometry_sheet",
    "geometry_tile",
    "top_map",
    "center_x",
    "center_y",
    "minimum_z",
    "maximum_z",
    "height_m",
    "footprint_x_m",
    "footprint_y_m",
    "point_count",
    "cross_position_m",
    "nearest_track_distance_m",
    "trajectory_distance_m",
    "base_near_ground",
    "vertical_occupied_ratio",
    "vertical_longest_run_ratio",
    "evidence_camera_indices",
]

LABEL_COLUMNS = [
    "existence",
    "asset_type",
    "geometry_evaluable",
    "evidence_state",
    "reviewer_confidence",
    "ambiguity_reason",
    "notes",
]


def _parse_segment(segment_id: str) -> tuple[int, int]:
    match = SEGMENT_PATTERN.fullmatch(segment_id)
    if not match:
        raise ValueError(f"Invalid segment id {segment_id!r}; expected s000_50m")
    start, end = int(match.group("start")), int(match.group("end"))
    if end <= start:
        raise ValueError(f"Invalid segment range: {segment_id}")
    return start, end


def _copy_evidence(source: Path, target: Path) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target.as_posix()


def _relative_copy(
    source_reference: str,
    output_root: Path,
    segment_id: str,
    category: str,
) -> str:
    source = Path(source_reference).resolve()
    target = output_root / "evidence" / segment_id / category / source.name
    _copy_evidence(source, target)
    return target.relative_to(output_root).as_posix()


def _build_segment_tasks(
    segment_id: str, feature_report_path: Path, output_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    start, end = _parse_segment(segment_id)
    value = load_json(feature_report_path)
    candidates = value.get("candidates")
    if not isinstance(candidates, list):
        raise TypeError(f"Feature report has no candidate list: {feature_report_path}")
    if value.get("candidate_count") != len(candidates):
        raise ValueError(f"Candidate count mismatch in {feature_report_path}")

    photo_sheets = [
        _relative_copy(path, output_root, segment_id, "photo")
        for path in value.get("contact_sheets", [])
    ]
    geometry_sheets = [
        _relative_copy(path, output_root, segment_id, "geometry")
        for path in value.get("geometry_profile_sheets", [])
    ]
    top_map = _relative_copy(
        str(value["annotated_top_map"]), output_root, segment_id, "overview"
    )
    expected_photo_pages = (len(candidates) + 5) // 6
    expected_geometry_pages = (len(candidates) + 11) // 12
    if len(photo_sheets) != expected_photo_pages:
        raise ValueError(
            f"Photo evidence page mismatch for {segment_id}: "
            f"expected {expected_photo_pages}, found {len(photo_sheets)}"
        )
    if len(geometry_sheets) != expected_geometry_pages:
        raise ValueError(
            f"Geometry evidence page mismatch for {segment_id}: "
            f"expected {expected_geometry_pages}, found {len(geometry_sheets)}"
        )

    tasks: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        candidate_id = int(candidate["id"])
        camera_indices = [
            str(view["camera_index"])
            for view in candidate.get("evidence_views", [])
            if "camera_index" in view
        ]
        task = {
            "task_id": f"VANN-{start:03d}-{candidate_id:03d}",
            "segment_id": segment_id,
            "evaluation_group_id": f"station-000-200m-eval-{start:03d}-{end:03d}",
            "source_candidate_id": candidate_id,
            "photo_sheet": photo_sheets[index // 6],
            "photo_tile": index % 6 + 1,
            "geometry_sheet": geometry_sheets[index // 12],
            "geometry_tile": index % 12 + 1,
            "top_map": top_map,
            "center_x": candidate.get("center_x"),
            "center_y": candidate.get("center_y"),
            "minimum_z": candidate.get("minimum_z"),
            "maximum_z": candidate.get("maximum_z"),
            "height_m": candidate.get("height_m"),
            "footprint_x_m": candidate.get("footprint_x_m"),
            "footprint_y_m": candidate.get("footprint_y_m"),
            "point_count": candidate.get("point_count"),
            "cross_position_m": candidate.get("cross_position_m"),
            "nearest_track_distance_m": candidate.get("nearest_track_distance_m"),
            "trajectory_distance_m": candidate.get("trajectory_distance_m"),
            "base_near_ground": candidate.get("base_near_ground"),
            "vertical_occupied_ratio": candidate.get("vertical_occupied_ratio"),
            "vertical_longest_run_ratio": candidate.get("vertical_longest_run_ratio"),
            "evidence_camera_indices": ";".join(camera_indices),
            **{column: "" for column in LABEL_COLUMNS},
        }
        tasks.append(task)
    source_record = {
        "segment_id": segment_id,
        "feature_report": str(feature_report_path),
        "feature_report_sha256": sha256_file(feature_report_path),
        "candidate_count": len(tasks),
        "photo_sheet_count": len(photo_sheets),
        "geometry_sheet_count": len(geometry_sheets),
    }
    return tasks, source_record


def _write_reviewer_csv(path: Path, tasks: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FEATURE_COLUMNS + LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(tasks)


def _write_instructions(path: Path) -> None:
    text = """# 竖直候选双人盲标说明

本任务包不得与现有分类结果、最终模型或另一位标注者的答案同时查看。

## 标签

- existence: present / absent / uncertain
- asset_type: catenary_mast / canopy_column / building_edge / signal_or_lineside_pole / other_asset / false_positive / uncertain
- geometry_evaluable: yes / no
- evidence_state: observed / weak / inferred / unsupported
- reviewer_confidence: 0.0–1.0

先查看 photo_sheet 对应 photo_tile，再查看 geometry_sheet 对应 geometry_tile，最后用 top_map 核对空间上下文。无法确认时必须选择 uncertain，不要根据铁路规律强行补成某一类别。
"""
    path.write_text(text, encoding="utf-8")


def create_vertical_annotation_package(
    benchmark_root: str | Path,
    sources: list[tuple[str, str | Path]],
    output_name: str = "vertical_candidates_blind_v1",
    seed: int = 20260826,
) -> Path:
    root = Path(benchmark_root).resolve()
    if not OUTPUT_NAME_PATTERN.fullmatch(output_name):
        raise ValueError("output_name must use lowercase letters, digits, '-' or '_'")
    output_root = root / "annotations" / output_name
    if output_root.exists():
        raise FileExistsError(f"Annotation package already exists: {output_root}")
    output_root.mkdir(parents=True)

    tasks: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    seen_segments: set[str] = set()
    for segment_id, reference in sources:
        if segment_id in seen_segments:
            raise ValueError(f"Duplicate segment source: {segment_id}")
        seen_segments.add(segment_id)
        segment_tasks, source_record = _build_segment_tasks(
            segment_id, Path(reference).resolve(), output_root
        )
        tasks.extend(segment_tasks)
        source_records.append(source_record)

    task_ids = [task["task_id"] for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Generated duplicate task IDs")
    reviewer_a = list(tasks)
    reviewer_b = list(tasks)
    random.Random(seed).shuffle(reviewer_a)
    random.Random(seed + 1).shuffle(reviewer_b)
    _write_reviewer_csv(output_root / "reviewer_a.csv", reviewer_a)
    _write_reviewer_csv(output_root / "reviewer_b.csv", reviewer_b)
    _write_instructions(output_root / "INSTRUCTIONS_CN.md")
    manifest = {
        "schema_version": "railway.annotation-package.v1",
        "package_id": output_name,
        "created_at": datetime.now(UTC).isoformat(),
        "blind_to_existing_classification": True,
        "task_count": len(tasks),
        "reviewer_count": 2,
        "shuffle_seeds": {"reviewer_a": seed, "reviewer_b": seed + 1},
        "sources": source_records,
        "reviewer_files": ["reviewer_a.csv", "reviewer_b.csv"],
        "allowed_asset_types": [
            "catenary_mast",
            "canopy_column",
            "building_edge",
            "signal_or_lineside_pole",
            "other_asset",
            "false_positive",
            "uncertain",
        ],
        "forbidden_inputs": [
            "existing classification.json",
            "final asset registry",
            "final model",
            "the other reviewer's answers",
        ],
    }
    write_json(output_root / "manifest.json", manifest)
    return output_root


def validate_vertical_annotation_package(package_root: str | Path) -> dict[str, Any]:
    root = Path(package_root).resolve()
    manifest = load_json(root / "manifest.json")
    errors: list[str] = []
    reviewer_rows: dict[str, list[dict[str, str]]] = {}
    forbidden_tokens = ("prediction", "classification", "existing_label", "final_model")
    for reviewer_file in manifest.get("reviewer_files", []):
        path = root / reviewer_file
        if not path.is_file():
            errors.append(f"Missing reviewer file: {reviewer_file}")
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
        leaking_headers = [
            field
            for field in fieldnames
            if any(token in field.lower() for token in forbidden_tokens)
        ]
        if leaking_headers:
            errors.append(f"{reviewer_file} has forbidden columns: {leaking_headers}")
        for row_index, row in enumerate(rows, start=2):
            nonempty_labels = [column for column in LABEL_COLUMNS if row.get(column, "").strip()]
            if nonempty_labels:
                errors.append(
                    f"{reviewer_file}:{row_index} has prefilled labels: {nonempty_labels}"
                )
            for evidence_column in ("photo_sheet", "geometry_sheet", "top_map"):
                reference = row.get(evidence_column, "")
                if not reference or not (root / reference).is_file():
                    errors.append(
                        f"{reviewer_file}:{row_index} missing evidence {evidence_column}"
                    )
        reviewer_rows[reviewer_file] = rows

    task_sets = {
        name: {row.get("task_id", "") for row in rows} for name, rows in reviewer_rows.items()
    }
    expected_count = int(manifest.get("task_count", 0))
    for name, tasks in task_sets.items():
        if len(tasks) != expected_count or "" in tasks:
            errors.append(
                f"{name} has {len(tasks)} unique valid task IDs; expected {expected_count}"
            )
    if task_sets and len({frozenset(tasks) for tasks in task_sets.values()}) != 1:
        errors.append("Reviewer files do not contain the same task set")
    reviewer_names = list(reviewer_rows)
    if len(reviewer_names) == 2:
        first_order = [row["task_id"] for row in reviewer_rows[reviewer_names[0]]]
        second_order = [row["task_id"] for row in reviewer_rows[reviewer_names[1]]]
        if first_order == second_order:
            errors.append("Reviewer files use identical task order")

    for source in manifest.get("sources", []):
        path = Path(source["feature_report"])
        if not path.is_file():
            errors.append(f"Missing source feature report: {path}")
        elif sha256_file(path) != source.get("feature_report_sha256"):
            errors.append(f"Source feature report hash changed: {path}")

    result = {
        "schema_version": "railway.annotation-package-validation.v1",
        "package_id": manifest.get("package_id"),
        "generated_at": datetime.now(UTC).isoformat(),
        "passed": not errors,
        "task_count": expected_count,
        "reviewer_file_count": len(reviewer_rows),
        "evidence_file_count": len(list((root / "evidence").rglob("*.*"))),
        "errors": errors,
    }
    write_json(root / "validation.json", result)
    return result
