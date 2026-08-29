from __future__ import annotations

import csv
import random
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .io import load_json, sha256_file, write_json

OUTPUT_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{2,63}$")
FEATURE_COLUMNS = [
    "task_id",
    "scene_id",
    "segment_id",
    "block_id",
    "chainage_m",
    "cross_section_image",
    "plan_strip_image",
    "point_count",
    "cross_min_m",
    "cross_max_m",
    "z_min_m",
    "z_max_m",
    "slice_half_width_m",
]
LABEL_COLUMNS = [
    "observable",
    "visible_track_count",
    "rail_head_cross_positions_m",
    "rail_head_z_m",
    "track_pairs",
    "topology_event",
    "reviewer_confidence",
    "ambiguity_reason",
    "notes",
]


def _copy_evidence(source: Path, output_root: Path, task_id: str, role: str) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    target = output_root / "evidence" / task_id / f"{role}{source.suffix.lower()}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target.relative_to(output_root).as_posix()


def _neutral_manifest(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if value.get("schema_version") != "railway.rail-neutral-evidence.v1":
        raise ValueError("Expected railway.rail-neutral-evidence.v1")
    if value.get("evidence_source") != "raw_point_cloud":
        raise ValueError("Rail truth evidence must come from raw_point_cloud")
    if value.get("model_overlay") is not False:
        raise ValueError("Rail truth evidence must declare model_overlay=false")
    if value.get("candidate_overlay") is not False:
        raise ValueError("Rail truth evidence must declare candidate_overlay=false")
    stations = value.get("stations")
    if not isinstance(stations, list) or not stations:
        raise ValueError("Neutral evidence manifest contains no stations")
    return value


def _write_reviewer_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FEATURE_COLUMNS + LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_instructions(path: Path) -> None:
    path.write_text(
        """# 轨道几何双人盲标说明

只能查看任务包中的原始点云横断面和局部俯视图，不得查看现有模型、候选检测结果、质量热力图或另一位标注者的答案。

## 字段

- `observable`: yes / no；点密度不足或遮挡严重时填 no。
- `visible_track_count`: 当前断面可确认的股道数量。
- `rail_head_cross_positions_m`: 从左到右填写轨头横向坐标，分号分隔。
- `rail_head_z_m`: 与上一字段一一对应的轨顶高程，分号分隔。
- `track_pairs`: 轨头序号配对，例如 `1-2;3-4`。
- `topology_event`: normal / turnout / crossing / end / uncertain。
- `reviewer_confidence`: 0.0–1.0。

无法确认时必须标为不可观测或 uncertain，不得按标准轨距、既有模型或铁路规则补写坐标。
""",
        encoding="utf-8",
    )


def create_rail_truth_annotation_package(
    benchmark_root: str | Path,
    neutral_evidence_manifests: list[str | Path],
    output_name: str = "rail_geometry_blind_v1",
    seed: int = 20260827,
) -> Path:
    root = Path(benchmark_root).resolve()
    if not OUTPUT_NAME_PATTERN.fullmatch(output_name):
        raise ValueError("output_name must use lowercase letters, digits, '-' or '_'")
    output_root = root / "annotations" / output_name
    if output_root.exists():
        raise FileExistsError(f"Annotation package already exists: {output_root}")
    output_root.mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    task_ids: set[str] = set()
    for manifest_reference in neutral_evidence_manifests:
        manifest_path = Path(manifest_reference).resolve()
        value = _neutral_manifest(manifest_path)
        scene_id = str(value["scene_id"])
        for station in value["stations"]:
            task_id = str(station["task_id"])
            if task_id in task_ids:
                raise ValueError(f"Duplicate rail truth task id: {task_id}")
            task_ids.add(task_id)
            cross_section = _copy_evidence(
                Path(str(station["cross_section_image"])).resolve(),
                output_root,
                task_id,
                "cross-section",
            )
            plan_strip = _copy_evidence(
                Path(str(station["plan_strip_image"])).resolve(),
                output_root,
                task_id,
                "plan-strip",
            )
            rows.append(
                {
                    "task_id": task_id,
                    "scene_id": scene_id,
                    "segment_id": station["segment_id"],
                    "block_id": station["block_id"],
                    "chainage_m": station["chainage_m"],
                    "cross_section_image": cross_section,
                    "plan_strip_image": plan_strip,
                    "point_count": station["point_count"],
                    "cross_min_m": station["cross_min_m"],
                    "cross_max_m": station["cross_max_m"],
                    "z_min_m": station["z_min_m"],
                    "z_max_m": station["z_max_m"],
                    "slice_half_width_m": station["slice_half_width_m"],
                    **{column: "" for column in LABEL_COLUMNS},
                }
            )
        sources.append(
            {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
                "scene_id": scene_id,
                "station_count": len(value["stations"]),
            }
        )

    reviewer_a = list(rows)
    reviewer_b = list(rows)
    random.Random(seed).shuffle(reviewer_a)
    random.Random(seed + 1).shuffle(reviewer_b)
    if len(reviewer_a) > 1 and [item["task_id"] for item in reviewer_a] == [
        item["task_id"] for item in reviewer_b
    ]:
        reviewer_b = reviewer_b[1:] + reviewer_b[:1]
    _write_reviewer_csv(output_root / "reviewer_a.csv", reviewer_a)
    _write_reviewer_csv(output_root / "reviewer_b.csv", reviewer_b)
    _write_instructions(output_root / "INSTRUCTIONS_CN.md")
    manifest = {
        "schema_version": "railway.rail-truth-annotation-package.v1",
        "package_id": output_name,
        "created_at": datetime.now(UTC).isoformat(),
        "blind_to_existing_model": True,
        "evidence_source": "raw_point_cloud",
        "model_overlay": False,
        "candidate_overlay": False,
        "task_count": len(rows),
        "reviewer_count": 2,
        "reviewer_files": ["reviewer_a.csv", "reviewer_b.csv"],
        "shuffle_seeds": {"reviewer_a": seed, "reviewer_b": seed + 1},
        "sources": sources,
        "forbidden_inputs": [
            "rail candidates",
            "TrackGraph",
            "final model",
            "fit heatmap",
            "the other reviewer's answers",
        ],
    }
    write_json(output_root / "manifest.json", manifest)
    return output_root


def validate_rail_truth_annotation_package(package_root: str | Path) -> dict[str, Any]:
    root = Path(package_root).resolve()
    manifest = load_json(root / "manifest.json")
    errors: list[str] = []
    if manifest.get("schema_version") != "railway.rail-truth-annotation-package.v1":
        errors.append("Unsupported package schema")
    for key in ("blind_to_existing_model",):
        if manifest.get(key) is not True:
            errors.append(f"Manifest must declare {key}=true")
    for key in ("model_overlay", "candidate_overlay"):
        if manifest.get(key) is not False:
            errors.append(f"Manifest must declare {key}=false")

    task_sets: list[set[str]] = []
    task_orders: list[list[str]] = []
    for reviewer_file in manifest.get("reviewer_files", []):
        path = root / str(reviewer_file)
        if not path.is_file():
            errors.append(f"Missing reviewer file: {reviewer_file}")
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        task_sets.append({str(row.get("task_id", "")) for row in rows})
        task_orders.append([str(row.get("task_id", "")) for row in rows])
        for index, row in enumerate(rows, start=2):
            prefilled = [name for name in LABEL_COLUMNS if str(row.get(name, "")).strip()]
            if prefilled:
                errors.append(f"{reviewer_file}:{index} has prefilled labels: {prefilled}")
            for field in ("cross_section_image", "plan_strip_image"):
                reference = str(row.get(field, ""))
                if not reference or not (root / reference).is_file():
                    errors.append(f"{reviewer_file}:{index} missing {field}")
    expected = int(manifest.get("task_count", 0))
    for index, tasks in enumerate(task_sets):
        if len(tasks) != expected or "" in tasks:
            errors.append(f"Reviewer {index + 1} task set is incomplete")
    if len(task_sets) == 2 and task_sets[0] != task_sets[1]:
        errors.append("Reviewer files do not contain identical task sets")
    if len(task_orders) == 2 and len(task_orders[0]) > 1 and task_orders[0] == task_orders[1]:
        errors.append("Reviewer files use identical task order")
    for source in manifest.get("sources", []):
        path = Path(str(source["path"]))
        if not path.is_file() or sha256_file(path) != source.get("sha256"):
            errors.append(f"Neutral evidence source changed: {path}")

    result = {
        "schema_version": "railway.rail-truth-annotation-validation.v1",
        "package_id": manifest.get("package_id"),
        "generated_at": datetime.now(UTC).isoformat(),
        "passed": not errors,
        "task_count": expected,
        "reviewer_file_count": len(task_sets),
        "evidence_file_count": len(list((root / "evidence").rglob("*.*"))),
        "errors": errors,
    }
    write_json(root / "validation.json", result)
    return result
