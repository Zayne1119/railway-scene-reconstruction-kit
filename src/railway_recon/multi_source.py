from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from .camera import camera_trajectory, load_camera_rows
from .config import ProjectConfig
from .io import load_json, sha256_file, sha256_json, write_json
from .pointcloud import inspect_point_cloud


@dataclass(frozen=True)
class PointCloudSource:
    source_id: str
    path: Path
    priority: int


def configured_point_cloud_sources(project: ProjectConfig) -> list[PointCloudSource]:
    inputs = project.value.get("inputs", {})
    multiple = inputs.get("point_clouds")
    single = inputs.get("point_cloud")
    if multiple is not None and single:
        raise ValueError("Configure inputs.point_cloud or inputs.point_clouds, not both")
    if multiple is None:
        if not single:
            raise ValueError("No point-cloud input is configured")
        return [PointCloudSource("primary", project.resolve(single), 0)]
    if not isinstance(multiple, list) or not multiple:
        raise ValueError("inputs.point_clouds must be a non-empty array")
    sources: list[PointCloudSource] = []
    seen: set[str] = set()
    for position, value in enumerate(multiple):
        if not isinstance(value, dict):
            raise TypeError(f"inputs.point_clouds[{position}] must be an object")
        source_id = str(value.get("id", "")).strip()
        source_path = value.get("path")
        if not source_id or not source_path:
            raise ValueError(f"inputs.point_clouds[{position}] requires id and path")
        if source_id in seen:
            raise ValueError(f"Duplicate point-cloud source id: {source_id}")
        seen.add(source_id)
        sources.append(
            PointCloudSource(
                source_id,
                project.resolve(str(source_path)),
                int(value.get("priority", position)),
            )
        )
    return sources


def input_source_manifest_path(project: ProjectConfig) -> Path:
    configured = project.value.get("workspace", {}).get("input_source_manifest")
    if configured:
        return project.workspace_path("input_source_manifest")
    return project.workspace_path("manifests") / "input_sources.generated.json"


def filtered_camera_csv_path(project: ProjectConfig) -> Path:
    configured = project.value.get("workspace", {}).get("filtered_camera_csv")
    if configured:
        return project.workspace_path("filtered_camera_csv")
    return project.workspace_path("manifests") / "cameras.filtered.csv"


def _camera_inside_header(
    camera: dict[str, Any], point_cloud: dict[str, Any], margin_m: float
) -> bool:
    mins = point_cloud["mins"]
    maxs = point_cloud["maxs"]
    return (
        float(mins[0]) - margin_m <= float(camera["x"]) <= float(maxs[0]) + margin_m
        and float(mins[1]) - margin_m <= float(camera["y"]) <= float(maxs[1]) + margin_m
    )


def _write_filtered_camera_csv(
    source_rows: list[dict[str, str]], selected_indexes: set[int], output: Path
) -> None:
    rows = [row for row in source_rows if int(row["index"]) in selected_indexes]
    if not rows:
        raise ValueError("No camera rows fall inside the configured point-cloud coverage")
    fieldnames = [name for name in rows[0] if name]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


def prepare_input_sources(
    project: ProjectConfig,
    full_hash: bool = False,
    camera_bbox_margin_m: float = 0.0,
) -> dict[str, Any]:
    if camera_bbox_margin_m < 0:
        raise ValueError("camera_bbox_margin_m must be non-negative")
    sources = configured_point_cloud_sources(project)
    camera_path = project.input_path("camera_csv")
    if camera_path is None or not camera_path.is_file():
        raise FileNotFoundError(f"Camera CSV not found: {camera_path}")
    source_rows = load_camera_rows(camera_path)
    full_trajectory = camera_trajectory(source_rows)
    row_positions = {
        int(row["index"]): position for position, row in enumerate(source_rows)
    }
    if len(row_positions) != len(source_rows):
        raise ValueError("Camera CSV contains duplicate indexes")

    inspected: list[dict[str, Any]] = []
    membership_indexes_by_id: dict[str, list[int]] = {}
    selected_indexes: set[int] = set()
    for source in sources:
        if not source.path.is_file():
            raise FileNotFoundError(f"Point-cloud source not found: {source.path}")
        point_cloud = inspect_point_cloud(source.path)
        membership = [
            item
            for item in full_trajectory
            if _camera_inside_header(item, point_cloud, camera_bbox_margin_m)
        ]
        if not membership:
            raise ValueError(
                f"Point-cloud source contains no camera positions in its XY header bounds: "
                f"{source.source_id}"
            )
        indexes = [int(item["index"]) for item in membership]
        membership_positions = sorted(row_positions[index] for index in indexes)
        membership_indexes_by_id[source.source_id] = indexes
        selected_indexes.update(indexes)
        inspected.append(
            {
                "id": source.source_id,
                "path": str(source.path),
                "priority": source.priority,
                "file_size_bytes": int(point_cloud["file_size_bytes"]),
                "point_count": int(point_cloud["point_count"]),
                "sha256": (
                    sha256_file(source.path)
                    if full_hash
                    else "not_computed_use_--full-hash_for_release"
                ),
                "header": point_cloud,
                "camera_index_from": min(indexes),
                "camera_index_to": max(indexes),
                "camera_count": len(indexes),
                "camera_coverage_gap_count": sum(
                    current != previous + 1
                    for previous, current in pairwise(membership_positions)
                ),
            }
        )

    selected_rows = [
        row for row in source_rows if int(row["index"]) in selected_indexes
    ]
    selected_trajectory = camera_trajectory(selected_rows)
    selected_index_to_chainage = {
        int(item["index"]): float(item["distance_m"]) for item in selected_trajectory
    }
    selected_positions = sorted(row_positions[index] for index in selected_indexes)
    coverage_gap_count = sum(
        current != previous + 1
        for previous, current in pairwise(selected_positions)
    )

    for source in inspected:
        chainages = [
            selected_index_to_chainage[index]
            for index in membership_indexes_by_id[str(source["id"])]
            if index in selected_index_to_chainage
        ]
        source["context_chainage_start_m"] = min(chainages)
        source["context_chainage_end_m"] = max(chainages)

    inspected.sort(
        key=lambda item: (float(item["context_chainage_start_m"]), int(item["priority"]))
    )
    seams: list[dict[str, Any]] = []
    boundaries: list[float] = []
    for left, right in pairwise(inspected):
        left_end = float(left["context_chainage_end_m"])
        right_start = float(right["context_chainage_start_m"])
        if right_start <= left_end:
            overlap_start = right_start
            overlap_end = left_end
            boundary = (overlap_start + overlap_end) / 2.0
            status = "overlap_context_available"
            gap_m = 0.0
        else:
            overlap_start = None
            overlap_end = None
            boundary = (left_end + right_start) / 2.0
            status = "coverage_gap_review_required"
            gap_m = right_start - left_end
        boundaries.append(boundary)
        seams.append(
            {
                "id": f"{left['id']}--{right['id']}",
                "left_source_id": left["id"],
                "right_source_id": right["id"],
                "overlap_chainage_start_m": overlap_start,
                "overlap_chainage_end_m": overlap_end,
                "ownership_boundary_chainage_m": boundary,
                "gap_m": gap_m,
                "status": status,
            }
        )

    total = float(selected_trajectory[-1]["distance_m"])
    for position, source in enumerate(inspected):
        source["core_chainage_start_m"] = 0.0 if position == 0 else boundaries[position - 1]
        source["core_chainage_end_m"] = total if position == len(inspected) - 1 else boundaries[position]

    filtered_camera_path = filtered_camera_csv_path(project)
    _write_filtered_camera_csv(source_rows, selected_indexes, filtered_camera_path)
    manifest_path = input_source_manifest_path(project)
    result = {
        "schema_version": "railway.input-sources.v1",
        "project_id": project.project_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "assignment_method": "camera_xy_header_membership_core_midpoint_v1",
        "camera_bbox_margin_m": camera_bbox_margin_m,
        "camera": {
            "source_csv": str(camera_path),
            "source_csv_sha256": sha256_file(camera_path),
            "source_row_count": len(source_rows),
            "filtered_csv": str(filtered_camera_path),
            "filtered_row_count": len(selected_rows),
            "filtered_csv_sha256": sha256_file(filtered_camera_path),
            "first_index": int(selected_trajectory[0]["index"]),
            "last_index": int(selected_trajectory[-1]["index"]),
            "trajectory_length_m": total,
            "coverage_gap_count": coverage_gap_count,
        },
        "sources": inspected,
        "seams": seams,
        "status": (
            "ready_for_segment_planning"
            if coverage_gap_count == 0
            and all(item["camera_coverage_gap_count"] == 0 for item in inspected)
            and all(item["status"] != "coverage_gap_review_required" for item in seams)
            else "review_required_before_segment_planning"
        ),
        "limitations": [
            "LAS/LAZ header boxes select camera coverage but do not prove duplicate surface overlap.",
            "Seam geometry must be validated on rail/track topology before cross-source mesh export.",
            "Exactly one source owns each core chainage; adjacent sources remain context only.",
        ],
    }
    result["manifest_content_sha256"] = sha256_json(result)
    write_json(manifest_path, result)
    result["output_manifest_path"] = str(manifest_path)
    result["filtered_camera_path"] = str(filtered_camera_path)
    return result


def load_input_source_manifest(project: ProjectConfig) -> dict[str, Any]:
    path = input_source_manifest_path(project)
    if not path.is_file():
        raise FileNotFoundError(
            f"Input source manifest not found: {path}; run prepare-inputs first"
        )
    manifest = load_json(path)
    if manifest.get("schema_version") != "railway.input-sources.v1":
        raise ValueError("Unsupported input source manifest schema")
    if manifest.get("project_id") != project.project_id:
        raise ValueError("Input source manifest belongs to a different project")
    claimed_hash = manifest.get("manifest_content_sha256")
    hash_payload = dict(manifest)
    hash_payload.pop("manifest_content_sha256", None)
    if claimed_hash != sha256_json(hash_payload):
        raise ValueError("Input source manifest content hash is invalid")
    if manifest.get("status") != "ready_for_segment_planning":
        raise ValueError(
            "Input source manifest is not ready for segment planning: "
            f"{manifest.get('status')}"
        )
    camera = manifest.get("camera", {})
    configured_camera = project.input_path("camera_csv")
    if configured_camera is None or not configured_camera.is_file():
        raise FileNotFoundError(f"Camera CSV not found: {configured_camera}")
    if Path(str(camera.get("source_csv", ""))).resolve() != configured_camera.resolve():
        raise ValueError("Configured camera CSV differs from the input source manifest")
    if sha256_file(configured_camera) != camera.get("source_csv_sha256"):
        raise ValueError("Camera CSV changed after prepare-inputs")
    filtered = Path(str(camera.get("filtered_csv", "")))
    if not filtered.is_file():
        raise FileNotFoundError(f"Filtered camera CSV not found: {filtered}")
    if sha256_file(filtered) != camera.get("filtered_csv_sha256"):
        raise ValueError("Filtered camera CSV hash differs from the input source manifest")
    configured_sources = {
        source.source_id: source for source in configured_point_cloud_sources(project)
    }
    manifest_sources = {
        str(source.get("id")): source for source in manifest.get("sources", [])
    }
    if set(configured_sources) != set(manifest_sources):
        raise ValueError("Configured point-cloud sources differ from the input source manifest")
    for source_id, configured_source in configured_sources.items():
        record = manifest_sources[source_id]
        recorded_path = Path(str(record.get("path", ""))).resolve()
        if recorded_path != configured_source.path.resolve():
            raise ValueError(f"Point-cloud source path changed: {source_id}")
        if not configured_source.path.is_file():
            raise FileNotFoundError(f"Point-cloud source not found: {configured_source.path}")
        if configured_source.path.stat().st_size != int(record["file_size_bytes"]):
            raise ValueError(f"Point-cloud source size changed: {source_id}")
    return manifest


def effective_camera_csv_path(project: ProjectConfig) -> Path:
    """Return the route camera chain used by production geometry modules."""

    if "point_clouds" in project.value.get("inputs", {}):
        manifest = load_input_source_manifest(project)
        return Path(str(manifest["camera"]["filtered_csv"]))
    camera_path = project.input_path("camera_csv")
    if camera_path is None or not camera_path.is_file():
        raise FileNotFoundError(f"Camera CSV not found: {camera_path}")
    return camera_path


def assign_segment_sources(
    segments: list[dict[str, Any]], manifest: dict[str, Any]
) -> None:
    sources = manifest.get("sources", [])
    if not sources:
        raise ValueError("Input source manifest contains no sources")
    for segment in segments:
        start = float(segment["chainage_start_m"])
        end = float(segment["chainage_end_m"])
        midpoint = (start + end) / 2.0
        primary = next(
            (
                source
                for position, source in enumerate(sources)
                if float(source["core_chainage_start_m"]) <= midpoint
                and (
                    midpoint < float(source["core_chainage_end_m"])
                    or (
                        position == len(sources) - 1
                        and midpoint <= float(source["core_chainage_end_m"])
                    )
                )
            ),
            None,
        )
        if primary is None:
            raise ValueError(f"No primary point-cloud source owns segment {segment['id']}")
        context = [
            str(source["id"])
            for source in sources
            if float(source["context_chainage_start_m"]) <= end
            and float(source["context_chainage_end_m"]) >= start
        ]
        segment["primary_source_id"] = str(primary["id"])
        segment["context_source_ids"] = context
        segment["source_assignment_method"] = "core_chainage_midpoint_v1"
