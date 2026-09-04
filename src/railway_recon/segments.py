from __future__ import annotations

import copy
import os
import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import laspy
import numpy as np

from .camera import camera_trajectory, load_camera_rows
from .config import ProjectConfig
from .io import write_json
from .multi_source import (
    assign_segment_sources,
    input_source_manifest_path,
    load_input_source_manifest,
)


def _segment_intervals(
    total_m: float,
    default_length_m: float,
    adaptive_zones: list[dict[str, Any]],
) -> list[tuple[float, float]]:
    if total_m <= 0 or default_length_m <= 0:
        raise ValueError("trajectory and default segment length must be positive")
    zones = sorted(adaptive_zones, key=lambda item: float(item["start_m"]))
    previous_end = 0.0
    for zone in zones:
        start = float(zone["start_m"])
        end = float(zone["end_m"])
        length = float(zone["length_m"])
        if start < 0 or end > total_m or end <= start or length <= 0:
            raise ValueError(f"Invalid adaptive segment length zone: {zone}")
        if start < previous_end - 1e-9:
            raise ValueError("Adaptive segment length zones must not overlap")
        previous_end = end

    intervals: list[tuple[float, float]] = []

    def append_range(start_m: float, end_m: float, length_m: float) -> None:
        local: list[tuple[float, float]] = []
        cursor = start_m
        while cursor < end_m - 1e-9:
            boundary = min(end_m, cursor + length_m)
            local.append((cursor, boundary))
            cursor = boundary
        minimum_tail_m = min(1.0, length_m * 0.1)
        if len(local) >= 2 and local[-1][1] - local[-1][0] < minimum_tail_m:
            local[-2] = (local[-2][0], local[-1][1])
            local.pop()
        intervals.extend(local)

    cursor = 0.0
    for zone in zones:
        start = float(zone["start_m"])
        end = float(zone["end_m"])
        append_range(cursor, start, default_length_m)
        append_range(start, end, float(zone["length_m"]))
        cursor = end
    append_range(cursor, total_m, default_length_m)
    return intervals


def plan_segments(project: ProjectConfig) -> dict[str, Any]:
    multi_source = "point_clouds" in project.value.get("inputs", {})
    source_manifest: dict[str, Any] | None = None
    if multi_source:
        source_manifest = load_input_source_manifest(project)
        camera_path = Path(str(source_manifest["camera"]["filtered_csv"]))
    else:
        camera_path = project.input_path("camera_csv")
    if camera_path is None or not camera_path.is_file():
        raise FileNotFoundError(f"Camera CSV not found: {camera_path}")
    trajectory = camera_trajectory(load_camera_rows(camera_path))
    settings = project.value["segmentation"]
    length = float(settings["length_m"])
    padding = float(settings["corridor_half_width_m"])
    z_below = float(settings["z_below_camera_m"])
    z_above = float(settings["z_above_camera_m"])
    total = float(trajectory[-1]["distance_m"])
    if length <= 0 or padding <= 0:
        raise ValueError("segmentation length and corridor half width must be positive")

    adaptive_zones = settings.get("adaptive_length_zones", [])
    if not isinstance(adaptive_zones, list):
        raise TypeError("segmentation.adaptive_length_zones must be an array")
    intervals = _segment_intervals(total, length, adaptive_zones)
    segments: list[dict[str, Any]] = []
    distances = np.array([item["distance_m"] for item in trajectory], dtype=np.float64)
    segment_ids: set[str] = set()
    for start, end in intervals:
        chosen = [item for item in trajectory if start <= item["distance_m"] <= end]
        if len(chosen) < 2:
            nearest = np.argsort(np.minimum(abs(distances - start), abs(distances - end)))[:2]
            chosen = [trajectory[int(position)] for position in sorted(nearest)]
        xs = [item["x"] for item in chosen]
        ys = [item["y"] for item in chosen]
        zs = [item["z"] for item in chosen]
        segment_id = f"s{round(start):04d}_{round(end):04d}m"
        if segment_id in segment_ids:
            raise ValueError(
                "Adaptive segmentation produced duplicate rounded segment id: "
                f"{segment_id}"
            )
        segment_ids.add(segment_id)
        segments.append(
            {
                "id": segment_id,
                "chainage_start_m": start,
                "chainage_end_m": end,
                "camera_count": len(chosen),
                "camera_index_from": chosen[0]["index"],
                "camera_index_to": chosen[-1]["index"],
                "bounds": {
                    "min_x": min(xs) - padding,
                    "max_x": max(xs) + padding,
                    "min_y": min(ys) - padding,
                    "max_y": max(ys) + padding,
                    "min_z": min(zs) - z_below,
                    "max_z": max(zs) + z_above,
                },
            }
        )

    if source_manifest is not None:
        assign_segment_sources(segments, source_manifest)

    result = {
        "schema_version": "railway.segments.v1",
        "project_id": project.project_id,
        "planning_method": "camera_trajectory_axis_aligned_envelope",
        "trajectory_length_m": total,
        "segment_length_m": length,
        "adaptive_length_zones": adaptive_zones,
        "corridor_half_width_m": padding,
        "segments": segments,
        "input_source_manifest": (
            str(input_source_manifest_path(project))
            if source_manifest is not None
            else None
        ),
        "warning": (
            "Bounds are conservative axis-aligned envelopes around the camera trajectory. "
            "Review every bound before large production cropping."
        ),
    }
    write_json(project.workspace_path("segment_manifest"), result)
    return result


def _inside(points: laspy.ScaleAwarePointRecord, bounds: dict[str, float]) -> np.ndarray:
    return (
        (points.x >= bounds["min_x"])
        & (points.x <= bounds["max_x"])
        & (points.y >= bounds["min_y"])
        & (points.y <= bounds["max_y"])
        & (points.z >= bounds["min_z"])
        & (points.z <= bounds["max_z"])
    )


def crop_segments(
    project: ProjectConfig,
    requested_ids: set[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    multi_source = "point_clouds" in project.value.get("inputs", {})
    source_manifest: dict[str, Any] | None = None
    if multi_source:
        source_manifest = load_input_source_manifest(project)
        source_lookup = {
            str(item["id"]): Path(str(item["path"]))
            for item in source_manifest.get("sources", [])
        }
        source_records = {
            str(item["id"]): item for item in source_manifest.get("sources", [])
        }
    else:
        source = project.input_path("point_cloud")
        if source is None or not source.is_file():
            raise FileNotFoundError(f"Point cloud not found: {source}")
        source_lookup = {"primary": source}
        source_records = {}
    manifest_path = project.workspace_path("segment_manifest")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Segment manifest not found: {manifest_path}; run plan-segments first"
        )
    from .io import load_json

    manifest = load_json(manifest_path)
    known = {item["id"] for item in manifest["segments"]}
    requested = requested_ids or set()
    unknown = requested - known
    if unknown:
        raise ValueError(f"Unknown segment ids: {sorted(unknown)}")
    segments = [
        item for item in manifest["segments"] if not requested or item["id"] in requested
    ]
    unknown_source_ids = {
        str(item.get("primary_source_id", "primary")) for item in segments
    } - set(source_lookup)
    if unknown_source_ids:
        raise ValueError(
            "Segment manifest references unknown point-cloud sources: "
            f"{sorted(unknown_source_ids)}"
        )
    output_dir = project.workspace_path("segments")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {item["id"]: output_dir / f"{item['id']}.laz" for item in segments}
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite segment outputs: " + ", ".join(str(path) for path in existing)
        )

    counts = {item["id"]: 0 for item in segments}
    temp_paths = {
        item["id"]: output_dir / f".{item['id']}.{uuid.uuid4().hex}.partial.laz"
        for item in segments
    }
    source_point_counts: dict[str, int] = {}
    try:
        for source_id, source_path in source_lookup.items():
            assigned = [
                item
                for item in segments
                if str(item.get("primary_source_id", "primary")) == source_id
            ]
            if not assigned:
                continue
            if not source_path.is_file():
                raise FileNotFoundError(f"Point-cloud source not found: {source_path}")
            with laspy.open(source_path) as reader, ExitStack() as stack:
                source_count = int(reader.header.point_count)
                source_point_counts[source_id] = source_count
                expected = source_records.get(source_id)
                if expected is not None:
                    if source_count != int(expected["point_count"]):
                        raise ValueError(
                            f"Point count changed after prepare-inputs: {source_id}"
                        )
                    if source_path.stat().st_size != int(expected["file_size_bytes"]):
                        raise ValueError(
                            f"File size changed after prepare-inputs: {source_id}"
                        )
                writers = {
                    item["id"]: stack.enter_context(
                        laspy.open(
                            temp_paths[item["id"]],
                            mode="w",
                            header=copy.deepcopy(reader.header),
                            do_compress=True,
                        )
                    )
                    for item in assigned
                }
                chunk_size = int(project.value["segmentation"]["chunk_size_points"])
                for points in reader.chunk_iterator(chunk_size):
                    for item in assigned:
                        selected = points[_inside(points, item["bounds"])]
                        if len(selected):
                            writers[item["id"]].write_points(selected)
                            counts[item["id"]] += len(selected)
        for item in segments:
            os.replace(temp_paths[item["id"]], outputs[item["id"]])
    finally:
        for temporary in temp_paths.values():
            temporary.unlink(missing_ok=True)

    result = {
        "schema_version": "railway.segment-crop-report.v1",
        "project_id": project.project_id,
        "source": (
            str(source_lookup["primary"])
            if not multi_source
            else None
        ),
        "source_point_count": sum(source_point_counts.values()),
        "source_point_counts": source_point_counts,
        "input_source_manifest": (
            str(input_source_manifest_path(project))
            if source_manifest is not None
            else None
        ),
        "segments": [
            {
                "id": item["id"],
                "primary_source_id": str(item.get("primary_source_id", "primary")),
                "context_source_ids": list(item.get("context_source_ids", ["primary"])),
                "bounds": item["bounds"],
                "output": str(outputs[item["id"]]),
                "point_count": counts[item["id"]],
                "file_size_bytes": outputs[item["id"]].stat().st_size,
            }
            for item in segments
        ],
    }
    write_json(project.workspace_path("reports") / "segment_crop.json", result)
    return result


def crop_segment_context_sources(
    project: ProjectConfig,
    segment_id: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Crop one identical segment envelope from every declared context source.

    These files are evidence-only and never become source owners.  Their purpose is
    to compare rail heads and topology across a multi-LAZ boundary before mesh export.
    """
    if "point_clouds" not in project.value.get("inputs", {}):
        raise ValueError("Context-source cropping requires inputs.point_clouds")
    source_manifest = load_input_source_manifest(project)
    source_lookup = {
        str(item["id"]): (Path(str(item["path"])), item)
        for item in source_manifest.get("sources", [])
    }
    manifest_path = project.workspace_path("segment_manifest")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Segment manifest not found: {manifest_path}; run plan-segments first"
        )
    from .io import load_json

    segment_manifest = load_json(manifest_path)
    segment = next(
        (item for item in segment_manifest["segments"] if item["id"] == segment_id),
        None,
    )
    if segment is None:
        raise ValueError(f"Unknown segment id: {segment_id}")
    context_ids = [str(value) for value in segment.get("context_source_ids", [])]
    if len(context_ids) < 2:
        raise ValueError(
            f"Segment {segment_id} does not have two or more context sources"
        )
    unknown = set(context_ids) - set(source_lookup)
    if unknown:
        raise ValueError(f"Unknown context source ids: {sorted(unknown)}")

    output_dir = project.workspace_path("segments") / "context_evidence" / segment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {source_id: output_dir / f"{source_id}.laz" for source_id in context_ids}
    report_path = (
        project.workspace_path("reports") / "seam_context" / f"{segment_id}.json"
    )
    existing = [path for path in [*outputs.values(), report_path] if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite context evidence: "
            + ", ".join(str(path) for path in existing)
        )

    counts: dict[str, int] = {}
    chunk_size = int(project.value["segmentation"]["chunk_size_points"])
    temporary_paths: dict[str, Path] = {}
    try:
        for source_id in context_ids:
            source_path, expected = source_lookup[source_id]
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            temporary = output_dir / f".{source_id}.{uuid.uuid4().hex}.partial.laz"
            temporary_paths[source_id] = temporary
            count = 0
            with laspy.open(source_path) as reader:
                if int(reader.header.point_count) != int(expected["point_count"]):
                    raise ValueError(f"Point count changed after prepare-inputs: {source_id}")
                if source_path.stat().st_size != int(expected["file_size_bytes"]):
                    raise ValueError(f"File size changed after prepare-inputs: {source_id}")
                with laspy.open(
                    temporary,
                    mode="w",
                    header=copy.deepcopy(reader.header),
                    do_compress=True,
                ) as writer:
                    for points in reader.chunk_iterator(chunk_size):
                        selected = points[_inside(points, segment["bounds"])]
                        if len(selected):
                            writer.write_points(selected)
                            count += len(selected)
            os.replace(temporary, outputs[source_id])
            counts[source_id] = count
    finally:
        for temporary in temporary_paths.values():
            temporary.unlink(missing_ok=True)

    result = {
        "schema_version": "railway.segment-context-evidence.v1",
        "project_id": project.project_id,
        "segment_id": segment_id,
        "chainage_range_m": [
            float(segment["chainage_start_m"]),
            float(segment["chainage_end_m"]),
        ],
        "bounds": segment["bounds"],
        "primary_source_id": segment["primary_source_id"],
        "context_source_ids": context_ids,
        "sources": [
            {
                "source_id": source_id,
                "role": (
                    "owner" if source_id == segment["primary_source_id"] else "context_only"
                ),
                "source": str(source_lookup[source_id][0]),
                "output": str(outputs[source_id]),
                "point_count": counts[source_id],
                "file_size_bytes": outputs[source_id].stat().st_size,
            }
            for source_id in context_ids
        ],
        "report_path": str(report_path),
        "status": "evidence_only_no_duplicate_mesh_authority",
        "limitations": [
            "Identical axis-aligned bounds do not imply duplicate surface coverage.",
            "Context crops may support seam QA but must never emit a second owner mesh.",
        ],
    }
    write_json(report_path, result)
    return result
