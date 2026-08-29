from __future__ import annotations

import copy
import os
import uuid
from contextlib import ExitStack
from typing import Any

import laspy
import numpy as np

from .camera import camera_trajectory, load_camera_rows
from .config import ProjectConfig
from .io import write_json


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
        cursor = start_m
        while cursor < end_m - 1e-9:
            boundary = min(end_m, cursor + length_m)
            intervals.append((cursor, boundary))
            cursor = boundary

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

    result = {
        "schema_version": "railway.segments.v1",
        "project_id": project.project_id,
        "planning_method": "camera_trajectory_axis_aligned_envelope",
        "trajectory_length_m": total,
        "segment_length_m": length,
        "adaptive_length_zones": adaptive_zones,
        "corridor_half_width_m": padding,
        "segments": segments,
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
    source = project.input_path("point_cloud")
    if source is None or not source.is_file():
        raise FileNotFoundError(f"Point cloud not found: {source}")
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
    source_count = 0
    try:
        with laspy.open(source) as reader, ExitStack() as stack:
            source_count = int(reader.header.point_count)
            writers = {
                item["id"]: stack.enter_context(
                    laspy.open(
                        temp_paths[item["id"]],
                        mode="w",
                        header=copy.deepcopy(reader.header),
                        do_compress=True,
                    )
                )
                for item in segments
            }
            chunk_size = int(project.value["segmentation"]["chunk_size_points"])
            for points in reader.chunk_iterator(chunk_size):
                for item in segments:
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
        "source": str(source),
        "source_point_count": source_count,
        "segments": [
            {
                "id": item["id"],
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
