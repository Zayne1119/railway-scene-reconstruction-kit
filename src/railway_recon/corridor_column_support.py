from __future__ import annotations

import json
import math
from importlib import resources
from pathlib import Path
from typing import Any

import laspy
import numpy as np

from .config import ProjectConfig
from .io import load_json, sha256_file, write_json


def _resource_settings() -> dict[str, Any]:
    path = resources.files("railway_recon.resources").joinpath(
        "corridor-column-support.default.json"
    )
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _support_metrics(
    s: np.ndarray,
    c: np.ndarray,
    z: np.ndarray,
    item: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    station = float(item["longitudinal_position_m"])
    cross = float(item["cross_position_m"])
    bottom = float(item["minimum_z"])
    top = float(item["maximum_z"])
    radius = float(settings["support_radius_m"])
    bin_size = float(settings["vertical_bin_m"])
    mask = (
        ((s - station) ** 2 + (c - cross) ** 2 <= radius**2)
        & (z >= bottom + float(settings["bottom_clearance_m"]))
        & (z <= top + float(settings["top_tolerance_m"]))
    )
    values = z[mask]
    bin_count = max(1, math.ceil(max(top - bottom, bin_size) / bin_size))
    if values.size:
        indexes = np.clip(
            ((values - bottom) / bin_size).astype(np.int64), 0, bin_count - 1
        )
        occupied = int(np.unique(indexes).size)
    else:
        occupied = 0
    ratio = occupied / bin_count
    supported = (
        values.size >= int(settings["minimum_points"])
        and ratio >= float(settings["minimum_occupied_ratio"])
    )
    return {
        "radius_m": radius,
        "point_count": int(values.size),
        "vertical_bin_count": bin_count,
        "occupied_vertical_bin_count": occupied,
        "occupied_vertical_ratio": ratio,
        "supported": bool(supported),
    }


def validate_corridor_column_grid(
    project: ProjectConfig,
    grid_value: str | Path,
    output_value: str | Path,
    *,
    settings_value: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate every inferred column position against its owner point cloud."""

    grid_path = project.resolve(grid_value)
    output_path = project.resolve(output_value)
    settings_path = project.resolve(settings_value) if settings_value else None
    if not grid_path.is_file():
        raise FileNotFoundError(grid_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    grid_report = load_json(grid_path)
    if grid_report.get("schema_version") != "railway.corridor-column-grid-recovery.v1":
        raise ValueError("Column support validation requires a recovered corridor grid")
    settings = load_json(settings_path) if settings_path else _resource_settings()
    if settings.get("schema_version") != "railway.corridor-column-support-settings.v1":
        raise ValueError("Unsupported corridor-column support settings")

    ownership_path = Path(str(grid_report["ownership_plan"])).resolve()
    ownership = load_json(ownership_path)
    frame = ownership["frame"]
    origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    lateral = np.asarray(frame["lateral_xy"], dtype=np.float64)
    items = [dict(item) for item in grid_report.get("grid", [])]
    by_owner: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_owner.setdefault(str(item["owner_segment_id"]), []).append(item)

    vertical_reports = {
        str(segment_id): Path(str(path)).resolve()
        for segment_id, path in grid_report.get("vertical_reports", {}).items()
    }
    source_clouds: dict[str, str] = {}
    for segment_id, owned_items in by_owner.items():
        report_path = vertical_reports.get(segment_id)
        if report_path is None or not report_path.is_file():
            raise FileNotFoundError(f"Vertical report missing for {segment_id}")
        vertical = load_json(report_path)
        cloud_path = Path(str(vertical["source"])).resolve()
        if not cloud_path.is_file():
            raise FileNotFoundError(cloud_path)
        source_clouds[segment_id] = str(cloud_path)
        cloud = laspy.read(cloud_path)
        x = np.asarray(cloud.x, dtype=np.float64)
        y = np.asarray(cloud.y, dtype=np.float64)
        z = np.asarray(cloud.z, dtype=np.float64)
        delta = np.column_stack((x, y)) - origin
        s = delta @ along
        c = delta @ lateral
        for item in owned_items:
            support = _support_metrics(s, c, z, item, settings)
            item["point_support"] = support
            if support["supported"]:
                item["evidence_level"] = "point_supported_grid_inference"
                item["status"] = "candidate_geometry_allowed_review_required"
            else:
                item["evidence_level"] = "grid_inferred_pending_support"
                item["status"] = "grid_position_only_no_geometry"

    supported_count = sum(item["point_support"]["supported"] for item in items)
    result = dict(grid_report)
    result.update(
        {
            "schema_version": "railway.corridor-column-grid-supported.v1",
            "source_grid": str(grid_path),
            "source_grid_sha256": sha256_file(grid_path),
            "support_settings_source": (
                str(settings_path) if settings_path else "bundled_default"
            ),
            "source_clouds": source_clouds,
            "grid": items,
            "supported_position_count": supported_count,
            "unsupported_position_count": len(items) - supported_count,
            "geometry_allowed_position_ids": [
                str(item["id"])
                for item in items
                if item["point_support"]["supported"]
            ],
            "passed": True,
            "status": "local_point_support_evaluated_geometry_remains_candidate",
            "limitations": [
                "Point-supported positions remain candidate geometry until scene-level QA passes.",
                "Unsupported grid positions must not create renderable columns.",
            ],
        }
    )
    write_json(output_path, result)
    return result
