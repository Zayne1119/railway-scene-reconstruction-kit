from __future__ import annotations

import json
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

import laspy
import numpy as np

from .config import ProjectConfig
from .io import load_json, sha256_file, write_json


def _resource_settings() -> dict[str, Any]:
    path = resources.files("railway_recon.resources").joinpath(
        "platform-gap-point-evidence.default.json"
    )
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _surface_elevation(component: dict[str, Any], longitudinal: float, cross: float) -> float:
    fits = list(component.get("fit_segments", []))
    if not fits:
        raise ValueError("Platform component has no fitted surface segments")
    selected = [
        fit
        for fit in fits
        if float(fit["longitudinal_range_m"][0]) <= longitudinal
        <= float(fit["longitudinal_range_m"][1])
    ]
    if not selected:
        selected = [
            min(
                fits,
                key=lambda fit: abs(
                    longitudinal
                    - sum(float(value) for value in fit["longitudinal_range_m"]) / 2.0
                ),
            )
        ]
    elevations = []
    for fit in selected:
        a, b, d = (float(value) for value in fit["plane_z_equals_a_s_plus_b_c_plus_d"])
        elevations.append(a * longitudinal + b * cross + d)
    return float(np.median(elevations))


def analyze_platform_gap_point_evidence_data(
    platform: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    settings: dict[str, Any],
) -> dict[str, Any]:
    frame = platform["frame"]
    origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    delta = np.column_stack((x, y)) - origin
    longitudinal = delta @ np.asarray(frame["along_xy"], dtype=np.float64)
    cross = delta @ np.asarray(frame["cross_xy"], dtype=np.float64)
    minimum_density = float(settings["minimum_directional_point_density_per_m2"])
    minimum_span = float(settings["minimum_vertical_span_m"])
    dominance = float(settings["directional_density_dominance_ratio"])
    lower_offset = float(settings["surface_exclusion_offset_m"])
    maximum_above = float(settings["maximum_above_surface_m"])
    maximum_below = float(settings["maximum_below_surface_m"])
    records: list[dict[str, Any]] = []
    for component in platform.get("platform_components", []):
        for gap in component.get("interior_gap_candidates", []):
            if gap.get("gap_classification") != "unique_enclosed_gap_review_required":
                continue
            s_min, s_max = (float(value) for value in gap["longitudinal_range_m"])
            c_min, c_max = (float(value) for value in gap["cross_range_m"])
            center_s = (s_min + s_max) / 2.0
            center_c = (c_min + c_max) / 2.0
            surface_z = _surface_elevation(component, center_s, center_c)
            footprint = (
                (longitudinal >= s_min)
                & (longitudinal <= s_max)
                & (cross >= c_min)
                & (cross <= c_max)
            )
            local_z = z[footprint]
            area = max((s_max - s_min) * (c_max - c_min), 1e-9)
            above = local_z[
                (local_z >= surface_z + lower_offset)
                & (local_z <= surface_z + maximum_above)
            ]
            below = local_z[
                (local_z <= surface_z - lower_offset)
                & (local_z >= surface_z - maximum_below)
            ]
            above_density = float(len(above) / area)
            below_density = float(len(below) / area)
            above_span = float(np.ptp(above)) if len(above) > 1 else 0.0
            below_span = float(np.ptp(below)) if len(below) > 1 else 0.0
            above_supported = above_density >= minimum_density and above_span >= minimum_span
            below_supported = below_density >= minimum_density and below_span >= minimum_span
            if above_supported and above_density >= dominance * max(below_density, 1e-9):
                classification = "vertical_occluder_point_support_not_platform_opening"
                mesh_action = "preserve_platform_surface_do_not_create_opening"
            elif below_supported and below_density >= dominance * max(above_density, 1e-9):
                classification = "below_surface_void_or_stair_point_support_review_required"
                mesh_action = "keep_opening_unresolved_until_geometry_reconstruction"
            else:
                classification = "mixed_or_insufficient_point_support_review_required"
                mesh_action = "keep_opening_unresolved"
            records.append(
                {
                    "gap_id": gap["id"],
                    "component_id": component["id"],
                    "surface_elevation_m": surface_z,
                    "footprint_area_m2": area,
                    "footprint_point_count": len(local_z),
                    "above_surface_point_count": len(above),
                    "below_surface_point_count": len(below),
                    "above_surface_density_per_m2": above_density,
                    "below_surface_density_per_m2": below_density,
                    "above_surface_vertical_span_m": above_span,
                    "below_surface_vertical_span_m": below_span,
                    "classification": classification,
                    "mesh_action": mesh_action,
                    "status": "automatic_point_evidence_not_final_asset_confirmation",
                }
            )
    supported = sum(
        item["classification"] == "vertical_occluder_point_support_not_platform_opening"
        for item in records
    )
    return {
        "schema_version": "railway.platform-gap-point-evidence.v1",
        "project_id": platform.get("project_id"),
        "segment_id": platform["segment_id"],
        "gap_count": len(records),
        "vertical_occluder_supported_count": supported,
        "unresolved_count": len(records) - supported,
        "gaps": records,
        "settings": settings,
        "status": "point_evidence_generated_candidate_only",
        "limitations": [
            "Below-surface returns may represent stairs, voids, walls, or multipath and remain unresolved.",
            "Only dominant above-surface vertical support can suppress an opening candidate automatically.",
            "This evidence stage never edits geometry or the asset registry.",
        ],
    }


def analyze_platform_gap_point_evidence(
    project: ProjectConfig,
    segment_id: str,
    platform_report_value: str | Path,
    output_value: str | Path,
    *,
    settings_value: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    platform_path = project.resolve(platform_report_value)
    output_path = project.resolve(output_value)
    if not platform_path.is_file():
        raise FileNotFoundError(platform_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    platform = load_json(platform_path)
    if platform.get("schema_version") != "railway.platform-surface.v1":
        raise ValueError("Point evidence requires a platform-surface report")
    if str(platform.get("segment_id")) != segment_id:
        raise ValueError("Platform report belongs to a different segment")
    source = Path(platform["source"])
    if not source.is_file():
        raise FileNotFoundError(source)
    settings_path = project.resolve(settings_value) if settings_value else None
    settings = load_json(settings_path) if settings_path else _resource_settings()
    cloud = laspy.read(source)
    report = analyze_platform_gap_point_evidence_data(
        platform,
        np.asarray(cloud.x, dtype=np.float64),
        np.asarray(cloud.y, dtype=np.float64),
        np.asarray(cloud.z, dtype=np.float64),
        settings,
    )
    report.update(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "platform_report": str(platform_path),
            "platform_report_sha256": sha256_file(platform_path),
            "source": str(source),
            "settings_source": str(settings_path) if settings_path else "bundled_default",
            "output_report": str(output_path),
        }
    )
    write_json(output_path, report)
    return report
