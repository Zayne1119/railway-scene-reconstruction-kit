from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

import laspy
import numpy as np

from .algorithms.mesh import ObjWriter
from .config import ProjectConfig
from .geometry import CorridorFrame
from .io import load_json, sha256_file, write_json
from .mesh_audit import audit_obj
from .targeted_canopy_recovery import (
    _column_box,
    _mesh_roof_run,
    _profile_runs,
    audit_roof_profile_mesh_continuity,
    extract_target_roofs,
    split_roof_surfaces,
)


def _resource_settings() -> dict[str, Any]:
    path = resources.files("railway_recon.resources").joinpath(
        "corridor-canopy-candidate.default.json"
    )
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _consecutive_runs(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    ordered = sorted(items, key=lambda item: int(item["grid_index"]))
    runs: list[list[dict[str, Any]]] = []
    for item in ordered:
        if not runs or int(item["grid_index"]) != int(runs[-1][-1]["grid_index"]) + 1:
            runs.append([item])
        else:
            runs[-1].append(item)
    return runs


def _write_materials(path: Path) -> None:
    content = """# Corridor canopy candidate materials
newmtl CanopyRoofObserved
Kd 0.72 0.79 0.80
Ks 0.10 0.10 0.10
Ns 16

newmtl CanopyColumnPointSupported
Kd 0.48 0.68 0.72
Ks 0.08 0.08 0.08
Ns 10
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _roof_interface(
    column: dict[str, Any],
    surfaces: list[dict[str, Any]],
    thickness_m: float,
    maximum_adjustment_m: float,
    maximum_cross_adjustment_m: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    adjusted = dict(column)
    longitudinal = float(column["predicted_longitudinal_position_m"])
    cross = float(column["cross_position_m"])
    options = []
    for surface in surfaces:
        for profile in surface["footprint_profiles"]:
            low_s, high_s = (float(value) for value in profile["longitudinal_range_m"])
            if not low_s - 1e-8 <= longitudinal <= high_s + 1e-8:
                continue
            low_c, high_c = (float(value) for value in profile["cross_range_m"])
            nearest_cross = min(max(cross, low_c), high_c)
            slope, intercept = (
                float(value) for value in profile["z_equals_a_cross_plus_d"]
            )
            roof_top = slope * nearest_cross + intercept
            options.append((abs(nearest_cross - cross), roof_top, surface["id"]))
    if not options:
        return adjusted, {
            "column_id": column["id"],
            "geometry_allowed": False,
            "status": "no_observed_roof_profile_at_column",
        }
    cross_distance, roof_top, surface_id = min(options)
    roof_underside = roof_top - thickness_m
    adjustment = roof_underside - float(column["maximum_z"])
    allowed = (
        cross_distance <= maximum_cross_adjustment_m
        and roof_underside > float(column["minimum_z"])
        and abs(adjustment) <= maximum_adjustment_m
    )
    if allowed:
        adjusted["maximum_z"] = roof_underside
    return adjusted, {
        "column_id": column["id"],
        "roof_surface_id": surface_id,
        "nearest_roof_cross_distance_m": cross_distance,
        "observed_column_top_z_m": float(column["maximum_z"]),
        "candidate_roof_underside_z_m": roof_underside,
        "column_top_adjustment_m": adjustment,
        "maximum_allowed_adjustment_m": maximum_adjustment_m,
        "maximum_allowed_cross_adjustment_m": maximum_cross_adjustment_m,
        "geometry_allowed": allowed,
        "status": "pass_adjusted_to_roof_underside" if allowed else "blocked_interface",
    }


def build_corridor_canopy_candidate(
    project: ProjectConfig,
    ownership_plan_value: str | Path,
    supported_grid_value: str | Path,
    output_name: str,
    *,
    settings_value: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build only a consecutive, point-supported corridor canopy candidate."""

    ownership_path = project.resolve(ownership_plan_value)
    grid_path = project.resolve(supported_grid_value)
    for path in (ownership_path, grid_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    ownership = load_json(ownership_path)
    grid_report = load_json(grid_path)
    settings_path = project.resolve(settings_value) if settings_value else None
    settings = load_json(settings_path) if settings_path else _resource_settings()
    supported = [
        item
        for item in grid_report.get("grid", [])
        if item.get("status") == "candidate_geometry_allowed_review_required"
    ]
    minimum_columns = int(settings["minimum_consecutive_columns"])
    runs = [run for run in _consecutive_runs(supported) if len(run) >= minimum_columns]
    if not runs:
        raise ValueError("No consecutive point-supported canopy column run")
    run = max(runs, key=len)
    rejected_supported = [item["id"] for item in supported if item not in run]
    output_report = project.workspace_path("reports") / f"{output_name}.json"
    if len(run) < minimum_columns:
        raise ValueError("Longest supported column run is below the configured minimum")

    frame_value = ownership["frame"]
    frame = CorridorFrame.from_json(
        {
            "origin_xy": frame_value["origin_xy"],
            "along_xy": frame_value["along_xy"],
            "cross_xy": frame_value["lateral_xy"],
        }
    )
    ownership_entries = {
        str(item["segment_id"]): item for item in ownership["segments"]
    }
    source_clouds = grid_report.get("source_clouds", {})
    owner_ids = sorted({str(item["owner_segment_id"]) for item in run})
    x_parts = []
    y_parts = []
    z_parts = []
    source_records = []
    for segment_id in owner_ids:
        cloud_path = Path(source_clouds[segment_id])
        cloud = laspy.read(cloud_path)
        x = np.asarray(cloud.x, dtype=np.float64)
        y = np.asarray(cloud.y, dtype=np.float64)
        z = np.asarray(cloud.z, dtype=np.float64)
        longitudinal, _ = frame.project(x, y)
        interval = ownership_entries[segment_id]["owned_interval_m"]
        mask = (longitudinal >= float(interval[0])) & (longitudinal <= float(interval[1]))
        x_parts.append(x[mask])
        y_parts.append(y[mask])
        z_parts.append(z[mask])
        source_records.append(
            {
                "segment_id": segment_id,
                "cloud": str(cloud_path),
                "owned_point_count": int(mask.sum()),
            }
        )
    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts)
    z = np.concatenate(z_parts)
    longitudinal, cross = frame.project(x, y)
    spacing = float(np.median(np.diff([item["predicted_longitudinal_position_m"] for item in run])))
    cross_center = float(np.median([item["cross_position_m"] for item in run]))
    half_width = float(settings["roof_search_half_width_m"])
    component = {
        "longitudinal_range_m": [
            float(run[0]["predicted_longitudinal_position_m"]) - spacing * 0.6,
            float(run[-1]["predicted_longitudinal_position_m"]) + spacing * 0.6,
        ],
        "cross_range_m": [cross_center - half_width, cross_center + half_width],
    }
    robust_grid = [dict(item) for item in run]
    top_values = np.asarray([float(item["maximum_z"]) for item in robust_grid])
    median_top = float(np.median(top_values))
    top_tolerance = float(settings["column_top_outlier_tolerance_m"])
    for item in robust_grid:
        if abs(float(item["maximum_z"]) - median_top) > top_tolerance:
            item["roof_search_maximum_z_before_robust_clip"] = item["maximum_z"]
            item["maximum_z"] = median_top
    roofs, roof_indexes, roof_search = extract_target_roofs(
        robust_grid,
        robust_grid,
        component,
        longitudinal,
        cross,
        z,
        settings,
    )
    raw_surfaces: list[dict[str, Any]] = []
    for roof, indexes in zip(roofs, roof_indexes, strict=True):
        raw_surfaces.extend(
            split_roof_surfaces(roof, indexes, longitudinal, cross, z, settings)
        )
    profile_residual_limit = float(
        settings["maximum_candidate_roof_profile_residual_p90_m"]
    )
    surfaces = []
    for surface in raw_surfaces:
        candidate = dict(surface)
        candidate["footprint_profiles"] = [
            profile
            for profile in surface["footprint_profiles"]
            if float(profile["absolute_residual_p90_m"]) <= profile_residual_limit
        ]
        candidate["rejected_high_residual_profile_count"] = (
            len(surface["footprint_profiles"]) - len(candidate["footprint_profiles"])
        )
        if _profile_runs(candidate):
            candidate["status"] = "local_profile_qa_passed_global_plane_not_required"
            surfaces.append(candidate)
    if not surfaces:
        report = {
            "schema_version": "railway.corridor-canopy-candidate.v1",
            "generated_at": datetime.now(UTC).isoformat(),
            "project_id": project.project_id,
            "ownership_plan": str(ownership_path),
            "supported_grid": str(grid_path),
            "selected_grid_ids": [item["id"] for item in run],
            "rejected_isolated_supported_grid_ids": rejected_supported,
            "source_clouds": source_records,
            "roof_search": roof_search,
            "roof_component_count": len(roofs),
            "raw_roof_surface_count": len(raw_surfaces),
            "raw_roof_surfaces": raw_surfaces,
            "roof_surface_count": 0,
            "passed": False,
            "status": "blocked_no_roof_surface_passed_candidate_qa",
            "formal_release": False,
        }
        write_json(output_report, report)
        return report
    profile_audit = audit_roof_profile_mesh_continuity(surfaces)
    thickness = float(settings["roof_candidate_thickness_m"])
    interface_limit = float(settings["column_roof_maximum_adjustment_m"])
    interface_cross_limit = float(
        settings.get("column_roof_maximum_cross_adjustment_m", 0.05)
    )
    adjusted_columns = []
    interfaces = []
    for column in run:
        adjusted, interface = _roof_interface(
            column, surfaces, thickness, interface_limit, interface_cross_limit
        )
        interfaces.append(interface)
        if interface["geometry_allowed"]:
            adjusted_columns.append(adjusted)
    if len(adjusted_columns) < minimum_columns:
        report = {
            "schema_version": "railway.corridor-canopy-candidate.v1",
            "generated_at": datetime.now(UTC).isoformat(),
            "project_id": project.project_id,
            "ownership_plan": str(ownership_path),
            "supported_grid": str(grid_path),
            "selected_grid_ids": [item["id"] for item in run],
            "rejected_isolated_supported_grid_ids": rejected_supported,
            "source_clouds": source_records,
            "roof_search": roof_search,
            "roof_component_count": len(roofs),
            "raw_roof_surface_count": len(raw_surfaces),
            "roof_surface_count": len(surfaces),
            "roof_surfaces": surfaces,
            "column_interface_count": len(interfaces),
            "geometry_column_count": len(adjusted_columns),
            "column_interfaces": interfaces,
            "passed": False,
            "status": "blocked_too_few_columns_pass_roof_interface",
            "formal_release": False,
        }
        write_json(output_report, report)
        return report

    output_dir = project.workspace_path("exports") / output_name
    output_obj = output_dir / f"{output_name}.obj"
    output_mtl = output_dir / f"{output_name}.mtl"
    output_origin = output_dir / "model_origin.json"
    output_audit = output_dir / "mesh_audit.json"
    outputs = (output_obj, output_mtl, output_origin, output_audit, output_report)
    if not overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite canopy candidate: {existing}")
    column_xy = frame.world_xy(
        np.asarray([item["predicted_longitudinal_position_m"] for item in adjusted_columns]),
        np.asarray([item["cross_position_m"] for item in adjusted_columns]),
    )
    origin = np.asarray(
        [float(np.mean(column_xy[:, 0])), float(np.mean(column_xy[:, 1])), 0.0]
    )
    writer = ObjWriter(origin, material_library=output_mtl.name)
    roof_run_count = 0
    for surface in surfaces:
        for run_number, profile_run in enumerate(_profile_runs(surface), start=1):
            vertices, faces = _mesh_roof_run(profile_run, frame, thickness)
            writer.add_mesh(
                f"{surface['id']}-RUN-{run_number:02d}",
                vertices,
                faces,
                "CanopyRoofObserved",
            )
            roof_run_count += 1
    for column in adjusted_columns:
        vertices, faces = _column_box(column, frame)
        writer.add_mesh(
            column["id"], vertices, faces, "CanopyColumnPointSupported"
        )
    writer.write(output_obj)
    _write_materials(output_mtl)
    write_json(output_origin, {"origin_xyz": origin.tolist(), "units": "metre", "axis": "Z-up"})
    audit = audit_obj(output_obj)
    audit["status"] = "pass" if audit["passed"] else "fail"
    write_json(output_audit, audit)
    if not audit["passed"]:
        raise ValueError("Corridor canopy candidate failed mesh audit")
    report = {
        "schema_version": "railway.corridor-canopy-candidate.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "project_id": project.project_id,
        "ownership_plan": str(ownership_path),
        "supported_grid": str(grid_path),
        "supported_grid_sha256": sha256_file(grid_path),
        "source_clouds": source_records,
        "selected_grid_ids": [item["id"] for item in run],
        "rejected_isolated_supported_grid_ids": rejected_supported,
        "roof_search": roof_search,
        "roof_component_count": len(roofs),
        "raw_roof_surface_count": len(raw_surfaces),
        "roof_surface_count": len(surfaces),
        "roof_run_count": roof_run_count,
        "roof_profile_mesh_audit": profile_audit,
        "column_interface_count": len(interfaces),
        "geometry_column_count": len(adjusted_columns),
        "column_interfaces": interfaces,
        "output_obj": str(output_obj),
        "output_mtl": str(output_mtl),
        "output_origin": str(output_origin),
        "output_mesh_audit": str(output_audit),
        "passed": True,
        "status": "point_supported_canopy_candidate_written_review_required",
        "formal_release": False,
        "limitations": [
            "Only the longest consecutive point-supported column run is emitted.",
            "Roof geometry follows observed top-envelope profiles and is split at unsafe transitions.",
            "Columns with incompatible observed roof interfaces are withheld from geometry.",
        ],
    }
    write_json(output_report, report)
    return report
