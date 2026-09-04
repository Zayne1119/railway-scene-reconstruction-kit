from __future__ import annotations

import copy
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .canopy_second_row_candidate import clone_obj_objects
from .io import load_json, sha256_file, write_json
from .mesh_audit import audit_obj
from .model_point_support import object_vertex_indices, parse_obj_model
from .registry import new_registry, summarize_registry, validate_registry_value


def infer_two_face_column_center(cross_positions_m: list[float]) -> dict[str, float]:
    values = np.sort(np.asarray(cross_positions_m, dtype=np.float64))
    if len(values) < 2:
        raise ValueError("At least two cross-face candidates are required")
    gaps = np.diff(values)
    split = int(np.argmax(gaps)) + 1
    separation = float(gaps[split - 1])
    if separation < 0.45:
        raise ValueError("Candidate cross positions do not resolve two column faces")
    low = float(np.median(values[:split]))
    high = float(np.median(values[split:]))
    return {
        "low_face_cross_m": low,
        "high_face_cross_m": high,
        "observed_cross_extent_m": high - low,
        "target_cross_center_m": (low + high) / 2.0,
        "largest_cluster_gap_m": separation,
    }


def _object_station_cross(
    model: Any,
    object_name: str,
    origin: np.ndarray,
    frame: dict[str, Any],
) -> tuple[float, float]:
    indexes = object_vertex_indices(model, object_name)
    if not len(indexes):
        raise ValueError(f"Column object is absent: {object_name}")
    vertices = model.vertices[indexes] + origin
    center_xy = (np.min(vertices, axis=0) + np.max(vertices, axis=0))[:2] / 2.0
    delta = center_xy - np.asarray(frame["origin_xy"], dtype=np.float64)
    return (
        float(delta @ np.asarray(frame["along_xy"], dtype=np.float64)),
        float(delta @ np.asarray(frame["cross_xy"], dtype=np.float64)),
    )


def derive_lateral_refit_plan(
    *,
    model: Any,
    origin: np.ndarray,
    registry: dict[str, Any],
    gap_report: dict[str, Any],
    frame: dict[str, Any],
) -> list[dict[str, Any]]:
    assets = {str(item["id"]): item for item in registry.get("assets", [])}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in gap_report["vertical_candidates"]:
        nearest = candidate.get("nearest_existing_vertical_asset", {})
        if not (
            candidate.get("priority") == "P0"
            and candidate.get("build_eligible_current_segment") is True
            and candidate.get("asset_relation") == "existing_asset_geometry_refinement"
            and nearest.get("asset_type") == "canopy_column"
        ):
            continue
        grouped.setdefault(str(nearest["asset_id"]), []).append(candidate)
    plans: list[dict[str, Any]] = []
    cross_axis = np.asarray(frame["cross_xy"], dtype=np.float64)
    for asset_id, candidates in sorted(grouped.items()):
        if asset_id not in assets or asset_id not in model.faces_by_object:
            continue
        try:
            face_fit = infer_two_face_column_center(
                [float(item["cross_m"]) for item in candidates]
            )
        except ValueError:
            continue
        current_station, current_cross = _object_station_cross(
            model, asset_id, origin, frame
        )
        shift = float(face_fit["target_cross_center_m"] - current_cross)
        if not 0.15 <= abs(shift) <= 0.60:
            continue
        observed_station = float(np.median([item["station_m"] for item in candidates]))
        if abs(observed_station - current_station) > 0.75:
            continue
        associated = sorted(
            name
            for name in model.faces_by_object
            if name == asset_id or name.startswith(f"{asset_id}-")
        )
        plans.append(
            {
                "asset_id": asset_id,
                "asset_type": "canopy_column",
                "candidate_ids": [str(item["candidate_id"]) for item in candidates],
                "candidate_count": len(candidates),
                "current_station_m": current_station,
                "observed_station_m": observed_station,
                "current_cross_m": current_cross,
                **face_fit,
                "cross_shift_m": shift,
                "translation_world_xyz_m": [
                    float(cross_axis[0] * shift),
                    float(cross_axis[1] * shift),
                    0.0,
                ],
                "associated_objects": associated,
            }
        )
    return plans


def build_canopy_column_lateral_refit(
    *,
    source_obj: str | Path,
    source_mtl: str | Path,
    source_origin: str | Path,
    source_registry: str | Path,
    gap_report_path: str | Path,
    frame_report_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Path]:
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty refit directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model = parse_obj_model(source_obj)
    origin_value = load_json(Path(source_origin))
    origin = np.asarray(origin_value["origin_xyz"], dtype=np.float64)
    registry = load_json(Path(source_registry))
    gap_report = load_json(Path(gap_report_path))
    frame = load_json(Path(frame_report_path))["frame"]
    plans = derive_lateral_refit_plan(
        model=model,
        origin=origin,
        registry=registry,
        gap_report=gap_report,
        frame=frame,
    )
    if len(plans) != 13:
        raise ValueError(f"Expected 13 two-face canopy-column refits; got {len(plans)}")
    clone_specs = [
        {
            "source_object": object_name,
            "clone_object": object_name,
            "translation_local_xyz_m": plan["translation_world_xyz_m"],
        }
        for plan in plans
        for object_name in plan["associated_objects"]
    ]
    omitted = {str(item["source_object"]) for item in clone_specs}
    output_name = output.name
    output_obj = output / f"{output_name}.obj"
    output_mtl = output / f"{output_name}.mtl"
    output_origin = output / "model_origin.json"
    clone_obj_objects(
        source_obj,
        output_obj,
        clone_specs,
        output_mtl_name=output_mtl.name,
        omit_objects=omitted,
    )
    shutil.copy2(source_mtl, output_mtl)
    shutil.copy2(source_origin, output_origin)

    updated_registry = copy.deepcopy(registry)
    plan_by_object = {
        object_name: plan
        for plan in plans
        for object_name in plan["associated_objects"]
    }
    changed_assets: list[dict[str, Any]] = []
    for asset in updated_registry.get("assets", []):
        if isinstance(asset.get("geometry"), dict):
            asset["geometry"]["file"] = str(output_obj)
        asset_id = str(asset.get("id"))
        if asset_id not in plan_by_object:
            continue
        plan = plan_by_object[asset_id]
        asset["status"] = "candidate"
        asset["evidence_level"] = "observed"
        asset["confidence"] = 0.96 if asset.get("type") == "canopy_column" else 0.90
        asset.setdefault("parameters", {}).update(
            {
                "lateral_refit_cross_shift_m": plan["cross_shift_m"],
                "lateral_refit_target_cross_m": plan["target_cross_center_m"],
                "lateral_refit_candidate_ids": plan["candidate_ids"],
                "geometry_method": "paired_observed_faces_lateral_refit",
            }
        )
        asset.setdefault("sources", []).append(
            {
                "kind": "point_cloud",
                "reference": str(Path(gap_report_path).resolve()),
                "note": "Two independently detected shaft faces determine the lateral center.",
            }
        )
        asset["limitations"] = [
            "Lateral position is refitted from new point-cloud faces; hidden section detail remains inherited."
        ]
        changed_assets.append(copy.deepcopy(asset))
    isolated = new_registry(str(registry.get("project_id", "site-b")))
    isolated["assets"] = changed_assets
    isolated["summary"] = summarize_registry(isolated)
    changed_errors = validate_registry_value(isolated)
    if changed_errors:
        raise ValueError("Refitted canopy assets are invalid: " + "; ".join(changed_errors))
    inherited_errors = validate_registry_value(registry)
    updated_registry["release_id"] = output_name
    updated_registry["updated_at"] = datetime.now(UTC).isoformat()
    updated_registry["summary"] = summarize_registry(updated_registry)
    output_registry = output / "asset_registry.json"
    write_json(output_registry, updated_registry)
    mesh_audit = audit_obj(output_obj)
    output_mesh_audit = output / "mesh_audit.json"
    write_json(output_mesh_audit, mesh_audit)
    if not mesh_audit["passed"]:
        raise ValueError("Canopy-column lateral refit failed mesh audit")
    output_report = output / "canopy_column_lateral_refit_report.json"
    write_json(
        output_report,
        {
            "schema_version": "railway.canopy-column-lateral-refit.v1",
            "source_obj": str(Path(source_obj).resolve()),
            "output_obj": str(output_obj),
            "refit_column_count": len(plans),
            "refit_component_count": len(clone_specs),
            "plans": plans,
            "new_or_changed_assets_schema_valid": True,
            "inherited_registry_validation_error_count": len(inherited_errors),
            "mesh_audit_passed": True,
            "output_obj_sha256": sha256_file(output_obj),
            "status": "candidate_requires_reverse_gap_delta_and_fixed_view_review",
        },
    )
    return {
        "obj": output_obj,
        "mtl": output_mtl,
        "origin": output_origin,
        "registry": output_registry,
        "mesh_audit": output_mesh_audit,
        "report": output_report,
    }
