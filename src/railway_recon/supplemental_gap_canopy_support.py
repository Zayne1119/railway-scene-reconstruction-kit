from __future__ import annotations

import copy
import shutil
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .canopy_second_row_candidate import clone_obj_objects
from .io import load_json, sha256_file, write_json
from .mesh_audit import audit_obj
from .model_point_support import object_vertex_indices, parse_obj_model
from .registry import new_registry, summarize_registry, validate_registry_value
from .supplemental_endpoint_columns import _sample_object_surface
from .supplemental_gap_masts import _load_candidate_points


def fit_occluded_canopy_support(
    points_station_cross_z: np.ndarray,
    *,
    minimum_z_m: float,
    maximum_z_m: float,
) -> dict[str, Any]:
    points = np.asarray(points_station_cross_z, dtype=np.float64)
    if len(points) < 100:
        raise ValueError("Insufficient base and roof-band points for canopy support")
    base = points[points[:, 2] <= minimum_z_m + 0.40]
    upper = points[points[:, 2] >= maximum_z_m - 1.20]
    if len(base) < 15 or len(upper) < 60:
        raise ValueError("Canopy support lacks a stable base or upper connection band")
    base_center = np.median(base[:, :2], axis=0)
    upper_center = np.median(upper[:, :2], axis=0)
    centre = (base_center + upper_center) / 2.0
    return {
        "station_m": float(centre[0]),
        "cross_m": float(centre[1]),
        "base_z_m": float(minimum_z_m),
        "maximum_observed_z_m": float(maximum_z_m),
        "base_point_count": len(base),
        "upper_connection_point_count": len(upper),
        "source_point_count": len(points),
        "base_to_upper_station_offset_m": float(upper_center[0] - base_center[0]),
        "base_to_upper_cross_offset_m": float(upper_center[1] - base_center[1]),
        "continuous_shaft_observed": False,
        "evidence_mode": "point_supported_end_bands_plus_multi_view_photo_shaft",
    }


def _object_center_and_base(
    model: Any,
    object_name: str,
    origin: np.ndarray,
) -> tuple[np.ndarray, float]:
    indexes = object_vertex_indices(model, object_name)
    if not len(indexes):
        raise ValueError(f"Template object is absent: {object_name}")
    vertices = model.vertices[indexes] + origin
    return (np.min(vertices, axis=0) + np.max(vertices, axis=0))[:2] / 2.0, float(
        np.min(vertices[:, 2])
    )


def _select_same_station_template(
    model: Any,
    registry: dict[str, Any],
    origin: np.ndarray,
    frame: dict[str, Any],
    station_m: float,
) -> dict[str, Any]:
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    candidates: list[dict[str, Any]] = []
    for asset in registry.get("assets", []):
        if asset.get("type") != "canopy_column":
            continue
        name = str(asset.get("geometry", {}).get("node", asset.get("id", "")))
        if "OPPOSITE-CANOPY-COLUMN-" not in name or name not in model.faces_by_object:
            continue
        center_xy, base_z = _object_center_and_base(model, name, origin)
        delta = center_xy - frame_origin
        candidates.append(
            {
                "asset_id": name,
                "station_m": float(delta @ along),
                "cross_m": float(delta @ cross),
                "base_z_m": base_z,
            }
        )
    if not candidates:
        raise ValueError("No accepted opposite-platform canopy-column template is available")
    selected = min(candidates, key=lambda item: abs(item["station_m"] - station_m))
    selected["station_delta_m"] = float(station_m - selected["station_m"])
    if abs(selected["station_delta_m"]) > 0.75:
        raise ValueError("No same-station canopy-column template within 0.75 m")
    return selected


def _band_to_assembly_support(
    model: Any,
    origin: np.ndarray,
    object_names: list[str],
    points: np.ndarray,
    frame: dict[str, Any],
    minimum_z_m: float,
    maximum_z_m: float,
) -> dict[str, Any]:
    selected = points[
        (points[:, 2] <= minimum_z_m + 0.40)
        | (points[:, 2] >= maximum_z_m - 1.20)
    ]
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    xy = (
        frame_origin[None, :]
        + selected[:, 0, None] * along[None, :]
        + selected[:, 1, None] * cross[None, :]
    )
    world = np.column_stack((xy, selected[:, 2]))
    surfaces = [
        _sample_object_surface(model, name, origin, spacing_m=0.035)
        for name in object_names
    ]
    distances, _ = cKDTree(np.vstack(surfaces)).query(world, workers=-1)
    gates = {
        "p95_within_0_35m_half_template_section": (
            float(np.percentile(distances, 95)) <= 0.35
        ),
        "coverage_at_0_30m_at_least_0_90": float(np.mean(distances <= 0.30)) >= 0.90,
    }
    return {
        "direction": "observed_base_and_upper_connection_bands_to_cloned_assembly_surface",
        "point_count": len(selected),
        "p50_m": float(np.percentile(distances, 50)),
        "p90_m": float(np.percentile(distances, 90)),
        "p95_m": float(np.percentile(distances, 95)),
        "coverage_at_0_20m": float(np.mean(distances <= 0.20)),
        "coverage_at_0_30m": float(np.mean(distances <= 0.30)),
        "gates": gates,
        "passed": all(gates.values()),
        "interpretation": (
            "The candidate bands sample the support centre/joint region; the accepted "
            "same-station template supplies the full external section."
        ),
    }


def build_supplemental_gap_canopy_support(
    *,
    cloud_path: str | Path,
    gap_report_path: str | Path,
    disposition_path: str | Path,
    photo_evidence_path: str | Path,
    frame_report_path: str | Path,
    source_obj: str | Path,
    source_mtl: str | Path,
    source_origin: str | Path,
    source_registry: str | Path,
    output_directory: str | Path,
) -> dict[str, Path]:
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty candidate directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cloud = Path(cloud_path).resolve()
    gap_path = Path(gap_report_path).resolve()
    disposition_file = Path(disposition_path).resolve()
    source_model = Path(source_obj).resolve()
    source_material = Path(source_mtl).resolve()
    source_origin_file = Path(source_origin).resolve()
    source_registry_file = Path(source_registry).resolve()
    gap = load_json(gap_path)
    disposition = load_json(disposition_file)
    photo = load_json(Path(photo_evidence_path))
    frame = load_json(Path(frame_report_path))["frame"]
    decision = next(
        (
            item
            for item in disposition.get("decisions", [])
            if item.get("candidate_id") == "GAP-VERTICAL-0018"
            and item.get("reviewed_class") == "possible_opposite_canopy_support"
            and item.get("model_action") == "require_geometry_gate"
        ),
        None,
    )
    if decision is None:
        raise ValueError("GAP-VERTICAL-0018 lacks an explicit geometry-gate decision")
    photo_record = next(
        (
            item
            for item in photo.get("candidates", [])
            if item.get("candidate_id") == "GAP-VERTICAL-0018"
        ),
        None,
    )
    if photo_record is None or int(photo_record["trusted_reviewable_view_count"]) < 2:
        raise ValueError("GAP-VERTICAL-0018 lacks multi-view photo evidence")
    candidate = next(
        item
        for item in gap["vertical_candidates"]
        if item["candidate_id"] == "GAP-VERTICAL-0018"
    )
    points = _load_candidate_points(cloud, [candidate], frame)["GAP-VERTICAL-0018"]
    fit = fit_occluded_canopy_support(
        points,
        minimum_z_m=float(candidate["minimum_xyz_m"][2]),
        maximum_z_m=float(candidate["maximum_xyz_m"][2]),
    )

    model = parse_obj_model(source_model)
    origin_value = load_json(source_origin_file)
    origin = np.asarray(origin_value["origin_xyz"], dtype=np.float64)
    registry = load_json(source_registry_file)
    template = _select_same_station_template(model, registry, origin, frame, fit["station_m"])
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    target_xy = frame_origin + fit["station_m"] * along + fit["cross_m"] * cross
    source_xy, source_base_z = _object_center_and_base(
        model, template["asset_id"], origin
    )
    translation = np.asarray(
        [
            target_xy[0] - source_xy[0],
            target_xy[1] - source_xy[1],
            fit["base_z_m"] - source_base_z,
        ],
        dtype=np.float64,
    )
    associated = sorted(
        name
        for name in model.faces_by_object
        if name == template["asset_id"] or name.startswith(f"{template['asset_id']}-")
    )
    target_root = "SUPPLEMENTAL-OPPOSITE-CANOPY-OUTER-COLUMN-001"
    specs = [
        {
            "source_object": source_name,
            "clone_object": f"{target_root}{source_name[len(template['asset_id']):]}",
            "translation_local_xyz_m": translation.tolist(),
        }
        for source_name in associated
    ]
    output_name = output.name
    output_obj = output / f"{output_name}.obj"
    output_mtl = output / f"{output_name}.mtl"
    output_origin = output / "model_origin.json"
    clone_obj_objects(
        source_model,
        output_obj,
        specs,
        output_mtl_name=output_mtl.name,
    )
    shutil.copy2(source_material, output_mtl)
    shutil.copy2(source_origin_file, output_origin)

    output_model = parse_obj_model(output_obj)
    clone_names = [str(item["clone_object"]) for item in specs]
    support = _band_to_assembly_support(
        output_model,
        origin,
        clone_names,
        points,
        frame,
        float(candidate["minimum_xyz_m"][2]),
        float(candidate["maximum_xyz_m"][2]),
    )
    if not support["passed"]:
        raise ValueError("Supplemental canopy support failed base/roof-band support")

    assets_by_id = {str(item["id"]): item for item in registry.get("assets", [])}
    updated = copy.deepcopy(registry)
    for asset in updated.get("assets", []):
        if isinstance(asset.get("geometry"), dict):
            asset["geometry"]["file"] = str(output_obj)
    added_assets: list[dict[str, Any]] = []
    for spec in specs:
        source_asset = copy.deepcopy(assets_by_id[str(spec["source_object"])])
        source_asset["id"] = str(spec["clone_object"])
        source_asset["status"] = "candidate"
        source_asset["evidence_level"] = "photo_interpreted"
        source_asset["confidence"] = float(decision["confidence"])
        source_asset["chainage_m"] = fit["station_m"]
        source_asset["geometry"] = {
            "file": str(output_obj),
            "node": str(spec["clone_object"]),
        }
        source_asset["sources"] = [
            {"kind": "point_cloud", "reference": str(cloud)},
            {"kind": "panorama", "reference": str(photo_record["evidence_sheet"])},
            {"kind": "manual_review", "reference": str(disposition_file)},
            {
                "kind": "rule",
                "reference": str(template["asset_id"]),
                "note": "Same-station accepted inner support supplies hidden section and connection geometry.",
            },
        ]
        source_asset.setdefault("parameters", {}).update(
            {
                "source_candidate_id": "GAP-VERTICAL-0018",
                "source_template_asset_id": str(spec["source_object"]),
                "target_station_m": fit["station_m"],
                "target_cross_m": fit["cross_m"],
                "lateral_pair_offset_m": fit["cross_m"] - template["cross_m"],
                "geometry_method": "same_station_column_assembly_photo_interpreted_lateral_pair",
            }
        )
        source_asset["limitations"] = [
            "Base and roof connection bands are observed; the occluded shaft reuses a same-station accepted template.",
            "Candidate remains separate from the formal baseline until final promotion.",
        ]
        added_assets.append(source_asset)
        updated["assets"].append(source_asset)
    added_relations: list[dict[str, Any]] = []
    for previous, following in pairwise(clone_names):
        relation = {
            "id": f"REL-{previous}-SUPPORTS-{following}",
            "type": "supports",
            "from": previous,
            "to": following,
        }
        added_relations.append(relation)
        updated.setdefault("relations", []).append(relation)
    updated["release_id"] = output_name
    updated["updated_at"] = datetime.now(UTC).isoformat()
    updated["summary"] = summarize_registry(updated)
    addition = new_registry(str(updated.get("project_id", "site-b")))
    addition["assets"] = copy.deepcopy(added_assets)
    addition["relations"] = copy.deepcopy(added_relations)
    addition["summary"] = summarize_registry(addition)
    addition_errors = validate_registry_value(addition)
    if addition_errors:
        raise ValueError("Supplemental canopy-support additions are invalid: " + "; ".join(addition_errors))
    inherited_errors = validate_registry_value(registry)
    output_registry = output / "asset_registry.json"
    write_json(output_registry, updated)
    mesh_audit = audit_obj(output_obj)
    output_mesh_audit = output / "mesh_audit.json"
    write_json(output_mesh_audit, mesh_audit)
    if not mesh_audit["passed"]:
        raise ValueError("Supplemental canopy-support candidate failed mesh audit")
    output_report = output / "supplemental_gap_canopy_support_report.json"
    write_json(
        output_report,
        {
            "schema_version": "railway.supplemental-gap-canopy-support.v1",
            "source_obj": str(source_model),
            "output_obj": str(output_obj),
            "candidate_id": "GAP-VERTICAL-0018",
            "asset_root": target_root,
            "template": template,
            "fit": fit,
            "translation_xyz_m": translation.tolist(),
            "clone_objects": clone_names,
            "trusted_reviewable_photo_views": int(
                photo_record["trusted_reviewable_view_count"]
            ),
            "point_band_to_assembly_support": support,
            "added_asset_count": len(added_assets),
            "added_relation_count": len(added_relations),
            "new_assets_schema_valid": True,
            "inherited_registry_validation_error_count": len(inherited_errors),
            "mesh_audit_passed": True,
            "output_obj_sha256": sha256_file(output_obj),
            "status": "photo_interpreted_candidate_requires_fixed_view_and_interface_gate",
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
