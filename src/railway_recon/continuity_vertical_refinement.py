from __future__ import annotations

import copy
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .canopy_second_row_candidate import clone_obj_objects
from .io import load_json, write_json
from .mesh_audit import audit_obj
from .model_point_support import object_vertex_indices, parse_obj_model
from .registry import summarize_registry


def _object_center_global_xy(model: Any, name: str, origin: np.ndarray) -> np.ndarray:
    vertices = model.vertices[object_vertex_indices(model, name)] + origin
    return ((np.min(vertices, axis=0) + np.max(vertices, axis=0)) / 2.0)[:2]


def _station_cross(point_xy: np.ndarray, frame: dict[str, Any]) -> tuple[float, float]:
    delta = point_xy - np.asarray(frame["origin_xy"], dtype=np.float64)
    return (
        float(delta @ np.asarray(frame["along_xy"], dtype=np.float64)),
        float(delta @ np.asarray(frame["cross_xy"], dtype=np.float64)),
    )


def build_clone_plan(
    model: Any,
    origin: np.ndarray,
    registry: dict[str, Any],
    comparison: dict[str, Any],
    decisions: dict[str, Any],
    frame: dict[str, Any],
) -> tuple[list[dict[str, Any]], set[str], list[dict[str, Any]]]:
    records = {str(item["id"]): item for item in comparison["records"]}
    assets = {str(item["id"]): item for item in registry.get("assets", [])}
    mast_template = "SEG2050-CATENARY--CATENARY-MAST-0001"
    mast_foundation_template = f"{mast_template}-FOUNDATION"
    replacement_root = "SEG2100-CATENARY--CATENARY-MAST-0002"
    required = {mast_template, mast_foundation_template}
    if not required.issubset(model.faces_by_object) or not required.issubset(assets):
        raise ValueError("Accepted catenary mast template assembly is incomplete")

    mast_center = _object_center_global_xy(model, mast_template, origin)
    clone_specs: list[dict[str, Any]] = []
    omit_objects: set[str] = set()
    accepted: list[dict[str, Any]] = []
    for decision in decisions.get("decisions", []):
        candidate_id = str(decision["candidate_id"])
        if candidate_id not in records:
            raise ValueError(f"Decision candidate is absent from comparison: {candidate_id}")
        record = records[candidate_id]
        target_xy = np.asarray([record["center_x"], record["center_y"]], dtype=np.float64)
        station, cross = _station_cross(target_xy, frame)
        action = str(decision["action"])
        if decision["asset_class"] == "catenary_mast":
            if action == "replace_existing_mast_0002":
                root = replacement_root
                omit_objects.update(
                    name for name in model.faces_by_object if name.startswith(replacement_root)
                )
            elif action == "add_shaft_and_foundation":
                root = f"SUPPLEMENTAL-CATENARY-MAST-{candidate_id.rsplit('-', 1)[-1]}"
            else:
                raise ValueError(f"Unsupported catenary action: {action}")
            translation = target_xy - mast_center
            clone_specs.extend(
                [
                    {
                        "source_object": mast_template,
                        "clone_object": root,
                        "translation_local_xyz_m": [*translation.tolist(), 0.0],
                        "candidate_id": candidate_id,
                        "asset_class": "catenary_mast",
                        "chainage_m": station,
                        "cross_m": cross,
                    },
                    {
                        "source_object": mast_foundation_template,
                        "clone_object": f"{root}-FOUNDATION",
                        "translation_local_xyz_m": [*translation.tolist(), 0.0],
                        "candidate_id": candidate_id,
                        "asset_class": "catenary_foundation",
                        "chainage_m": station,
                        "cross_m": cross,
                    },
                ]
            )
            accepted.append(
                {
                    "candidate_id": candidate_id,
                    "asset_class": "catenary_mast",
                    "action": action,
                    "asset_id": root,
                    "chainage_m": station,
                    "cross_m": cross,
                    "height_m": float(record["maximum_z"] - record["minimum_z"]),
                }
            )
            continue

        if decision["asset_class"] != "canopy_column" or action != "add":
            raise ValueError(f"Unsupported vertical decision: {decision}")
        source_column = str(record["nearest_asset_id"])
        if source_column not in model.faces_by_object or source_column not in assets:
            raise ValueError(f"Canopy template is absent: {source_column}")
        column_center = _object_center_global_xy(model, source_column, origin)
        translation = target_xy - column_center
        target_root = f"SUPPLEMENTAL-RIGHT-CANOPY-COLUMN-{candidate_id.rsplit('-', 1)[-1]}"
        associated = sorted(
            name
            for name in model.faces_by_object
            if name == source_column or name.startswith(f"{source_column}-")
        )
        for source_name in associated:
            suffix = source_name[len(source_column) :]
            clone_specs.append(
                {
                    "source_object": source_name,
                    "clone_object": f"{target_root}{suffix}",
                    "translation_local_xyz_m": [*translation.tolist(), 0.0],
                    "candidate_id": candidate_id,
                    "asset_class": str(assets[source_name]["type"]),
                    "chainage_m": station,
                    "cross_m": cross,
                }
            )
        accepted.append(
            {
                "candidate_id": candidate_id,
                "asset_class": "canopy_column",
                "action": action,
                "asset_id": target_root,
                "chainage_m": station,
                "cross_m": cross,
                "height_m": float(record["maximum_z"] - record["minimum_z"]),
            }
        )
    return clone_specs, omit_objects, accepted


def _update_registry(
    source_path: Path,
    output_path: Path,
    output_obj: Path,
    clone_specs: list[dict[str, Any]],
    omit_objects: set[str],
    comparison_path: Path,
) -> dict[str, Any]:
    registry = load_json(source_path)
    original_assets = {str(item["id"]): item for item in registry.get("assets", [])}
    assets = []
    removed_ids = {
        str(item["id"])
        for item in registry.get("assets", [])
        if str(item["id"]) in omit_objects
    }
    for asset in registry.get("assets", []):
        if str(asset["id"]) in removed_ids:
            continue
        if isinstance(asset.get("geometry"), dict):
            asset["geometry"]["file"] = str(output_obj)
        assets.append(asset)

    for spec in clone_specs:
        source_id = str(spec["source_object"])
        clone_id = str(spec["clone_object"])
        cloned = copy.deepcopy(original_assets[source_id])
        cloned["id"] = clone_id
        cloned["status"] = "candidate_point_continuity_confirmed"
        cloned["chainage_m"] = float(spec["chainage_m"])
        cloned["confidence"] = 0.94 if spec["asset_class"] in {"canopy_column", "catenary_mast"} else 0.80
        cloned["evidence_level"] = "supplemental_point_cloud_full_height_continuity"
        cloned["geometry"]["file"] = str(output_obj)
        cloned["geometry"]["node"] = clone_id
        cloned.setdefault("parameters", {}).update(
            {
                "source_template_asset_id": source_id,
                "supplemental_candidate_id": spec["candidate_id"],
                "cross_position_m": float(spec["cross_m"]),
                "geometry_scope": (
                    "observed_shaft" if spec["asset_class"] == "catenary_mast" else "template_component"
                ),
            }
        )
        cloned.setdefault("sources", []).append(
            {
                "kind": "point_cloud",
                "reference": str(comparison_path),
                "note": "Full-height Z-bin continuity and corridor periodicity passed.",
            }
        )
        if spec["asset_class"].startswith("catenary"):
            cloned["limitations"] = [
                "Only the point-supported mast shaft and standard foundation are emitted.",
                "Cantilever, insulator and positioner orientation remains withheld pending evidence.",
            ]
        assets.append(cloned)

    relationships = []
    for relation in registry.get("relationships", []):
        if relation.get("from") in removed_ids or relation.get("to") in removed_ids:
            continue
        relationships.append(relation)
    mast_roots = {
        str(spec["clone_object"])
        for spec in clone_specs
        if spec["asset_class"] == "catenary_mast"
    }
    for root in sorted(mast_roots):
        relationships.append(
            {
                "id": f"REL-{root}-HAS-{root}-FOUNDATION",
                "type": "has_component",
                "from": root,
                "to": f"{root}-FOUNDATION",
            }
        )
    registry["assets"] = assets
    registry["relationships"] = relationships
    registry["release_id"] = "site_b_s2050_2200_supplement_v2_continuity_candidate"
    registry["updated_at"] = datetime.now(UTC).isoformat()
    registry["summary"] = summarize_registry(registry)
    write_json(output_path, registry)
    return registry


def build_continuity_vertical_refinement(
    source_obj: str | Path,
    source_mtl: str | Path,
    source_registry: str | Path,
    model_origin: str | Path,
    comparison_path: str | Path,
    decisions_path: str | Path,
    frame_report_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Path]:
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty candidate directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    name = "site_b_s2050_2200_supplement_v2_continuity_candidate"
    output_obj = output / f"{name}.obj"
    output_mtl = output / f"{name}.mtl"
    output_registry = output / "asset_registry.json"
    output_origin = output / "model_origin.json"
    output_report = output / "continuity_vertical_refinement_report.json"
    output_mesh_audit = output / "mesh_audit.json"

    model = parse_obj_model(source_obj)
    origin = np.asarray(load_json(Path(model_origin))["origin_xyz"], dtype=np.float64)
    registry = load_json(Path(source_registry))
    comparison = load_json(Path(comparison_path))
    decisions = load_json(Path(decisions_path))
    frame = load_json(Path(frame_report_path))["frame"]
    specs, omitted, accepted = build_clone_plan(
        model, origin, registry, comparison, decisions, frame
    )
    clone_obj_objects(
        source_obj,
        output_obj,
        specs,
        output_mtl_name=output_mtl.name,
        omit_objects=omitted,
    )
    shutil.copy2(source_mtl, output_mtl)
    shutil.copy2(model_origin, output_origin)
    updated_registry = _update_registry(
        Path(source_registry),
        output_registry,
        output_obj,
        specs,
        omitted,
        Path(comparison_path).resolve(),
    )
    mesh_audit = audit_obj(output_obj)
    write_json(output_mesh_audit, mesh_audit)
    if not mesh_audit["passed"]:
        raise ValueError("Continuity-refined candidate failed OBJ mesh audit")
    write_json(
        output_report,
        {
            "schema_version": "railway.continuity-vertical-refinement.v1",
            "source_obj": str(Path(source_obj).resolve()),
            "output_obj": str(output_obj),
            "comparison": str(Path(comparison_path).resolve()),
            "decisions": str(Path(decisions_path).resolve()),
            "accepted_verticals": accepted,
            "omitted_replaced_objects": sorted(omitted),
            "clone_count": len(specs),
            "asset_count": updated_registry["summary"]["asset_count"],
            "mesh_audit": {
                "passed": mesh_audit["passed"],
                "object_count": mesh_audit["object_count"],
                "triangle_count": mesh_audit["triangle_count_after_fan_triangulation"],
                "duplicate_face_count": mesh_audit["duplicate_face_count"],
                "degenerate_triangle_count": mesh_audit["degenerate_triangle_count"],
            },
            "status": "candidate_requires_fixed_view_and_point_support_review",
            "limitations": [
                "New catenary accessories are withheld until their orientation is evidenced.",
                "Unclassified station and building-edge candidates remain non-geometric review items.",
            ],
        },
    )
    return {
        "obj": output_obj,
        "mtl": output_mtl,
        "registry": output_registry,
        "origin": output_origin,
        "mesh_audit": output_mesh_audit,
        "report": output_report,
    }
