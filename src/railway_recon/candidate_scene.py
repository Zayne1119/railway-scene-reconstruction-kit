from __future__ import annotations

import shutil
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .io import load_json, sha256_file, write_json
from .mesh_audit import audit_obj
from .registry import summarize_registry, validate_registry_value
from .targeted_canopy_integration import merge_candidate_objs


def _component(value: dict[str, str], project_id: str) -> dict[str, Any]:
    obj = Path(value["obj"]).resolve()
    origin = Path(value["origin"]).resolve()
    registry_path = Path(value["registry"]).resolve()
    for path in (obj, origin, registry_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    registry = load_json(registry_path)
    if registry.get("project_id") != project_id:
        raise ValueError(f"Component registry project mismatch: {registry_path}")
    return {
        "obj": obj,
        "origin": origin,
        "registry_path": registry_path,
        "registry": registry,
    }


def compose_candidate_scene(
    project: ProjectConfig,
    release_id: str,
    components: list[dict[str, str]],
    *,
    rejected_asset_ids: set[str] | None = None,
    extra_relations: list[dict[str, str]] | None = None,
    update_canonical_registry: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Compose reviewed candidate components without promoting a formal release."""
    if not release_id or Path(release_id).name != release_id:
        raise ValueError("release_id must be one safe path component")
    if len(components) < 2:
        raise ValueError("At least two candidate components are required")
    rejected = set(rejected_asset_ids or set())
    requested_relations = list(extra_relations or [])
    resolved = [_component(value, project.project_id) for value in components]

    asset_ids: list[str] = []
    relation_ids: list[str] = []
    for item in resolved:
        asset_ids.extend(str(asset["id"]) for asset in item["registry"].get("assets", []))
        relation_ids.extend(
            str(relation["id"]) for relation in item["registry"].get("relations", [])
        )
    duplicate_assets = sorted(
        {value for value in asset_ids if asset_ids.count(value) > 1}
    )
    duplicate_relations = sorted(
        {value for value in relation_ids if relation_ids.count(value) > 1}
    )
    if duplicate_assets or duplicate_relations:
        raise ValueError(
            f"Candidate component identifier collision: assets={duplicate_assets}, "
            f"relations={duplicate_relations}"
        )
    rejected_present = sorted(rejected.intersection(asset_ids))
    if rejected_present:
        raise ValueError(f"Rejected assets are present in candidate inputs: {rejected_present}")

    output_dir = project.workspace_path("exports") / release_id
    output_obj = output_dir / f"{release_id}.obj"
    output_mtl = output_dir / f"{release_id}.mtl"
    output_origin = output_dir / "model_origin.json"
    output_registry = output_dir / "asset_registry.json"
    report_dir = project.workspace_path("reports")
    mesh_audit_path = report_dir / f"{release_id}_mesh_audit.json"
    mapping_audit_path = report_dir / f"{release_id}_asset_mapping_audit.json"
    composition_path = report_dir / f"{release_id}_composition.json"
    outputs = (
        output_obj,
        output_mtl,
        output_origin,
        output_registry,
        mesh_audit_path,
        mapping_audit_path,
        composition_path,
    )
    if not overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite composition outputs: {existing}")

    merge = merge_candidate_objs(
        [(item["obj"], item["origin"]) for item in resolved],
        output_obj,
        output_mtl,
        output_origin,
    )
    mesh_audit = audit_obj(output_obj)
    mesh_audit["status"] = "pass" if mesh_audit["passed"] else "fail"
    write_json(mesh_audit_path, mesh_audit)
    if not mesh_audit["passed"]:
        raise ValueError("Composed candidate OBJ failed mesh audit")

    registry = {
        "schema_version": "railway.asset-registry.v1",
        "project_id": project.project_id,
        "updated_at": datetime.now(UTC).isoformat(),
        "assets": [],
        "relations": [],
    }
    for item in resolved:
        registry["assets"].extend(item["registry"].get("assets", []))
        registry["relations"].extend(item["registry"].get("relations", []))
    existing_relation_ids = {
        str(item["id"]) for item in registry.get("relations", [])
    }
    requested_relation_ids = [str(item.get("id")) for item in requested_relations]
    duplicate_requested = sorted(
        {
            value
            for value in requested_relation_ids
            if requested_relation_ids.count(value) > 1 or value in existing_relation_ids
        }
    )
    if duplicate_requested:
        raise ValueError(
            f"Candidate extra relation identifier collision: {duplicate_requested}"
        )
    registry["relations"].extend(requested_relations)
    for asset in registry["assets"]:
        geometry = asset.get("geometry")
        if isinstance(geometry, dict):
            geometry["file"] = str(output_obj)
    registry["candidate_scene"] = {
        "release_id": release_id,
        "model": str(output_obj),
        "status": "candidate",
        "formal_release": False,
    }
    registry["summary"] = summarize_registry(registry)
    registry_errors = validate_registry_value(registry)
    if registry_errors:
        raise ValueError("Composed asset registry is invalid: " + "; ".join(registry_errors))

    object_names = set(mesh_audit["object_names"])
    geometry_nodes = {
        str(asset["geometry"]["node"])
        for asset in registry["assets"]
        if isinstance(asset.get("geometry"), dict) and asset["geometry"].get("node")
    }
    missing_nodes = sorted(geometry_nodes - object_names)
    unmapped_objects = sorted(object_names - geometry_nodes)
    rejected_objects = sorted(
        name for name in object_names if any(name == value or name.startswith(value + "-") for value in rejected)
    )
    mapping_audit = {
        "schema_version": "railway.candidate-scene-asset-mapping-audit.v1",
        "project_id": project.project_id,
        "release_id": release_id,
        "object_count": len(object_names),
        "registry_asset_count": len(registry["assets"]),
        "geometry_node_count": len(geometry_nodes),
        "missing_geometry_nodes": missing_nodes,
        "unmapped_object_names": unmapped_objects,
        "rejected_asset_ids": sorted(rejected),
        "rejected_geometry_nodes": rejected_objects,
        "passed": not missing_nodes and not unmapped_objects and not rejected_objects,
    }
    mapping_audit["status"] = "pass" if mapping_audit["passed"] else "fail"
    write_json(mapping_audit_path, mapping_audit)
    if not mapping_audit["passed"]:
        raise ValueError("Composed candidate scene failed asset mapping audit")

    write_json(output_registry, registry)
    registry_path = project.workspace_path("asset_registry")
    backup_path: Path | None = None
    if update_canonical_registry:
        history_dir = registry_path.parent / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        backup_path = history_dir / f"asset_registry.before_{release_id}.json"
        if registry_path.is_file() and not backup_path.exists():
            shutil.copy2(registry_path, backup_path)
        write_json(registry_path, registry)

    report = {
        "schema_version": "railway.candidate-scene-composition.v1",
        "project_id": project.project_id,
        "release_id": release_id,
        "status": "candidate_scene_composed",
        "formal_release": False,
        "component_count": len(resolved),
        "components": [
            {
                "obj": str(item["obj"]),
                "obj_sha256": sha256_file(item["obj"]),
                "origin": str(item["origin"]),
                "registry": str(item["registry_path"]),
                "registry_sha256": sha256_file(item["registry_path"]),
            }
            for item in resolved
        ],
        "merge": merge,
        "asset_count": len(registry["assets"]),
        "relation_count": len(registry["relations"]),
        "extra_relation_count": len(requested_relations),
        "rejected_asset_ids": sorted(rejected),
        "output_obj": str(output_obj),
        "output_mtl": str(output_mtl),
        "output_origin": str(output_origin),
        "output_registry": str(output_registry),
        "canonical_registry": str(registry_path) if update_canonical_registry else None,
        "registry_backup": str(backup_path) if backup_path else None,
        "output_mesh_audit": str(mesh_audit_path),
        "output_mapping_audit": str(mapping_audit_path),
        "model_sha256": sha256_file(output_obj),
        "registry_sha256": sha256_file(output_registry),
        "limitations": [
            "This is a working candidate scene, not an accepted or frozen release.",
            "Component evidence levels remain unchanged during composition.",
            "Fixed-view and real-time renderer acceptance remain required.",
        ],
    }
    write_json(composition_path, report)
    return report


def audit_track_platform_candidate(
    graph: dict[str, Any],
    platform_report: dict[str, Any],
    rail_report: dict[str, Any],
    track_build: dict[str, Any],
    *,
    rejected_track_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Audit accepted-track continuity, overlap and the observed platform interface."""
    rejected = set(rejected_track_ids or set())
    observations = sorted(
        graph.get("observations", []), key=lambda item: float(item["lateral_offset_m"])
    )
    bed_width = float(track_build["track_bed"]["bottom_width_m"])
    track_checks = []
    for observation in observations:
        track_checks.append(
            {
                "track_id": observation["global_track_id"],
                "source_track_id": observation["local_track_id"],
                "joint_support_ratio": observation.get("joint_support_ratio"),
                "asymmetric_support_ratio": observation.get("asymmetric_support_ratio"),
                "maximum_internal_joint_gap_m": observation.get(
                    "maximum_internal_joint_gap_m"
                ),
                "fitted_rail_top_crosslevel_m": observation.get(
                    "rail_top_crosslevel_m"
                ),
                "source_reported_rail_top_crosslevel_m": observation.get(
                    "source_reported_rail_top_crosslevel_m"
                ),
                "continuity_status": observation.get("pair_continuity_status"),
                "status": (
                    "pass"
                    if observation.get("pair_continuity_status") == "pass"
                    and float(observation.get("maximum_internal_joint_gap_m", 1.0)) <= 0.0
                    else "fail"
                ),
            }
        )
    spacing_checks = []
    for left, right in pairwise(observations):
        centre_spacing = float(right["lateral_offset_m"]) - float(
            left["lateral_offset_m"]
        )
        bed_clearance = centre_spacing - bed_width
        spacing_checks.append(
            {
                "left_track_id": left["global_track_id"],
                "right_track_id": right["global_track_id"],
                "track_center_spacing_m": centre_spacing,
                "parametric_bed_edge_clearance_m": bed_clearance,
                "status": "pass" if bed_clearance >= 0.0 else "track_bed_overlap",
            }
        )

    rail_longitudinal = [float(value) for value in rail_report["longitudinal_range_m"]]
    nearest = max(
        observations,
        key=lambda item: max(float(value) for value in item["rail_cross_positions_local_m"]),
    )
    nearest_outer_rail = max(
        float(value) for value in nearest["rail_cross_positions_local_m"]
    )
    nearest_bed_edge = float(nearest["lateral_offset_m"]) + bed_width * 0.5
    interface_samples = []
    for component in platform_report.get("platform_components", []):
        for fit in component.get("fit_segments", []):
            low, high = (float(value) for value in fit["longitudinal_range_m"])
            midpoint = 0.5 * (low + high)
            if midpoint < rail_longitudinal[0] or midpoint > rail_longitudinal[1]:
                continue
            edge = float(fit["rail_side_edge_cross_m"])
            a, b, d = (
                float(value) for value in fit["plane_z_equals_a_s_plus_b_c_plus_d"]
            )
            platform_z = a * midpoint + b * edge + d
            fraction = (midpoint - rail_longitudinal[0]) / max(
                rail_longitudinal[1] - rail_longitudinal[0], 1.0e-9
            )
            rail_top_z = float(nearest["rail_top_start_z_m"]) + fraction * (
                float(nearest["rail_top_end_z_m"])
                - float(nearest["rail_top_start_z_m"])
            )
            interface_samples.append(
                {
                    "platform_component_id": component["id"],
                    "longitudinal_midpoint_m": midpoint,
                    "platform_rail_side_edge_cross_m": edge,
                    "nearest_accepted_track_id": nearest["global_track_id"],
                    "nearest_outer_rail_cross_m": nearest_outer_rail,
                    "rail_to_platform_horizontal_gap_m": edge - nearest_outer_rail,
                    "bed_to_platform_horizontal_gap_m": edge - nearest_bed_edge,
                    "platform_top_z_m": platform_z,
                    "nearest_track_rail_top_z_m": rail_top_z,
                    "platform_above_rail_top_m": platform_z - rail_top_z,
                }
            )
    horizontal_overlap = any(
        float(item["rail_to_platform_horizontal_gap_m"]) < 0.0
        or float(item["bed_to_platform_horizontal_gap_m"]) < 0.0
        for item in interface_samples
    )
    accepted_source_ids = {str(item["local_track_id"]) for item in observations}
    rejected_present = sorted(accepted_source_ids.intersection(rejected))
    geometry_passed = (
        bool(track_checks)
        and all(item["status"] == "pass" for item in track_checks)
        and all(item["status"] == "pass" for item in spacing_checks)
        and not horizontal_overlap
        and not rejected_present
    )
    unresolved_platform_adjacent = bool(rejected)
    return {
        "schema_version": "railway.track-platform-candidate-audit.v1",
        "accepted_track_ids": sorted(accepted_source_ids),
        "rejected_track_ids": sorted(rejected),
        "rejected_tracks_present": rejected_present,
        "track_continuity": track_checks,
        "track_spacing": spacing_checks,
        "platform_interface_samples": interface_samples,
        "interface_sample_count": len(interface_samples),
        "horizontal_overlap_detected": horizontal_overlap,
        "geometry_checks_passed": geometry_passed,
        "completeness_status": (
            "review_required" if unresolved_platform_adjacent else "complete_for_scope"
        ),
        "status": "review_required" if geometry_passed else "fail",
        "limitations": [
            "Clearances are model-to-model diagnostics, not railway design acceptance limits.",
            "Rejected platform-adjacent rail candidates remain an explicit scene gap.",
            "Absolute surveying accuracy is not claimed without control points and a verified CRS.",
        ],
    }
