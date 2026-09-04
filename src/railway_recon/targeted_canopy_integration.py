from __future__ import annotations

import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import ProjectConfig
from .io import load_json, sha256_file, write_json
from .mesh_audit import audit_obj
from .registry import summarize_registry, validate_registry_value


def _origin(path: Path) -> np.ndarray:
    value = load_json(path)
    origin = np.asarray(value["origin_xyz"], dtype=np.float64)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError(f"Invalid model origin: {path}")
    return origin


def _material_names(path: Path) -> set[str]:
    names: set[str] = set()
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("newmtl "):
            names.add(line.split(maxsplit=1)[1])
    return names


def _obj_material_path(obj_path: Path) -> Path:
    references = []
    for raw in obj_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("mtllib "):
            references.append(line.split(maxsplit=1)[1])
    if len(references) != 1:
        raise ValueError(
            f"Expected exactly one mtllib reference in {obj_path}; got {references}"
        )
    material_path = (obj_path.parent / references[0]).resolve()
    if not material_path.is_file():
        raise FileNotFoundError(material_path)
    return material_path


def _translate_obj_lines(
    path: Path,
    translation: np.ndarray,
    vertex_offset: int,
    texture_offset: int,
    normal_offset: int,
    namespace: str | None = None,
) -> tuple[list[str], int, int, int]:
    lines: list[str] = []
    vertex_count = 0
    texture_count = 0
    normal_count = 0
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "mtllib ")):
            continue
        parts = line.split()
        kind = parts[0]
        if kind == "v" and len(parts) >= 4:
            point = np.asarray([float(value) for value in parts[1:4]], dtype=np.float64)
            point += translation
            suffix = " " + " ".join(parts[4:]) if len(parts) > 4 else ""
            lines.append(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}{suffix}")
            vertex_count += 1
        elif kind == "vt":
            lines.append(line)
            texture_count += 1
        elif kind == "vn":
            lines.append(line)
            normal_count += 1
        elif kind in {"o", "g"} and len(parts) >= 2 and namespace:
            lines.append(f"{kind} {namespace}--{' '.join(parts[1:])}")
        elif kind == "usemtl" and len(parts) >= 2 and namespace:
            lines.append(f"usemtl {namespace}--{' '.join(parts[1:])}")
        elif kind == "f":
            adjusted: list[str] = []
            for token in parts[1:]:
                indexes = token.split("/")
                values: list[str] = []
                counts = (vertex_count, texture_count, normal_count)
                offsets = (vertex_offset, texture_offset, normal_offset)
                for position, value in enumerate(indexes):
                    if not value:
                        values.append("")
                        continue
                    index = int(value)
                    if index < 0:
                        index = counts[position] + 1 + index
                    values.append(str(index + offsets[position]))
                adjusted.append("/".join(values))
            lines.append("f " + " ".join(adjusted))
        else:
            lines.append(line)
    return lines, vertex_count, texture_count, normal_count


def merge_candidate_objs(
    sources: list[tuple[Path, Path]],
    output_obj: Path,
    output_mtl: Path,
    output_origin: Path,
    *,
    source_namespaces: list[str] | None = None,
) -> dict[str, Any]:
    if len(sources) < 2:
        raise ValueError("At least two source OBJs are required")
    if source_namespaces is not None:
        if len(source_namespaces) != len(sources):
            raise ValueError("source_namespaces must match the source count")
        if any(not value or " " in value for value in source_namespaces):
            raise ValueError("Source namespaces must be non-empty and contain no spaces")
        if len(set(source_namespaces)) != len(source_namespaces):
            raise ValueError("Source namespaces must be unique")
    namespaces = source_namespaces or [None] * len(sources)
    target_origin = _origin(sources[0][1])
    material_paths = [_obj_material_path(obj) for obj, _ in sources]
    material_sets = [_material_names(path) for path in material_paths]
    duplicates = (
        []
        if source_namespaces is not None
        else sorted(
            name
            for name in set().union(*material_sets)
            if sum(name in names for names in material_sets) > 1
        )
    )
    if duplicates:
        raise ValueError(f"Source material names collide: {duplicates}")
    output_obj.parent.mkdir(parents=True, exist_ok=True)
    obj_lines = [
        "# Site B evidence-aware integrated candidate scene",
        "# Coordinates are local metres.",
        f"# Global origin: {target_origin[0]:.6f} {target_origin[1]:.6f} {target_origin[2]:.6f}",
        f"mtllib {output_mtl.name}",
    ]
    material_lines = ["# Site B integrated candidate materials"]
    vertex_offset = texture_offset = normal_offset = 0
    source_records: list[dict[str, Any]] = []
    for (obj_path, origin_path), material_path, namespace in zip(
        sources, material_paths, namespaces, strict=True
    ):
        if not obj_path.is_file() or not origin_path.is_file():
            raise FileNotFoundError(obj_path if not obj_path.is_file() else origin_path)
        source_origin = _origin(origin_path)
        translation = source_origin - target_origin
        translated, vertices, textures, normals = _translate_obj_lines(
            obj_path,
            translation,
            vertex_offset,
            texture_offset,
            normal_offset,
            namespace,
        )
        obj_lines.extend(("", f"# source: {obj_path}", *translated))
        material_content = material_path.read_text(
            encoding="utf-8", errors="replace"
        ).strip()
        if namespace:
            material_content = "\n".join(
                f"newmtl {namespace}--{line.split(maxsplit=1)[1]}"
                if line.strip().startswith("newmtl ")
                else line
                for line in material_content.splitlines()
            )
        material_lines.extend(("", f"# source: {material_path}", material_content))
        source_records.append(
            {
                "obj": str(obj_path),
                "origin": str(origin_path),
                "namespace": namespace,
                "translation_to_target_origin_m": translation.tolist(),
                "vertex_count": vertices,
            }
        )
        vertex_offset += vertices
        texture_offset += textures
        normal_offset += normals
    temporary_obj = output_obj.with_suffix(output_obj.suffix + ".tmp")
    temporary_mtl = output_mtl.with_suffix(output_mtl.suffix + ".tmp")
    temporary_obj.write_text("\n".join(obj_lines) + "\n", encoding="utf-8", newline="\n")
    temporary_mtl.write_text(
        "\n".join(material_lines) + "\n", encoding="utf-8", newline="\n"
    )
    os.replace(temporary_obj, output_obj)
    os.replace(temporary_mtl, output_mtl)
    write_json(
        output_origin,
        {"origin_xyz": target_origin.tolist(), "units": "metre", "axis": "Z-up"},
    )
    return {
        "target_origin_xyz": target_origin.tolist(),
        "source_records": source_records,
        "vertex_count": vertex_offset,
    }


def _review_confidence(semantic_review: dict[str, Any], candidate_id: str) -> float:
    decision = next(
        (
            item
            for item in semantic_review.get("decisions", [])
            if str(item.get("candidate_id")) == candidate_id
        ),
        None,
    )
    if decision is None:
        raise ValueError(f"Missing semantic review decision for {candidate_id}")
    return float(decision["confidence"])


def _roof_confidence(surface: dict[str, Any]) -> float:
    residual = float(surface["plane_diagnostic"]["absolute_residual_p90_m"])
    return round(max(0.45, min(0.9, 0.9 - residual)), 3)


def _candidate_assets(
    recovery: dict[str, Any],
    semantic_review: dict[str, Any],
    visual_review_path: Path,
    scene_obj: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    report_reference = str(
        Path(recovery["output_mesh_audit"]).parent
        / f"{recovery['segment_id']}_targeted_canopy_recovery.json"
    )
    visual_reference = str(visual_review_path)
    grid_review = recovery.get("grid_review")
    assets: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    roof_asset_ids: dict[str, str] = {}
    for surface in recovery["roof_surfaces"]:
        object_name = f"{surface['id']}-RUN-01"
        roof_asset_ids[str(surface["id"])] = object_name
        assets.append(
            {
                "id": object_name,
                "type": "canopy_roof_surface",
                "subtype": str(surface["surface_type"]),
                "status": "candidate",
                "chainage_m": None,
                "evidence_level": "observed",
                "confidence": _roof_confidence(surface),
                "sources": [
                    {"kind": "point_cloud", "reference": report_reference},
                    {"kind": "manual_review", "reference": visual_reference},
                ],
                "parameters": {
                    "longitudinal_range_m": surface["longitudinal_range_m"],
                    "cross_range_m": surface["cross_range_m"],
                    "absolute_residual_p90_m": surface["plane_diagnostic"][
                        "absolute_residual_p90_m"
                    ],
                    "sampling": surface["surface_sampling"],
                },
                "geometry": {"node": object_name, "file": str(scene_obj)},
                "limitations": [
                    "Point-cloud-observed candidate surface; not a design roof model.",
                    "Cross-band observation gaps are deliberately preserved.",
                ],
            }
        )
    by_column = {str(item["id"]): item for item in recovery["column_grid"]}
    for node in recovery["local_column_roof_nodes"]:
        column = by_column[str(node["column_grid_id"])]
        candidate_id = str(node["reviewed_seed_id"])
        column_id = str(node["column_grid_id"])
        capital_id = f"{column_id}-CAPITAL"
        connector_id = f"{column_id}-LOCAL-UNDERROOF"
        roof_id = roof_asset_ids[str(node["nearest_observed_roof_surface_id"])]
        common_sources = [
            {"kind": "panorama", "reference": str(recovery["semantic_review"])},
            {"kind": "manual_review", "reference": visual_reference},
        ]
        assets.extend(
            [
                {
                    "id": column_id,
                    "type": "canopy_column",
                    "subtype": "photo_confirmed_rectangular_support",
                    "status": "candidate",
                    "chainage_m": None,
                    "evidence_level": "photo_interpreted",
                    "confidence": _review_confidence(semantic_review, candidate_id),
                    "sources": common_sources,
                    "parameters": {
                        "source_vertical_hypothesis_id": candidate_id,
                        "longitudinal_position_m": column["longitudinal_position_m"],
                        "cross_position_m": column["cross_position_m"],
                        "minimum_z_m": column["minimum_z"],
                        "capital_bottom_z_m": node["capital_bottom_z_m"],
                    },
                    "geometry": {"node": column_id, "file": str(scene_obj)},
                    "limitations": [
                        "Column semantics are photo interpreted from calibrated panoramas.",
                        "This record does not validate an engineering section type.",
                    ],
                },
                {
                    "id": capital_id,
                    "type": "canopy_capital",
                    "subtype": "local_photo_interpreted_node",
                    "status": "candidate",
                    "chainage_m": None,
                    "evidence_level": "photo_interpreted",
                    "confidence": float(node["confidence"]),
                    "sources": common_sources,
                    "parameters": {
                        "height_m": float(node["capital_top_z_m"])
                        - float(node["capital_bottom_z_m"]),
                        "along_size_m": node["capital_along_size_m"],
                        "cross_size_m": node["capital_cross_size_m"],
                    },
                    "geometry": {"node": capital_id, "file": str(scene_obj)},
                    "limitations": [
                        "Capital proportions are low-confidence photo interpretation.",
                    ],
                },
                {
                    "id": connector_id,
                    "type": "canopy_underroof_connector",
                    "subtype": "local_observed_boundary_bridge",
                    "status": "candidate",
                    "chainage_m": None,
                    "evidence_level": "photo_interpreted",
                    "confidence": float(node["confidence"]),
                    "sources": [
                        *common_sources,
                        {"kind": "point_cloud", "reference": report_reference},
                    ],
                    "parameters": {
                        "cross_recovery_distance_m": node[
                            "cross_recovery_distance_m"
                        ],
                        "thickness_m": 0.12,
                        "nearest_observed_roof_surface_id": roof_id,
                    },
                    "geometry": {"node": connector_id, "file": str(scene_obj)},
                    "limitations": [
                        "Connector is local and must not be repeated as a full-span roof.",
                        "Final real-time material acceptance remains required.",
                    ],
                },
            ]
        )
        relations.extend(
            [
                {
                    "id": f"{column_id}-SUPPORTS-{capital_id}",
                    "type": "supports",
                    "from": column_id,
                    "to": capital_id,
                },
                {
                    "id": f"{capital_id}-SUPPORTS-{connector_id}",
                    "type": "supports",
                    "from": capital_id,
                    "to": connector_id,
                },
                {
                    "id": f"{connector_id}-CONNECTS-{roof_id}",
                    "type": "connects_to",
                    "from": connector_id,
                    "to": roof_id,
                },
            ]
        )
    if grid_review:
        for asset in assets:
            asset["sources"].append(
                {
                    "kind": "manual_review",
                    "reference": str(grid_review),
                    "note": "constant-spacing grid rejected; only reviewed columns retained",
                }
            )
    return assets, relations


def audit_column_platform_contacts(
    recovery: dict[str, Any],
    platform: dict[str, Any],
    tolerance_m: float = 0.08,
) -> dict[str, Any]:
    component_id = str(recovery["platform_component_id"])
    component = next(
        (
            item
            for item in platform.get("platform_components", [])
            if str(item.get("id")) == component_id
        ),
        None,
    )
    if component is None:
        raise ValueError(f"Platform component is absent: {component_id}")
    results: list[dict[str, Any]] = []
    for column in recovery["column_grid"]:
        if column.get("status") != "confirmed_photo_seed":
            continue
        longitudinal = float(column["longitudinal_position_m"])
        cross = float(column["cross_position_m"])
        fit = next(
            (
                item
                for item in component["fit_segments"]
                if float(item["longitudinal_range_m"][0]) - 1e-8
                <= longitudinal
                <= float(item["longitudinal_range_m"][1]) + 1e-8
            ),
            None,
        )
        if fit is None:
            results.append(
                {
                    "column_grid_id": column["id"],
                    "status": "no_platform_fit_at_column",
                }
            )
            continue
        low_cross, high_cross = (float(value) for value in fit["observed_cross_range_m"])
        if not low_cross - 1e-8 <= cross <= high_cross + 1e-8:
            results.append(
                {
                    "column_grid_id": column["id"],
                    "status": "column_outside_observed_platform_cross_range",
                    "observed_cross_range_m": [low_cross, high_cross],
                }
            )
            continue
        a, b, d = (float(value) for value in fit["plane_z_equals_a_s_plus_b_c_plus_d"])
        platform_z = a * longitudinal + b * cross + d
        column_base = float(column["minimum_z"])
        gap = column_base - platform_z
        passed = abs(gap) <= tolerance_m
        results.append(
            {
                "column_grid_id": column["id"],
                "reviewed_seed_id": column["reviewed_seed_id"],
                "longitudinal_position_m": longitudinal,
                "cross_position_m": cross,
                "column_base_z_m": column_base,
                "platform_fit_z_m": platform_z,
                "vertical_gap_m": gap,
                "maximum_allowed_absolute_gap_m": tolerance_m,
                "status": "pass" if passed else "column_platform_vertical_gap",
            }
        )
    unresolved = [item for item in results if item["status"] != "pass"]
    return {
        "schema_version": "railway.column-platform-contact-audit.v1",
        "platform_component_id": component_id,
        "confirmed_column_count": len(results),
        "pass_count": len(results) - len(unresolved),
        "unresolved_count": len(unresolved),
        "maximum_allowed_absolute_gap_m": tolerance_m,
        "contacts": results,
        "passed": not unresolved and bool(results),
        "status": "pass" if not unresolved and results else "fail",
    }


def integrate_targeted_canopy_candidate(
    project: ProjectConfig,
    segment_id: str,
    recovery_report_path: str | Path,
    visual_review_path: str | Path,
    release_id: str,
    approve_candidate_merge: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not approve_candidate_merge:
        raise PermissionError("Candidate integration requires --approve-candidate-merge")
    recovery_path = project.resolve(recovery_report_path)
    visual_path = project.resolve(visual_review_path)
    recovery = load_json(recovery_path)
    visual = load_json(visual_path)
    if recovery.get("segment_id") != segment_id or visual.get("segment_id") != segment_id:
        raise ValueError("Segment mismatch between integration inputs")
    if visual.get("candidate_visual_gate") != "passed":
        raise ValueError("Targeted canopy visual gate has not passed")
    if visual.get("asset_registry_write") is not False:
        raise ValueError("Visual review input is not an immutable pre-merge review")
    if recovery.get("local_node_chain_audit", {}).get("candidate_chain_gate") != "passed":
        raise ValueError("Local node chain gate has not passed")
    if any(
        item.get("status") != "confirmed_photo_seed"
        for item in recovery.get("column_grid", [])
        if item.get("reviewed_seed_id") is not None
    ):
        raise ValueError("A reviewed canopy seed is no longer confirmed")
    approved_columns = {
        str(item["column_grid_id"]) for item in recovery["local_column_roof_nodes"]
    }
    if len(approved_columns) != 3:
        raise ValueError("Integration is locked to the three reviewed local nodes")

    output_dir = project.workspace_path("exports") / release_id
    output_obj = output_dir / f"{release_id}.obj"
    output_mtl = output_dir / f"{release_id}.mtl"
    output_origin = output_dir / "model_origin.json"
    audit_path = project.workspace_path("reports") / f"{release_id}_mesh_audit.json"
    mapping_path = project.workspace_path("reports") / f"{release_id}_asset_mapping_audit.json"
    contact_path = project.workspace_path("reports") / f"{release_id}_column_platform_contact_audit.json"
    integration_path = project.workspace_path("reports") / f"{release_id}_integration.json"
    versioned_registry = output_dir / "asset_registry.json"
    all_outputs = (
        output_obj,
        output_mtl,
        output_origin,
        audit_path,
        mapping_path,
        contact_path,
        integration_path,
        versioned_registry,
    )
    if not overwrite:
        existing = [str(path) for path in all_outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite integration outputs: {existing}")

    platform_obj = (
        project.workspace_path("exports")
        / segment_id
        / "platform_candidate"
        / "platform_candidate.obj"
    )
    platform_origin = platform_obj.parent / "model_origin.json"
    canopy_obj = Path(recovery["output_obj"])
    canopy_origin = Path(recovery["output_origin"])
    merge = merge_candidate_objs(
        [(platform_obj, platform_origin), (canopy_obj, canopy_origin)],
        output_obj,
        output_mtl,
        output_origin,
    )
    mesh_audit = audit_obj(output_obj)
    mesh_audit["status"] = "pass" if mesh_audit["passed"] else "fail"
    write_json(audit_path, mesh_audit)
    if not mesh_audit["passed"]:
        raise ValueError("Integrated candidate OBJ failed mesh audit")

    platform_report = load_json(Path(recovery["platform_report"]))
    contact_audit = audit_column_platform_contacts(recovery, platform_report)
    contact_audit.update(
        {
            "project_id": project.project_id,
            "segment_id": segment_id,
            "release_id": release_id,
        }
    )
    write_json(contact_path, contact_audit)
    if not contact_audit["passed"]:
        raise ValueError("Integrated canopy columns failed platform contact audit")

    registry_path = project.workspace_path("asset_registry")
    registry = load_json(registry_path)
    semantic_review = load_json(Path(recovery["semantic_review"]))
    canopy_assets, canopy_relations = _candidate_assets(
        recovery, semantic_review, visual_path, output_obj
    )
    incoming_ids = {str(item["id"]) for item in canopy_assets}
    registry["assets"] = [
        item for item in registry.get("assets", []) if str(item["id"]) not in incoming_ids
    ]
    for asset in registry["assets"]:
        geometry = asset.get("geometry")
        if isinstance(geometry, dict) and geometry.get("node") in mesh_audit["object_names"]:
            geometry["file"] = str(output_obj)
    registry["assets"].extend(canopy_assets)
    relation_ids = {str(item["id"]) for item in canopy_relations}
    registry["relations"] = [
        item
        for item in registry.get("relations", [])
        if str(item.get("id")) not in relation_ids
    ]
    registry["relations"].extend(canopy_relations)
    registry["updated_at"] = datetime.now(UTC).isoformat()
    registry["candidate_scene"] = {
        "release_id": release_id,
        "model": str(output_obj),
        "status": "candidate",
        "formal_release": False,
    }
    registry["summary"] = summarize_registry(registry)
    errors = validate_registry_value(registry)
    if errors:
        raise ValueError("Integrated asset registry is invalid: " + "; ".join(errors))

    object_names = set(mesh_audit["object_names"])
    missing_geometry_nodes = sorted(
        str(asset["geometry"]["node"])
        for asset in registry["assets"]
        if isinstance(asset.get("geometry"), dict)
        and asset["geometry"].get("node") not in object_names
    )
    rejected_geometry = sorted(
        name
        for name in object_names
        if name.startswith("RIGHT-CANOPY-GRID-")
        and not any(name.startswith(column_id) for column_id in approved_columns)
    )
    mapping_audit = {
        "schema_version": "railway.candidate-scene-asset-mapping-audit.v1",
        "project_id": project.project_id,
        "segment_id": segment_id,
        "release_id": release_id,
        "object_count": len(object_names),
        "registry_asset_count": len(registry["assets"]),
        "missing_geometry_nodes": missing_geometry_nodes,
        "rejected_grid_geometry_nodes": rejected_geometry,
        "approved_column_geometry_nodes": sorted(approved_columns),
        "passed": not missing_geometry_nodes and not rejected_geometry,
        "status": "pass" if not missing_geometry_nodes and not rejected_geometry else "fail",
    }
    write_json(mapping_path, mapping_audit)
    if not mapping_audit["passed"]:
        raise ValueError("Integrated candidate scene failed asset mapping audit")

    write_json(versioned_registry, registry)
    history_dir = registry_path.parent / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    backup_path = history_dir / f"asset_registry.before_{release_id}.json"
    if not backup_path.exists():
        shutil.copy2(registry_path, backup_path)
    write_json(registry_path, registry)
    report = {
        "schema_version": "railway.targeted-canopy-integration.v1",
        "project_id": project.project_id,
        "segment_id": segment_id,
        "release_id": release_id,
        "status": "candidate_scene_integrated",
        "formal_release": False,
        "approved_column_count": len(approved_columns),
        "roof_surface_count": len(recovery["roof_surfaces"]),
        "rejected_grid_geometry_count": len(rejected_geometry),
        "asset_count": len(registry["assets"]),
        "asset_count_added": len(canopy_assets),
        "relation_count_added": len(canopy_relations),
        "merge": merge,
        "output_obj": str(output_obj),
        "output_mtl": str(output_mtl),
        "output_origin": str(output_origin),
        "output_registry": str(versioned_registry),
        "canonical_registry": str(registry_path),
        "registry_backup": str(backup_path),
        "output_mesh_audit": str(audit_path),
        "output_mapping_audit": str(mapping_path),
        "output_contact_audit": str(contact_path),
        "model_sha256": sha256_file(output_obj),
        "registry_sha256": sha256_file(versioned_registry),
        "limitations": [
            "This is a versioned candidate scene, not an accepted or frozen release.",
            "Only the three photo-reviewed canopy supports are included.",
            "The rejected constant-spacing column grid remains absent.",
            "Final UE/PBR material acceptance remains outstanding.",
        ],
    }
    write_json(integration_path, report)
    return report
