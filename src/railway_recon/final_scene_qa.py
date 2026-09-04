from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .io import load_json, sha256_file, write_json
from .mesh_audit import audit_obj
from .model_point_support import object_vertex_indices, parse_obj_model
from .registry import validate_registry_value


def bbox_gap(
    first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]
) -> float:
    """Return the Euclidean separation of two axis-aligned boxes.

    A zero value means that the boxes touch or overlap.  This is preferable to
    vertex-to-vertex distance for long, low-tessellation assets such as masts:
    their side faces are real even when they have no intermediate vertices.
    """

    first_minimum, first_maximum = first
    second_minimum, second_maximum = second
    separation = np.maximum(
        0.0,
        np.maximum(first_minimum - second_maximum, second_minimum - first_maximum),
    )
    return float(np.linalg.norm(separation))


def object_bounds(obj_path: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    model = parse_obj_model(obj_path)
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in model.faces_by_object:
        indexes = object_vertex_indices(model, name)
        if len(indexes):
            xyz = model.vertices[indexes]
            result[name] = (xyz.min(axis=0), xyz.max(axis=0))
    return result


def _check(check_id: str, passed: bool, details: Any) -> dict[str, Any]:
    return {"id": check_id, "passed": bool(passed), "details": details}


def audit_final_scene(
    *,
    obj_path: str | Path,
    registry_path: str | Path,
    integration_report_path: str | Path,
    small_asset_report_path: str | Path,
    support_comparison_path: str | Path,
    track_closure_path: str | Path,
    corridor_view_manifest_path: str | Path,
    catenary_view_manifest_path: str | Path,
    output_path: str | Path,
    maximum_component_gap_m: float = 0.15,
) -> dict[str, Any]:
    obj_file = Path(obj_path).resolve()
    registry_file = Path(registry_path).resolve()
    integration_file = Path(integration_report_path).resolve()
    small_asset_file = Path(small_asset_report_path).resolve()
    support_comparison_file = Path(support_comparison_path).resolve()
    track_closure_file = Path(track_closure_path).resolve()
    corridor_manifest_file = Path(corridor_view_manifest_path).resolve()
    catenary_manifest_file = Path(catenary_view_manifest_path).resolve()
    for path in (
        obj_file,
        registry_file,
        integration_file,
        small_asset_file,
        support_comparison_file,
        track_closure_file,
        corridor_manifest_file,
        catenary_manifest_file,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    registry = load_json(registry_file)
    integration = load_json(integration_file)
    small_asset = load_json(small_asset_file)
    comparison = load_json(support_comparison_file)
    track_closure = load_json(track_closure_file)
    corridor_manifest = load_json(corridor_manifest_file)
    catenary_manifest = load_json(catenary_manifest_file)
    bounds = object_bounds(obj_file)
    mesh = audit_obj(obj_file)

    registry_ids = {str(asset["id"]) for asset in registry.get("assets", [])}
    relation_pairs = {
        (str(relation.get("from")), str(relation.get("to")))
        for relation in registry.get("relations", [])
        if relation.get("type") == "has_component"
    }
    assemblies: list[dict[str, Any]] = []
    cross_assembly_boxes: list[tuple[str, str, tuple[np.ndarray, np.ndarray]]] = []
    for record in integration["records"]:
        parent = str(record["mast_asset_id"])
        cantilever = f"{parent}-CANTILEVER"
        positioner = f"{parent}-POSITIONER"
        insulators = [f"{parent}-INSULATOR-01", f"{parent}-INSULATOR-02"]
        components = [cantilever, positioner, *insulators]
        missing = [name for name in [parent, *components] if name not in bounds]
        gaps: dict[str, float | None] = {
            "mast_to_cantilever_m": None,
            "cantilever_to_positioner_m": None,
            "cantilever_to_insulator_01_m": None,
            "cantilever_to_insulator_02_m": None,
        }
        if not missing:
            gaps = {
                "mast_to_cantilever_m": bbox_gap(bounds[parent], bounds[cantilever]),
                "cantilever_to_positioner_m": bbox_gap(
                    bounds[cantilever], bounds[positioner]
                ),
                "cantilever_to_insulator_01_m": bbox_gap(
                    bounds[cantilever], bounds[insulators[0]]
                ),
                "cantilever_to_insulator_02_m": bbox_gap(
                    bounds[cantilever], bounds[insulators[1]]
                ),
            }
        support_passed = all(
            bool(value.get("passed")) for value in record.get("point_support", {}).values()
        )
        relations_complete = all((parent, component) in relation_pairs for component in components)
        maximum_gap = max(
            (float(value) for value in gaps.values() if value is not None), default=float("inf")
        )
        passed = (
            not missing
            and all(component in registry_ids for component in components)
            and relations_complete
            and support_passed
            and maximum_gap <= maximum_component_gap_m
        )
        assemblies.append(
            {
                "mast_asset_id": parent,
                "component_ids": components,
                "missing_geometry_nodes": missing,
                "registry_components_complete": all(
                    component in registry_ids for component in components
                ),
                "relations_complete": relations_complete,
                "point_support_passed": support_passed,
                "connectivity_bbox_gaps": gaps,
                "maximum_connectivity_gap_m": maximum_gap,
                "passed": passed,
            }
        )
        for component in components:
            if component in bounds:
                cross_assembly_boxes.append((parent, component, bounds[component]))

    cross_assembly_overlaps: list[dict[str, str]] = []
    for index, (parent, component, component_box) in enumerate(cross_assembly_boxes):
        for other_parent, other_component, other_box in cross_assembly_boxes[index + 1 :]:
            if parent != other_parent and bbox_gap(component_box, other_box) == 0.0:
                cross_assembly_overlaps.append(
                    {"first": component, "second": other_component}
                )

    withheld = [
        str(value) for value in integration["explicitly_withheld_low_confidence_masts"]
    ]
    accidentally_built = sorted(
        name
        for name in bounds
        if any(name.startswith(f"{mast}-") for mast in withheld)
    )
    added = [str(value) for value in comparison.get("added_objects", [])]
    expected_added = sorted(
        component
        for assembly in assemblies
        for component in assembly["component_ids"]
    )
    small_summary = small_asset.get("summary", {})
    corridor_views = corridor_manifest.get("views", [])
    catenary_views = catenary_manifest.get("views", [])
    corridor_missing_images = [
        str(item.get("output") or item.get("path") or item.get("image"))
        for item in corridor_views
        if not Path(
            str(item.get("output") or item.get("path") or item.get("image"))
        ).is_file()
    ]
    catenary_missing_images = [
        str(item.get("output") or item.get("path") or item.get("image"))
        for item in catenary_views
        if not Path(
            str(item.get("output") or item.get("path") or item.get("image"))
        ).is_file()
    ]

    checks = [
        _check("mesh.structural_audit", bool(mesh["passed"]), mesh),
        _check(
            "registry.schema_and_relations",
            not validate_registry_value(registry),
            {"errors": validate_registry_value(registry)},
        ),
        _check(
            "catenary.seven_complete_supported_assemblies",
            len(assemblies) == 7 and all(item["passed"] for item in assemblies),
            assemblies,
        ),
        _check(
            "catenary.no_cross_assembly_overlap",
            not cross_assembly_overlaps,
            cross_assembly_overlaps,
        ),
        _check(
            "catenary.two_low_confidence_masts_withheld",
            len(withheld) == 2 and not accidentally_built,
            {"withheld": withheld, "accidentally_built": accidentally_built},
        ),
        _check(
            "regression.common_geometry_preserved",
            int(comparison.get("common_object_count", 0)) > 0
            and not comparison.get("removed_objects")
            and sorted(added) == expected_added,
            {
                "common_object_count": comparison.get("common_object_count"),
                "removed_objects": comparison.get("removed_objects"),
                "added_object_count": len(added),
            },
        ),
        _check(
            "track.boundary_closure_inherited",
            bool(track_closure.get("passed")),
            {
                "track_seam_p90_after_m": track_closure.get("metrics", {}).get(
                    "track_seam_p90_after_m"
                ),
                "track_seam_p95_after_m": track_closure.get("metrics", {}).get(
                    "track_seam_p95_after_m"
                ),
            },
        ),
        _check(
            "small_assets.rescan_closed_without_unresolved_new_asset",
            int(
                small_summary.get("new_semantically_unresolved_candidate_count", -1)
            )
            == 0,
            small_summary,
        ),
        _check(
            "views.corridor_six",
            len(corridor_views) == 6 and not corridor_missing_images,
            {"view_count": len(corridor_views), "missing": corridor_missing_images},
        ),
        _check(
            "views.catenary_six",
            len(catenary_views) == 6 and not catenary_missing_images,
            {"view_count": len(catenary_views), "missing": catenary_missing_images},
        ),
    ]
    passed = all(check["passed"] for check in checks)
    report = {
        "schema_version": "railway.final-scene-qa.v1",
        "candidate_obj": str(obj_file),
        "candidate_obj_sha256": sha256_file(obj_file),
        "checks": checks,
        "summary": {
            "passed": passed,
            "check_count": len(checks),
            "failed_check_count": sum(not check["passed"] for check in checks),
            "supported_catenary_assembly_count": len(assemblies),
            "added_catenary_component_count": len(expected_added),
            "withheld_low_confidence_mast_count": len(withheld),
            "new_unresolved_small_asset_count": int(
                small_summary.get("new_semantically_unresolved_candidate_count", -1)
            ),
            "corridor_fixed_view_count": len(corridor_views),
            "catenary_fixed_view_count": len(catenary_views),
        },
        "collision_policy": {
            "method": "asset-aware_AABB_connectivity_and_forbidden_cross-assembly_overlap",
            "maximum_allowed_component_connectivity_gap_m": maximum_component_gap_m,
            "intentional_contacts": [
                "mast-to-cantilever",
                "cantilever-to-positioner",
                "cantilever-to-insulator",
            ],
            "note": "Generic all-pairs collision is not a valid gate because rails, sleepers, ballast, columns, capitals and roofs intentionally touch or overlap at joints.",
        },
        "limitations": [
            "AABB checks detect detached or grossly intersecting catenary assemblies, not exact solid Boolean self-intersections.",
            "The dense supplemental cloud does not provide uniform evidence for every inferred/context object in the 200 m candidate.",
            "Photoreal material, runtime Z-fighting and engine-specific collision still require target-renderer acceptance.",
        ],
        "status": "pass_evidence_limited_freeze_candidate" if passed else "fail_not_freezable",
    }
    write_json(Path(output_path), report)
    return report
