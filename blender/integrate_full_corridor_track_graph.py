"""Replace every legacy track asset in a complete-scene GLB with one TrackGraph mesh.

This is deliberately a versioned candidate builder.  It never edits the source
GLB and it keeps rule-inferred turnout rails visibly traceable in the registry.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import integrate_track_graph as integration


def arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-glb", type=Path, required=True)
    parser.add_argument("--track-obj", type=Path, required=True)
    parser.add_argument("--track-origin", type=Path, required=True)
    parser.add_argument("--full-origin", type=Path, required=True)
    parser.add_argument("--track-registry", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--chainage-start-m", type=float, default=0.0)
    parser.add_argument("--chainage-end-m", type=float, required=True)
    parser.add_argument("--chainage-origin-local-y-m", type=float, required=True)
    parser.add_argument("--profile", default="v9.6-trackgraph-visual-candidate-01")
    return parser.parse_args(values)


def release_fields(record: dict[str, Any]) -> dict[str, Any]:
    """Assign an honest delivery class without hiding rule-inferred geometry."""
    identifier = str(record["id"])
    evidence = str(record.get("evidenceScope") or "unknown")
    asset_type = str(record.get("type") or "")
    is_turnout_display = identifier.startswith("TURNOUT-")
    if is_turnout_display:
        return {
            "releaseClass": "visual_context",
            "defaultVisible": True,
            "geometryPromotionAllowed": False,
            "professionalStatus": "display_only_unresolved_turnout",
            "excludedFromPrecisionStatistics": True,
        }
    if asset_type in {"SleeperArray", "TrackBed"}:
        return {
            "releaseClass": "accepted_relative_geometry",
            "defaultVisible": True,
            "geometryPromotionAllowed": True,
            "professionalStatus": "accepted_parametric_geometry",
            "excludedFromPrecisionStatistics": True,
        }
    if evidence == "observed":
        return {
            "releaseClass": "accepted_relative_geometry",
            "defaultVisible": True,
            "geometryPromotionAllowed": True,
            "professionalStatus": "accepted_under_current_evidence",
            "excludedFromPrecisionStatistics": False,
        }
    return {
        "releaseClass": "visual_context",
        "defaultVisible": True,
        "geometryPromotionAllowed": False,
        "professionalStatus": "topology_inferred_display_context",
        "excludedFromPrecisionStatistics": True,
    }


def main() -> None:
    args = arguments()
    source_glb = args.source_glb.resolve()
    track_obj = args.track_obj.resolve()
    source_document = integration.read_glb_json(source_glb)
    source_registry = list(source_document.get("extras", {}).get("assetRegistry", []))
    if not source_registry:
        raise RuntimeError("Source GLB has no embedded asset registry")

    replacement_assets = [
        asset
        for asset in source_registry
        if str(asset.get("type")) in integration.REPLACE_TYPES
    ]
    replacement_ids = {str(asset["id"]) for asset in replacement_assets}
    if not replacement_ids:
        raise RuntimeError("Source scene has no legacy Rail/SleeperArray/TrackBed assets")

    sidecar = integration.read_json(args.track_registry.resolve())
    sidecar_assets = list(sidecar.get("assets", []))
    expected_object_ids = {
        str(asset["geometry"]["node"])
        for asset in sidecar_assets
        if isinstance(asset.get("geometry"), dict) and asset["geometry"].get("node")
    }
    if not expected_object_ids:
        raise RuntimeError("TrackGraph sidecar contains no geometry nodes")

    integration.reset_scene()
    bpy.ops.import_scene.gltf(filepath=str(source_glb), import_shading="NORMALS")
    source_objects = integration.mesh_objects()
    remove_objects = [
        obj for obj in source_objects if integration.asset_id(obj) in replacement_ids
    ]
    removed_object_ids = {integration.asset_id(obj) for obj in remove_objects}
    if removed_object_ids != replacement_ids:
        missing = sorted(replacement_ids - removed_object_ids)
        raise RuntimeError(f"Registry replacement assets have no mesh objects: {missing}")
    for obj in remove_objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    preserved_objects = integration.mesh_objects()

    before_import = set(bpy.context.scene.objects)
    bpy.ops.wm.obj_import(filepath=str(track_obj), forward_axis="NEGATIVE_Y", up_axis="Z")
    track_objects = [
        obj
        for obj in bpy.context.scene.objects
        if obj not in before_import and obj.type == "MESH"
    ]
    imported_ids = {obj.name for obj in track_objects}
    if imported_ids != expected_object_ids:
        raise RuntimeError(
            "TrackGraph OBJ/sidecar mismatch: "
            f"missing={sorted(expected_object_ids-imported_ids)}, "
            f"unexpected={sorted(imported_ids-expected_object_ids)}"
        )

    transform = integration.transform_track_objects(
        track_objects,
        integration.raw_obj_bounds(track_obj),
        Vector(integration.read_json(args.track_origin.resolve())["origin_xyz"]),
        Vector(integration.read_json(args.full_origin.resolve())["global_origin"]),
        args.chainage_origin_local_y_m,
    )
    # Blender's cached ``Object.bound_box`` can remain stale immediately after
    # bulk vertex edits.  The shared transform report is retained for backward
    # compatibility, but release coverage must be checked from actual vertices.
    transformed_range = integration.aggregate_geometry_record(track_objects)["bounds"]
    transform["transformed_vertex_bounds_local_xyz"] = transformed_range
    actual_start = float(transformed_range["min"][1])
    actual_end = float(transformed_range["max"][1])
    engineering_span = args.chainage_end_m - args.chainage_start_m
    expected_start = args.chainage_origin_local_y_m - engineering_span
    expected_end = args.chainage_origin_local_y_m
    if actual_start > expected_start + 1.0 or actual_end < expected_end - 1.0:
        raise RuntimeError(
            "Transformed TrackGraph does not cover the engineering corridor: "
            f"actual={actual_start}..{actual_end}, expected={expected_start}..{expected_end}"
        )

    layer = f"trackgraph_full_corridor_{args.profile}"
    new_assets = integration.convert_track_registry(
        args.track_registry.resolve(),
        track_objects,
        layer,
        args.profile,
        args.chainage_start_m,
        args.chainage_end_m,
    )
    for asset in new_assets:
        asset["parentId"] = "Corridor_s000_879m"
        asset.update(release_fields(asset))
        asset.setdefault("parameters", {})["full_corridor_replacement"] = True

    combined_objects = preserved_objects + track_objects
    combined_registry = [
        json.loads(json.dumps(asset))
        for asset in source_registry
        if str(asset.get("id")) not in replacement_ids
    ] + new_assets
    for index, asset in enumerate(combined_registry):
        asset["index"] = index
        parameters = asset.setdefault("parameters", {})
        if isinstance(parameters, dict):
            parameters["ue_render_profile"] = args.profile

    object_ids = {integration.asset_id(obj) for obj in combined_objects}
    registry_ids = {str(asset["id"]) for asset in combined_registry}
    if object_ids != registry_ids:
        raise RuntimeError(
            f"Object/registry mismatch: missing={sorted(object_ids-registry_ids)[:10]}, "
            f"unused={sorted(registry_ids-object_ids)[:10]}"
        )
    index_by_id = {str(asset["id"]): int(asset["index"]) for asset in combined_registry}
    for obj in combined_objects:
        identifier = integration.asset_id(obj)
        obj["assetId"] = identifier
        obj["assetIndex"] = index_by_id[identifier]
        obj["ueRenderProfile"] = "ue_clean_main"
        obj["v63_revision"] = args.profile

    summary = integration.geometry_summary(combined_objects)
    if summary["invalid_or_zero_area_polygon_count"] or summary["nonfinite_vertex_count"]:
        raise RuntimeError(f"Integrated scene failed pre-export geometry checks: {summary}")
    integration_record = {
        "profile": args.profile,
        "source_glb_sha256": integration.sha256(source_glb),
        "track_obj_sha256": integration.sha256(track_obj),
        "track_sidecar_sha256": integration.sha256(args.track_registry.resolve()),
        "replacement_scope": "all Rail/SleeperArray/TrackBed assets",
        "removed_asset_count": len(replacement_ids),
        "removed_object_count": len(remove_objects),
        "added_asset_count": len(new_assets),
        "added_object_count": len(track_objects),
        "chainage_range_m": [args.chainage_start_m, args.chainage_end_m],
        "transform": transform,
        "turnout_policy": "display-only rule-inferred rails; no dimensional claim",
    }
    paths = integration.export_outputs(
        combined_objects,
        args.output_prefix.resolve(),
        combined_registry,
        args.profile,
        integration_record,
    )
    report = {
        "schema_version": "railway.trackgraph-full-corridor-integration.v1",
        "status": "automatic_checks_passed_visual_review_required",
        "profile": args.profile,
        "source": str(source_glb),
        "integration": integration_record,
        "geometry": summary,
        "registry_asset_count": len(combined_registry),
        "removed_asset_ids": sorted(replacement_ids),
        "added_asset_ids": sorted(str(asset["id"]) for asset in new_assets),
        "outputs": {
            role: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": integration.sha256(path),
            }
            for role, path in paths.items()
        },
        "limitations": [
            "Turnout sleepers, ballast transitions, switch blades, frogs and guard rails are unresolved.",
            "Rule-inferred rail intervals remain visual context and are excluded from precision statistics.",
            "No confirmed CRS, vertical datum or independent control points are available.",
        ],
    }
    integration.write_json(args.report.resolve(), report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "objects": summary["object_count"],
                "assets": len(combined_registry),
                "triangles": summary["triangle_count"],
                "removed_objects": len(remove_objects),
                "added_objects": len(track_objects),
                "outputs": {role: str(path) for role, path in paths.items()},
                "report": str(args.report.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
