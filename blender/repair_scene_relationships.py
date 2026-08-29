"""Repair explicit column contact failures found by the relationship audit."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import bmesh
import bpy
from mathutils import Vector

BLENDER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BLENDER_DIR))
integration = importlib.import_module("integrate_track_graph")


def arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-glb", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profile", default="v9.2-relations-01")
    parser.add_argument("--contact-overlap-m", type=float, default=0.02)
    parser.add_argument("--top-transition-band-m", type=float, default=0.90)
    parser.add_argument("--bottom-transition-band-m", type=float, default=0.40)
    return parser.parse_args(values)


def engineering_z(point: Vector) -> float:
    return float(-point.y)


def move_boundary(
    objects: list[bpy.types.Object],
    *,
    boundary: str,
    displacement_m: float,
    transition_band_m: float,
) -> dict[str, Any]:
    world_points = [
        obj.matrix_world @ vertex.co for obj in objects for vertex in obj.data.vertices
    ]
    elevations = [engineering_z(point) for point in world_points]
    minimum = min(elevations)
    maximum = max(elevations)
    moved = 0
    for obj in objects:
        inverse = obj.matrix_world.inverted()
        for vertex in obj.data.vertices:
            world = obj.matrix_world @ vertex.co
            elevation = engineering_z(world)
            if boundary == "top":
                weight = 1.0 if elevation >= maximum - transition_band_m else 0.0
                delta = displacement_m * weight
            else:
                weight = 1.0 if elevation <= minimum + transition_band_m else 0.0
                delta = -displacement_m * weight
            if not delta:
                continue
            world.y -= delta
            vertex.co = inverse @ world
            moved += 1
        mesh = bmesh.new()
        mesh.from_mesh(obj.data)
        bmesh.ops.recalc_face_normals(mesh, faces=list(mesh.faces))
        mesh.to_mesh(obj.data)
        mesh.free()
        obj.data.validate(clean_customdata=False)
        obj.data.update()
    return {
        "boundary": boundary,
        "displacement_m": displacement_m,
        "transition_band_m": transition_band_m,
        "moved_vertex_count": moved,
        "elevation_before_m": {"min": minimum, "max": maximum},
    }


def main() -> None:
    args = arguments()
    source_glb = args.source_glb.resolve()
    registry_path = args.registry.resolve()
    audit_path = args.audit.resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("source_sha256") != integration.sha256(source_glb):
        raise RuntimeError("Relationship audit is not bound to the requested source GLB")
    if audit.get("registry_sha256") != integration.sha256(registry_path):
        raise RuntimeError("Relationship audit is not bound to the requested registry")
    failures = list(audit.get("failures", []))
    supported = {"column_to_cross_beam", "column_to_platform"}
    unsupported = [failure for failure in failures if failure.get("kind") not in supported]
    if unsupported:
        raise RuntimeError(f"Audit contains unsupported automatic repairs: {unsupported}")
    if not failures:
        raise RuntimeError("Audit contains no repairable failures")

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_by_id = {str(asset["id"]): asset for asset in registry}
    integration.reset_scene()
    bpy.ops.import_scene.gltf(filepath=str(source_glb), import_shading="NORMALS")
    objects = integration.mesh_objects()
    groups: dict[str, list[bpy.types.Object]] = {}
    for obj in objects:
        groups.setdefault(integration.asset_id(obj), []).append(obj)

    repairs = []
    for failure in failures:
        identifier = str(failure["source"])
        gap = float(failure["signed_gap_m"])
        if gap <= 0.0:
            raise RuntimeError(f"Automatic repair only supports positive gaps: {failure}")
        if identifier not in groups or identifier not in registry_by_id:
            raise RuntimeError(f"Repair source is missing: {identifier}")
        kind = str(failure["kind"])
        boundary = "top" if kind == "column_to_cross_beam" else "bottom"
        transition = (
            args.top_transition_band_m
            if boundary == "top"
            else args.bottom_transition_band_m
        )
        displacement = gap + args.contact_overlap_m
        repair = move_boundary(
            groups[identifier],
            boundary=boundary,
            displacement_m=displacement,
            transition_band_m=transition,
        )
        repair.update(
            {
                "kind": kind,
                "asset_id": identifier,
                "target": failure.get("target"),
                "original_signed_gap_m": gap,
                "target_signed_gap_m": -args.contact_overlap_m,
            }
        )
        repairs.append(repair)

    repaired_ids = {str(repair["asset_id"]) for repair in repairs}
    for index, asset in enumerate(registry):
        asset["index"] = index
        identifier = str(asset["id"])
        if identifier not in repaired_ids:
            continue
        asset.update(integration.aggregate_geometry_record(groups[identifier]))
        parameters = asset.setdefault("parameters", {})
        parameters["scene_relationship_repair_profile"] = args.profile
        parameters["scene_relationship_audit_sha256"] = integration.sha256(audit_path)
        parameters["explicit_normals"] = True
        asset["sourceObjects"] = [obj.name for obj in groups[identifier]]
        for obj in groups[identifier]:
            obj["v63_revision"] = args.profile
            obj["sceneRelationshipRepair"] = True

    summary = integration.geometry_summary(objects)
    if summary["invalid_or_zero_area_polygon_count"] or summary["nonfinite_vertex_count"]:
        raise RuntimeError(f"Repaired scene failed pre-export checks: {summary}")
    repair_metadata = {
        "profile": args.profile,
        "source_glb_sha256": integration.sha256(source_glb),
        "source_registry_sha256": integration.sha256(registry_path),
        "baseline_audit_sha256": integration.sha256(audit_path),
        "contact_overlap_m": args.contact_overlap_m,
        "repairs": repairs,
    }
    paths = integration.export_outputs(
        objects,
        args.output_prefix.resolve(),
        registry,
        args.profile,
        repair_metadata,
    )
    report = {
        "schema_version": "railway.scene-relationships-repair.v1",
        "status": "repaired_pending_post_audit_and_visual_review",
        "profile": args.profile,
        "source": str(source_glb),
        "repair": repair_metadata,
        "geometry": summary,
        "registry_asset_count": len(registry),
        "outputs": {
            role: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": integration.sha256(path),
            }
            for role, path in paths.items()
        },
    }
    integration.write_json(args.report.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
