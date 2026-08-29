"""Close ballast-bed endpoint seams without changing rails or sleepers."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import bmesh
import bpy
from mathutils import Vector

BLENDER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BLENDER_DIR))
integration = importlib.import_module("integrate_track_graph")
assembly_audit = importlib.import_module("audit_track_assembly_interface")


def arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-glb", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profile", default="v9.3-track-assembly-01")
    parser.add_argument("--transition-band-m", type=float, default=2.0)
    return parser.parse_args(values)


def normalized_direction(first: list[float], second: list[float]) -> list[float]:
    value = [first[axis] + second[axis] for axis in range(2)]
    length = math.hypot(value[0], value[1])
    if length <= 1e-9:
        raise RuntimeError("Cannot recover a common track direction")
    return [value[0] / length, value[1] / length]


def engineering_point(obj: bpy.types.Object, vertex: bpy.types.MeshVertex) -> Vector:
    point = obj.matrix_world @ vertex.co
    return Vector((point.x, point.z, -point.y))


def category(
    point: Vector,
    lateral_value: float,
    retained: dict[str, Any],
) -> str:
    midpoint_z = (float(retained["top_z_m"]) + float(retained["bottom_z_m"])) / 2.0
    if point.z > midpoint_z:
        level = "top"
        center = float(retained["top_center_x_m"])
    else:
        level = "bottom"
        center = float(retained["bottom_center_x_m"])
    side = "left" if lateral_value < center else "right"
    return f"{level}_{side}"


def repair_bed_endpoint(
    objects: list[bpy.types.Object],
    retained: dict[str, Any],
    target: dict[str, Any],
    direction: list[float],
    transition_band_m: float,
) -> dict[str, Any]:
    lateral = [-direction[1], direction[0]]
    samples: list[tuple[str, float, float, float]] = []
    for obj in objects:
        for vertex in obj.data.vertices:
            point = engineering_point(obj, vertex)
            longitudinal_value = point.x * direction[0] + point.y * direction[1]
            lateral_value = point.x * lateral[0] + point.y * lateral[1]
            samples.append(
                (
                    category(point, lateral_value, retained),
                    longitudinal_value,
                    lateral_value,
                    float(point.z),
                )
            )
    keys = {"top_left", "top_right", "bottom_left", "bottom_right"}
    if {sample[0] for sample in samples} != keys:
        raise RuntimeError(f"Incomplete ballast profile categories for {retained['asset_id']}")

    source: dict[str, dict[str, float]] = {}
    for key in keys:
        endpoint = max((sample for sample in samples if sample[0] == key), key=lambda value: value[1])
        source[key] = {
            "longitudinal_m": endpoint[1],
            "lateral_m": endpoint[2],
            "elevation_m": endpoint[3],
        }
    target_values = {
        "top_left": {
            "lateral_m": float(target["top_center_x_m"])
            - float(target["top_width_m"]) / 2.0,
            "elevation_m": float(target["top_z_m"]),
        },
        "top_right": {
            "lateral_m": float(target["top_center_x_m"])
            + float(target["top_width_m"]) / 2.0,
            "elevation_m": float(target["top_z_m"]),
        },
        "bottom_left": {
            "lateral_m": float(target["bottom_center_x_m"])
            - float(target["bottom_width_m"]) / 2.0,
            "elevation_m": float(target["bottom_z_m"]),
        },
        "bottom_right": {
            "lateral_m": float(target["bottom_center_x_m"])
            + float(target["bottom_width_m"]) / 2.0,
            "elevation_m": float(target["bottom_z_m"]),
        },
    }
    target_longitudinal = float(target["endpoint_longitudinal_m"])
    moved = 0
    maximum_displacement = 0.0
    for obj in objects:
        inverse = obj.matrix_world.inverted()
        for vertex in obj.data.vertices:
            point = engineering_point(obj, vertex)
            original = point.copy()
            longitudinal_value = point.x * direction[0] + point.y * direction[1]
            lateral_value = point.x * lateral[0] + point.y * lateral[1]
            key = category(point, lateral_value, retained)
            distance_inward = source[key]["longitudinal_m"] - longitudinal_value
            if distance_inward < -1e-4 or distance_inward > transition_band_m:
                continue
            ratio = 1.0 - max(0.0, distance_inward) / transition_band_m
            weight = ratio * ratio * (3.0 - 2.0 * ratio)
            longitudinal_value += weight * (
                target_longitudinal - source[key]["longitudinal_m"]
            )
            lateral_value += weight * (
                target_values[key]["lateral_m"] - source[key]["lateral_m"]
            )
            point.z += weight * (
                target_values[key]["elevation_m"] - source[key]["elevation_m"]
            )
            point.x = longitudinal_value * direction[0] + lateral_value * lateral[0]
            point.y = longitudinal_value * direction[1] + lateral_value * lateral[1]
            world = Vector((point.x, -point.z, point.y))
            vertex.co = inverse @ world
            maximum_displacement = max(maximum_displacement, (point - original).length)
            moved += 1
        mesh = bmesh.new()
        mesh.from_mesh(obj.data)
        bmesh.ops.recalc_face_normals(mesh, faces=list(mesh.faces))
        mesh.to_mesh(obj.data)
        mesh.free()
        obj.data.validate(clean_customdata=False)
        obj.data.update()
    return {
        "asset_id": retained["asset_id"],
        "target_asset_id": target["asset_id"],
        "transition_band_m": transition_band_m,
        "moved_vertex_count": moved,
        "maximum_vertex_displacement_m": maximum_displacement,
        "source_profile": source,
        "target_profile": target_values,
        "target_longitudinal_m": target_longitudinal,
    }


def main() -> None:
    args = arguments()
    source_glb = args.source_glb.resolve()
    registry_path = args.registry.resolve()
    audit_path = args.audit.resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("source_sha256") != integration.sha256(source_glb):
        raise RuntimeError("Track assembly audit is not bound to the requested source GLB")
    if audit.get("registry_sha256") != integration.sha256(registry_path):
        raise RuntimeError("Track assembly audit is not bound to the requested registry")
    tracks = list(audit.get("tracks", []))
    if len(tracks) != 3:
        raise RuntimeError("Expected three audited track interfaces")
    if any(track["sleeper_interface"]["status"] != "pass" for track in tracks):
        raise RuntimeError("Automatic bed repair requires all sleeper interfaces to pass")
    if any(track["bed_interface"]["status"] != "fail" for track in tracks):
        raise RuntimeError("Expected all three ballast-bed interfaces to require repair")

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_by_id = {str(asset["id"]): asset for asset in registry}
    integration.reset_scene()
    bpy.ops.import_scene.gltf(filepath=str(source_glb), import_shading="NORMALS")
    objects = integration.mesh_objects()
    groups: dict[str, list[bpy.types.Object]] = {}
    for obj in objects:
        groups.setdefault(integration.asset_id(obj), []).append(obj)

    repairs = []
    for track in tracks:
        retained = dict(track["bed_interface"]["retained"])
        target = dict(track["bed_interface"]["new"])
        direction = normalized_direction(
            list(track["sleeper_interface"]["new"]["longitudinal_direction_xy"]),
            list(track["sleeper_interface"]["retained"]["longitudinal_direction_xy"]),
        )
        identifier = str(retained["asset_id"])
        if identifier not in groups or identifier not in registry_by_id:
            raise RuntimeError(f"Repair source is missing: {identifier}")
        repairs.append(
            repair_bed_endpoint(
                groups[identifier],
                retained,
                target,
                direction,
                args.transition_band_m,
            )
        )

    repaired_ids = {str(repair["asset_id"]) for repair in repairs}
    for index, asset in enumerate(registry):
        asset["index"] = index
        identifier = str(asset["id"])
        if identifier not in repaired_ids:
            continue
        asset.update(integration.aggregate_geometry_record(groups[identifier]))
        parameters = asset.setdefault("parameters", {})
        parameters["track_assembly_repair_profile"] = args.profile
        parameters["track_assembly_audit_sha256"] = integration.sha256(audit_path)
        parameters["explicit_normals"] = True
        asset["sourceObjects"] = [obj.name for obj in groups[identifier]]
        for obj in groups[identifier]:
            obj["trackAssemblyRepair"] = True
            obj["trackAssemblyRepairProfile"] = args.profile

    summary = integration.geometry_summary(objects)
    if summary["invalid_or_zero_area_polygon_count"] or summary["nonfinite_vertex_count"]:
        raise RuntimeError(f"Repaired scene failed pre-export checks: {summary}")
    repair_metadata = {
        "profile": args.profile,
        "source_glb_sha256": integration.sha256(source_glb),
        "source_registry_sha256": integration.sha256(registry_path),
        "baseline_audit_sha256": integration.sha256(audit_path),
        "scope": "three retained ballast-bed endpoints only; rails and sleepers unchanged",
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
        "schema_version": "railway.track-assembly-interface-repair.v1",
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
