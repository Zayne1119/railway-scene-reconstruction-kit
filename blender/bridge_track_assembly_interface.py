"""Bridge ballast-bed endpoint gaps without deforming either source segment."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

import bmesh
import bpy
from mathutils import Vector

BLENDER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BLENDER_DIR))
integration = importlib.import_module("integrate_track_graph")
repair = importlib.import_module("repair_track_assembly_interface")


def arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-glb", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profile", default="v9.3.1-track-assembly-bridge-01")
    return parser.parse_args(values)


def endpoint_profile(
    objects: list[bpy.types.Object],
    retained: dict[str, Any],
    direction: list[float],
) -> dict[str, dict[str, float]]:
    lateral = [-direction[1], direction[0]]
    samples: list[tuple[str, float, float, float]] = []
    for obj in objects:
        for vertex in obj.data.vertices:
            point = repair.engineering_point(obj, vertex)
            longitudinal_value = point.x * direction[0] + point.y * direction[1]
            lateral_value = point.x * lateral[0] + point.y * lateral[1]
            samples.append(
                (
                    repair.category(point, lateral_value, retained),
                    longitudinal_value,
                    lateral_value,
                    float(point.z),
                )
            )
    keys = {"top_left", "top_right", "bottom_left", "bottom_right"}
    if {sample[0] for sample in samples} != keys:
        raise RuntimeError(f"Incomplete ballast profile for {retained['asset_id']}")
    result = {}
    for key in keys:
        endpoint = max(
            (sample for sample in samples if sample[0] == key),
            key=lambda value: value[1],
        )
        result[key] = {
            "longitudinal_m": endpoint[1],
            "lateral_m": endpoint[2],
            "elevation_m": endpoint[3],
        }
    return result


def target_profile(target: dict[str, Any]) -> dict[str, dict[str, float]]:
    longitudinal = float(target["endpoint_longitudinal_m"])
    return {
        "top_left": {
            "longitudinal_m": longitudinal,
            "lateral_m": float(target["top_center_x_m"])
            - float(target["top_width_m"]) / 2.0,
            "elevation_m": float(target["top_z_m"]),
        },
        "top_right": {
            "longitudinal_m": longitudinal,
            "lateral_m": float(target["top_center_x_m"])
            + float(target["top_width_m"]) / 2.0,
            "elevation_m": float(target["top_z_m"]),
        },
        "bottom_left": {
            "longitudinal_m": longitudinal,
            "lateral_m": float(target["bottom_center_x_m"])
            - float(target["bottom_width_m"]) / 2.0,
            "elevation_m": float(target["bottom_z_m"]),
        },
        "bottom_right": {
            "longitudinal_m": longitudinal,
            "lateral_m": float(target["bottom_center_x_m"])
            + float(target["bottom_width_m"]) / 2.0,
            "elevation_m": float(target["bottom_z_m"]),
        },
    }


def remove_interface_caps(
    objects: list[bpy.types.Object],
    direction: list[float],
    interface_longitudinal_m: float,
    search_half_width_m: float = 0.8,
) -> int:
    """Remove internal transverse end caps that would shadow the splice collar."""

    lateral = [-direction[1], direction[0]]
    removed = 0
    for obj in objects:
        mesh = bmesh.new()
        mesh.from_mesh(obj.data)
        candidates = []
        for face in mesh.faces:
            if len(face.verts) < 3:
                continue
            local_points = []
            for vertex in face.verts:
                point = repair.engineering_point(obj, vertex)
                local_points.append(
                    Vector(
                        (
                            point.x * direction[0] + point.y * direction[1],
                            point.x * lateral[0] + point.y * lateral[1],
                            point.z,
                        )
                    )
                )
            center_longitudinal = sum(point.x for point in local_points) / len(
                local_points
            )
            if (
                abs(center_longitudinal - interface_longitudinal_m)
                > search_half_width_m
            ):
                continue
            normal = (local_points[1] - local_points[0]).cross(
                local_points[2] - local_points[0]
            )
            if normal.length <= 1e-9:
                continue
            if abs(normal.normalized().x) >= 0.8:
                candidates.append(face)
        if candidates:
            removed += len(candidates)
            bmesh.ops.delete(mesh, geom=candidates, context="FACES")
            mesh.to_mesh(obj.data)
            obj.data.validate(clean_customdata=False)
            obj.data.update()
        mesh.free()
    return removed


def blender_point(
    value: dict[str, float], direction: list[float], lateral: list[float]
) -> tuple[float, float, float]:
    x = value["longitudinal_m"] * direction[0] + value["lateral_m"] * lateral[0]
    chainage = (
        value["longitudinal_m"] * direction[1]
        + value["lateral_m"] * lateral[1]
    )
    return x, -value["elevation_m"], chainage


def create_bridge(
    identifier: str,
    source: dict[str, dict[str, float]],
    target: dict[str, dict[str, float]],
    direction: list[float],
    material: bpy.types.Material | None,
    template: bpy.types.Object,
) -> bpy.types.Object:
    lateral = [-direction[1], direction[0]]

    def interpolate(
        first: dict[str, float], second: dict[str, float], ratio: float
    ) -> dict[str, float]:
        return {
            key: first[key] + ratio * (second[key] - first[key])
            for key in ("longitudinal_m", "lateral_m", "elevation_m")
        }

    def gap_crossing(level: str) -> tuple[float, str]:
        left_key = f"{level}_left"
        right_key = f"{level}_right"
        left_delta = (
            target[left_key]["longitudinal_m"]
            - source[left_key]["longitudinal_m"]
        )
        right_delta = (
            target[right_key]["longitudinal_m"]
            - source[right_key]["longitudinal_m"]
        )
        side = "right" if right_delta >= left_delta else "left"
        if left_delta * right_delta < 0.0:
            ratio = -left_delta / (right_delta - left_delta)
        elif max(left_delta, right_delta) > 0.0:
            ratio = 0.0 if side == "right" else 1.0
        else:
            raise RuntimeError(f"No positive {level} gap to bridge for {identifier}")
        return max(0.0, min(1.0, ratio)), side

    def local_tuple(value: dict[str, float]) -> tuple[float, float, float]:
        return (
            value["longitudinal_m"],
            value["lateral_m"],
            value["elevation_m"],
        )

    def oriented_triangles(
        polygon: list[dict[str, float]], desired_normal: tuple[float, float, float]
    ) -> list[tuple[dict[str, float], dict[str, float], dict[str, float]]]:
        result = []
        for offset in range(1, len(polygon) - 1):
            first, second, third = polygon[0], polygon[offset], polygon[offset + 1]
            a = local_tuple(first)
            b = local_tuple(second)
            c = local_tuple(third)
            ab = [b[axis] - a[axis] for axis in range(3)]
            ac = [c[axis] - a[axis] for axis in range(3)]
            normal = (
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            )
            dot = sum(normal[axis] * desired_normal[axis] for axis in range(3))
            result.append((first, second, third) if dot >= 0.0 else (first, third, second))
        return result

    def midpoint(
        first: dict[str, float], second: dict[str, float]
    ) -> dict[str, float]:
        return {
            key: (first[key] + second[key]) / 2.0
            for key in ("longitudinal_m", "lateral_m", "elevation_m")
        }

    gap_crossing("top")
    gap_crossing("bottom")
    surface_relief_m = 0.003
    overlap_m = 0.55
    skirt_offset_m = 0.03
    start_longitudinal = min(
        value["longitudinal_m"] for value in (*source.values(), *target.values())
    ) - overlap_m
    end_longitudinal = max(
        value["longitudinal_m"] for value in (*source.values(), *target.values())
    ) + overlap_m
    middle_longitudinal = float(target["top_left"]["longitudinal_m"])
    middle_ratio = (middle_longitudinal - start_longitudinal) / (
        end_longitudinal - start_longitudinal
    )
    top_start_elevation = (
        source["top_left"]["elevation_m"]
        + source["top_right"]["elevation_m"]
    ) / 2.0 + surface_relief_m
    top_end_elevation = (
        target["top_left"]["elevation_m"]
        + target["top_right"]["elevation_m"]
    ) / 2.0 + surface_relief_m
    bottom_start_elevation = (
        source["bottom_left"]["elevation_m"]
        + source["bottom_right"]["elevation_m"]
    ) / 2.0
    bottom_end_elevation = (
        target["bottom_left"]["elevation_m"]
        + target["bottom_right"]["elevation_m"]
    ) / 2.0
    top_middle_elevation = top_start_elevation + middle_ratio * (
        top_end_elevation - top_start_elevation
    )
    bottom_middle_elevation = bottom_start_elevation + middle_ratio * (
        bottom_end_elevation - bottom_start_elevation
    )

    def station_point(
        longitudinal_m: float, lateral_m: float, elevation_m: float
    ) -> dict[str, float]:
        return {
            "longitudinal_m": longitudinal_m,
            "lateral_m": lateral_m,
            "elevation_m": elevation_m,
        }

    top_start_left = station_point(
        start_longitudinal, source["top_left"]["lateral_m"], top_start_elevation
    )
    top_start_right = station_point(
        start_longitudinal, source["top_right"]["lateral_m"], top_start_elevation
    )
    top_middle_left = station_point(
        middle_longitudinal,
        min(source["top_left"]["lateral_m"], target["top_left"]["lateral_m"])
        - skirt_offset_m,
        top_middle_elevation,
    )
    top_middle_right = station_point(
        middle_longitudinal,
        max(source["top_right"]["lateral_m"], target["top_right"]["lateral_m"])
        + skirt_offset_m,
        top_middle_elevation,
    )
    top_end_left = station_point(
        end_longitudinal, target["top_left"]["lateral_m"], top_end_elevation
    )
    top_end_right = station_point(
        end_longitudinal, target["top_right"]["lateral_m"], top_end_elevation
    )
    bottom_start_left = station_point(
        start_longitudinal,
        source["bottom_left"]["lateral_m"],
        bottom_start_elevation,
    )
    bottom_start_right = station_point(
        start_longitudinal,
        source["bottom_right"]["lateral_m"],
        bottom_start_elevation,
    )
    bottom_middle_left = station_point(
        middle_longitudinal,
        min(
            source["bottom_left"]["lateral_m"],
            target["bottom_left"]["lateral_m"],
        )
        - skirt_offset_m,
        bottom_middle_elevation,
    )
    bottom_middle_right = station_point(
        middle_longitudinal,
        max(
            source["bottom_right"]["lateral_m"],
            target["bottom_right"]["lateral_m"],
        )
        + skirt_offset_m,
        bottom_middle_elevation,
    )
    bottom_end_left = station_point(
        end_longitudinal,
        target["bottom_left"]["lateral_m"],
        bottom_end_elevation,
    )
    bottom_end_right = station_point(
        end_longitudinal,
        target["bottom_right"]["lateral_m"],
        bottom_end_elevation,
    )
    triangles = []
    stations = [
        (
            top_start_left,
            top_start_right,
            bottom_start_left,
            bottom_start_right,
        ),
        (
            top_middle_left,
            top_middle_right,
            bottom_middle_left,
            bottom_middle_right,
        ),
        (top_end_left, top_end_right, bottom_end_left, bottom_end_right),
    ]
    for first, second in pairwise(stations):
        first_top_left, first_top_right, first_bottom_left, first_bottom_right = first
        second_top_left, second_top_right, second_bottom_left, second_bottom_right = (
            second
        )
        triangles.extend(
            oriented_triangles(
                [
                    first_top_left,
                    second_top_left,
                    second_top_right,
                    first_top_right,
                ],
                (0.0, 0.0, 1.0),
            )
        )
        triangles.extend(
            oriented_triangles(
                [
                    first_bottom_left,
                    first_bottom_right,
                    second_bottom_right,
                    second_bottom_left,
                ],
                (0.0, 0.0, -1.0),
            )
        )
        triangles.extend(
            oriented_triangles(
                [
                    first_top_left,
                    first_bottom_left,
                    second_bottom_left,
                    second_top_left,
                ],
                (0.0, -1.0, 0.0),
            )
        )
        triangles.extend(
            oriented_triangles(
                [
                    first_top_right,
                    second_top_right,
                    second_bottom_right,
                    first_bottom_right,
                ],
                (0.0, 1.0, 0.0),
            )
        )
    triangles.extend(
        oriented_triangles(
            [top_start_left, top_start_right, bottom_start_right, bottom_start_left],
            (-1.0, 0.0, 0.0),
        )
    )
    triangles.extend(
        oriented_triangles(
            [top_end_left, bottom_end_left, bottom_end_right, top_end_right],
            (1.0, 0.0, 0.0),
        )
    )
    vertices = [
        blender_point(value, direction, lateral)
        for triangle in triangles
        for value in triangle
    ]
    faces = [tuple(range(index, index + 3)) for index in range(0, len(vertices), 3)]
    mesh = bpy.data.meshes.new(f"{identifier}_InterfaceBridgeMesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.validate(clean_customdata=False)
    mesh.update()
    bridge = bpy.data.objects.new(f"{identifier}_InterfaceBridge", mesh)
    bpy.context.collection.objects.link(bridge)
    bridge_material = bpy.data.materials.get("TrackAssemblyBridgeBallast")
    if bridge_material is None:
        bridge_material = bpy.data.materials.new("TrackAssemblyBridgeBallast")
        color = (
            tuple(float(value) for value in material.diffuse_color)
            if material is not None
            else (0.36, 0.34, 0.31, 1.0)
        )
        bridge_material.diffuse_color = color
        bridge_material.use_nodes = True
        principled = bridge_material.node_tree.nodes.get("Principled BSDF")
        principled.inputs["Base Color"].default_value = color
        principled.inputs["Metallic"].default_value = 0.0
        principled.inputs["Roughness"].default_value = 0.92
        bridge_material.use_backface_culling = True
    mesh.materials.append(bridge_material)
    bridge["assetId"] = identifier
    bridge["trackAssemblyBridge"] = True
    bridge["trackAssemblyBridgeProfile"] = "capped-full-width-splice-collar"
    for key in ("v63Layer", "v63_layer", "confidence"):
        if key in template:
            bridge[key] = template[key]
    return bridge


def main() -> None:
    args = arguments()
    source_glb = args.source_glb.resolve()
    registry_path = args.registry.resolve()
    audit_path = args.audit.resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("source_sha256") != integration.sha256(source_glb):
        raise RuntimeError("Audit is not bound to the requested source GLB")
    if audit.get("registry_sha256") != integration.sha256(registry_path):
        raise RuntimeError("Audit is not bound to the requested registry")
    tracks = list(audit.get("tracks", []))
    if len(tracks) != 3 or any(
        track["sleeper_interface"]["status"] != "pass" for track in tracks
    ):
        raise RuntimeError("Bridge generation requires three passing sleeper interfaces")

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_by_id = {str(asset["id"]): asset for asset in registry}
    integration.reset_scene()
    bpy.ops.import_scene.gltf(filepath=str(source_glb), import_shading="NORMALS")
    objects = integration.mesh_objects()
    groups: dict[str, list[bpy.types.Object]] = {}
    for obj in objects:
        groups.setdefault(integration.asset_id(obj), []).append(obj)

    bridges = []
    cap_removals = []
    for track in tracks:
        retained = dict(track["bed_interface"]["retained"])
        target = dict(track["bed_interface"]["new"])
        direction = repair.normalized_direction(
            list(track["sleeper_interface"]["new"]["longitudinal_direction_xy"]),
            list(track["sleeper_interface"]["retained"]["longitudinal_direction_xy"]),
        )
        identifier = str(retained["asset_id"])
        if identifier not in groups or identifier not in registry_by_id:
            raise RuntimeError(f"Bridge source is missing: {identifier}")
        source_values = endpoint_profile(groups[identifier], retained, direction)
        target_values = target_profile(target)
        template = groups[identifier][0]
        material = template.data.materials[0] if template.data.materials else None
        target_identifier = str(target["asset_id"])
        if target_identifier not in groups or target_identifier not in registry_by_id:
            raise RuntimeError(f"Bridge target is missing: {target_identifier}")
        interface_longitudinal = float(target["endpoint_longitudinal_m"])
        retained_caps = remove_interface_caps(
            groups[identifier], direction, interface_longitudinal
        )
        target_caps = remove_interface_caps(
            groups[target_identifier], direction, interface_longitudinal
        )
        cap_removals.append(
            {
                "retained_asset_id": identifier,
                "retained_cap_faces_removed": retained_caps,
                "target_asset_id": target_identifier,
                "target_cap_faces_removed": target_caps,
            }
        )
        bridge = create_bridge(
            identifier,
            source_values,
            target_values,
            direction,
            material,
            template,
        )
        groups[identifier].append(bridge)
        objects.append(bridge)
        bridges.append(
            {
                "asset_id": identifier,
                "target_asset_id": target["asset_id"],
                "source_profile": source_values,
                "target_profile": target_values,
                "object_name": bridge.name,
                "surface_relief_m": 0.003,
                "face_count": len(bridge.data.polygons),
                "triangle_count": sum(len(face.vertices) - 2 for face in bridge.data.polygons),
            }
        )

    repaired_ids = {
        identifier
        for value in bridges
        for identifier in (str(value["asset_id"]), str(value["target_asset_id"]))
    }
    for index, asset in enumerate(registry):
        asset["index"] = index
        identifier = str(asset["id"])
        if identifier not in repaired_ids:
            continue
        asset.update(integration.aggregate_geometry_record(groups[identifier]))
        parameters = asset.setdefault("parameters", {})
        parameters["track_assembly_repair_profile"] = args.profile
        parameters["track_assembly_repair_strategy"] = (
            "capped_full_width_splice_collar"
        )
        parameters["track_assembly_audit_sha256"] = integration.sha256(audit_path)
        parameters["explicit_normals"] = True
        asset["sourceObjects"] = [obj.name for obj in groups[identifier]]

    summary = integration.geometry_summary(objects)
    if summary["invalid_or_zero_area_polygon_count"] or summary["nonfinite_vertex_count"]:
        raise RuntimeError(f"Bridged scene failed pre-export checks: {summary}")
    metadata = {
        "profile": args.profile,
        "strategy": "capped_full_width_splice_collar",
        "source_glb_sha256": integration.sha256(source_glb),
        "source_registry_sha256": integration.sha256(registry_path),
        "baseline_audit_sha256": integration.sha256(audit_path),
        "scope": (
            "three full-width ballast splice collars with internal end-cap cleanup; "
            "rails and sleepers unchanged"
        ),
        "bridges": bridges,
        "cap_removals": cap_removals,
    }
    paths = integration.export_outputs(
        objects, args.output_prefix.resolve(), registry, args.profile, metadata
    )
    report = {
        "schema_version": "railway.track-assembly-interface-bridge.v1",
        "status": "bridged_pending_post_audit_and_visual_review",
        "profile": args.profile,
        "source": str(source_glb),
        "repair": metadata,
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
