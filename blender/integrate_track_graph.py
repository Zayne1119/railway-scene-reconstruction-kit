"""Replace one legacy track layer in a GLB with a continuous TrackGraph OBJ."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import bmesh
import bpy
from mathutils import Matrix, Vector

REPLACE_TYPES = {"Rail", "SleeperArray", "TrackBed"}
TYPE_MAP = {
    "rail": "Rail",
    "sleeper_group": "SleeperArray",
    "track_bed": "TrackBed",
}


def arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-glb", type=Path, required=True)
    parser.add_argument("--track-obj", type=Path, required=True)
    parser.add_argument("--track-origin", type=Path, required=True)
    parser.add_argument("--full-origin", type=Path, required=True)
    parser.add_argument("--track-registry", type=Path, required=True)
    parser.add_argument("--replace-layer", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--chainage-start-m", type=float, default=0.0)
    parser.add_argument("--chainage-end-m", type=float, default=200.0)
    parser.add_argument("--chainage-origin-local-y-m", type=float, default=100.0)
    parser.add_argument("--retained-interface-prefix", default="s200_250m.Track")
    parser.add_argument("--interface-blend-length-m", type=float, default=2.0)
    parser.add_argument("--interface-cap-gap-m", type=float, default=0.005)
    parser.add_argument("--profile", default="v9.1-trackgraph-01")
    return parser.parse_args(values)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def read_glb_json(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if data[:4] != b"glTF":
        raise ValueError(f"Not a GLB: {path}")
    json_length, json_type = struct.unpack_from("<II", data, 12)
    if json_type != 0x4E4F534A:
        raise ValueError(f"Missing JSON chunk: {path}")
    return json.loads(data[20 : 20 + json_length].decode("utf-8").rstrip("\x00 "))


def patch_glb_metadata(
    path: Path,
    registry: list[dict[str, Any]],
    profile: str,
    integration: dict[str, Any],
) -> None:
    data = path.read_bytes()
    json_length, json_type = struct.unpack_from("<II", data, 12)
    document = json.loads(data[20 : 20 + json_length].decode("utf-8").rstrip("\x00 "))
    remainder = data[20 + json_length :]
    index_by_id = {str(asset["id"]): int(asset["index"]) for asset in registry}
    extras = document.setdefault("extras", {})
    extras["assetRegistry"] = registry
    extras["ueRenderProfile"] = {
        "version": profile,
        "profile": "ue_clean_main",
        "opaqueMaterialsSingleSided": True,
        "explicitNormals": True,
        "trackGraphReplacement": True,
    }
    extras["trackGraphIntegration"] = integration
    for node in document.get("nodes", []):
        if "mesh" not in node:
            continue
        node_extras = node.setdefault("extras", {})
        asset_id = str(node_extras.get("assetId") or node.get("name") or "")
        if asset_id not in index_by_id:
            raise RuntimeError(f"Exported object is not registered: {asset_id}")
        node_extras["assetIndex"] = index_by_id[asset_id]
        node_extras["ueRenderProfile"] = "ue_clean_main"
        mesh = document["meshes"][int(node["mesh"])]
        for primitive in mesh.get("primitives", []):
            primitive_extras = primitive.setdefault("extras", {})
            primitive_extras["assetId"] = asset_id
            primitive_extras["assetIndex"] = index_by_id[asset_id]
    json_bytes = json.dumps(document, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    new_total = 12 + 8 + len(json_bytes) + len(remainder)
    header = b"glTF" + struct.pack("<II", 2, new_total)
    json_chunk = struct.pack("<II", len(json_bytes), json_type) + json_bytes
    path.write_bytes(header + json_chunk + remainder)


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0


def mesh_objects() -> list[bpy.types.Object]:
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def asset_id(obj: bpy.types.Object) -> str:
    return str(obj.get("assetId") or obj.name)


def select_only(objects: Iterable[bpy.types.Object]) -> list[bpy.types.Object]:
    result = list(objects)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in result:
        obj.select_set(True)
    if result:
        bpy.context.view_layer.objects.active = result[0]
    return result


def raw_obj_bounds(path: Path) -> tuple[Vector, Vector]:
    minimum = Vector((math.inf, math.inf, math.inf))
    maximum = Vector((-math.inf, -math.inf, -math.inf))
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("v "):
                continue
            point = Vector(tuple(float(value) for value in line.split()[1:4]))
            for axis in range(3):
                minimum[axis] = min(minimum[axis], point[axis])
                maximum[axis] = max(maximum[axis], point[axis])
    if not all(math.isfinite(value) for value in (*minimum, *maximum)):
        raise ValueError(f"OBJ contains no finite vertices: {path}")
    return minimum, maximum


def object_bounds_blender(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    return minimum, maximum


def blender_to_local_xyz(point: Vector) -> Vector:
    # The established full-scene Blender encoding is
    # (x_lateral, -z_elevation, y_chainage).
    return Vector((point.x, point.z, -point.y))


def transform_track_objects(
    objects: list[bpy.types.Object],
    raw_bounds: tuple[Vector, Vector],
    track_origin: Vector,
    full_origin: Vector,
    chainage_origin_local_y_m: float,
) -> dict[str, Any]:
    imported_min, imported_max = object_bounds_blender(objects)
    raw_min, raw_max = raw_bounds
    expected_imported_min = Vector((-raw_max.x, -raw_max.y, raw_min.z))
    expected_imported_max = Vector((-raw_min.x, -raw_min.y, raw_max.z))
    mapping_error = max(
        max(abs(imported_min[axis] - expected_imported_min[axis]) for axis in range(3)),
        max(abs(imported_max[axis] - expected_imported_max[axis]) for axis in range(3)),
    )
    if mapping_error > 1e-3:
        raise RuntimeError(f"Unexpected Blender OBJ axis mapping; bounds error={mapping_error}")

    lateral_offset = float(track_origin.x - full_origin.x)
    vertical_offset = float(track_origin.z - full_origin.z)
    global_y_offset = float(track_origin.y - full_origin.y)
    chainage_alignment_correction = chainage_origin_local_y_m - global_y_offset
    signed_volumes_before: dict[str, float] = {}
    signed_volumes_after: dict[str, float] = {}
    for obj in objects:
        matrix = obj.matrix_world.copy()
        for vertex in obj.data.vertices:
            imported = matrix @ vertex.co
            raw = Vector((-imported.x, -imported.y, imported.z))
            local_xyz = Vector(
                (
                    raw.x + lateral_offset,
                    raw.y + chainage_origin_local_y_m,
                    raw.z + vertical_offset,
                )
            )
            # Match the established full-scene encoding exactly.  Using the
            # opposite signs here mirrors the inserted track in both
            # chainage and elevation relative to all retained assets.
            vertex.co = Vector((local_xyz.x, -local_xyz.z, local_xyz.y))
        obj.matrix_world = Matrix.Identity(4)
        mesh = bmesh.new()
        mesh.from_mesh(obj.data)
        signed_volumes_before[obj.name] = float(mesh.calc_volume(signed=True))
        bmesh.ops.recalc_face_normals(mesh, faces=list(mesh.faces))
        signed_volumes_after[obj.name] = float(mesh.calc_volume(signed=True))
        mesh.to_mesh(obj.data)
        mesh.free()
        obj["faceOrientationRepair"] = "bmesh_recalculate_outside"
        obj.data.validate(clean_customdata=False)
        obj.data.update()
    transformed_min, transformed_max = object_bounds_blender(objects)
    local_corners = [
        blender_to_local_xyz(Vector((x, y, z)))
        for x in (transformed_min.x, transformed_max.x)
        for y in (transformed_min.y, transformed_max.y)
        for z in (transformed_min.z, transformed_max.z)
    ]
    local_min = Vector(tuple(min(point[axis] for point in local_corners) for axis in range(3)))
    local_max = Vector(tuple(max(point[axis] for point in local_corners) for axis in range(3)))
    return {
        "obj_import_axis_recovery": "raw=(-blender_x,-blender_y,blender_z)",
        "target_blender_axis_encoding": "(x_lateral,-z_elevation,y_chainage)",
        "obj_import_mapping_bounds_error_m": mapping_error,
        "lateral_global_origin_offset_m": lateral_offset,
        "vertical_global_origin_offset_m": vertical_offset,
        "global_y_origin_offset_m": global_y_offset,
        "chainage_origin_local_y_m": chainage_origin_local_y_m,
        "chainage_alignment_correction_from_global_origin_m": chainage_alignment_correction,
        "face_orientation_recalculated_object_count": len(objects),
        "negative_signed_volume_count_before": sum(
            value < 0.0 for value in signed_volumes_before.values()
        ),
        "negative_signed_volume_count_after": sum(
            value < 0.0 for value in signed_volumes_after.values()
        ),
        "signed_volume_m3_before": signed_volumes_before,
        "signed_volume_m3_after": signed_volumes_after,
        "transformed_bounds_local_xyz": {
            "min": list(local_min),
            "max": list(local_max),
        },
    }


def repair_negative_closed_assets(
    objects: list[bpy.types.Object],
    registry_by_id: dict[str, dict[str, Any]],
    asset_types: set[str],
) -> list[dict[str, Any]]:
    """Reorient closed retained assets whose signed volume is negative."""

    repairs: list[dict[str, Any]] = []
    for obj in objects:
        identifier = asset_id(obj)
        record = registry_by_id.get(identifier)
        if record is None or str(record.get("type")) not in asset_types:
            continue
        mesh = bmesh.new()
        mesh.from_mesh(obj.data)
        before = float(mesh.calc_volume(signed=True))
        if before >= -1e-9:
            mesh.free()
            continue
        bmesh.ops.recalc_face_normals(mesh, faces=list(mesh.faces))
        after = float(mesh.calc_volume(signed=True))
        if after <= 0.0:
            mesh.free()
            raise RuntimeError(
                f"Retained asset orientation repair failed for {identifier}: {after}"
            )
        mesh.to_mesh(obj.data)
        mesh.free()
        obj.data.validate(clean_customdata=False)
        obj.data.update()
        obj["faceOrientationRepair"] = "bmesh_recalculate_outside"
        repairs.append(
            {
                "asset_id": identifier,
                "type": str(record.get("type")),
                "signed_volume_m3_before": before,
                "signed_volume_m3_after": after,
            }
        )
    return repairs


def rail_endpoint(
    obj: bpy.types.Object, interface_chainage_m: float
) -> dict[str, float]:
    points = [
        blender_to_local_xyz(obj.matrix_world @ vertex.co)
        for vertex in obj.data.vertices
    ]
    minimum = min(point.y for point in points)
    maximum = max(point.y for point in points)
    endpoint = min((minimum, maximum), key=lambda value: abs(value - interface_chainage_m))
    distances = sorted(abs(point.y - endpoint) for point in points)
    tolerance = max(0.02, distances[min(len(distances) - 1, 31)] + 1e-6)
    section = [point for point in points if abs(point.y - endpoint) <= tolerance]
    return {
        "chainage_m": float(endpoint),
        "center_x_m": float(sum(point.x for point in section) / len(section)),
        "rail_top_z_m": float(max(point.z for point in section)),
    }


def align_track_interface(
    new_objects: list[bpy.types.Object],
    retained_objects: list[bpy.types.Object],
    interface_chainage_m: float,
    retained_prefix: str,
    blend_length_m: float,
    cap_gap_m: float,
) -> list[dict[str, Any]]:
    if blend_length_m <= 0.0 or cap_gap_m < 0.0:
        raise ValueError("Interface blend length must be positive and cap gap non-negative")
    new_rails = [obj for obj in new_objects if "-RAIL-" in obj.name]
    retained_rails = [
        obj
        for obj in retained_objects
        if asset_id(obj).startswith(retained_prefix) and "Rail" in asset_id(obj)
    ]
    new_records = sorted(
        ((obj, rail_endpoint(obj, interface_chainage_m)) for obj in new_rails),
        key=lambda item: item[1]["center_x_m"],
    )
    retained_records = sorted(
        ((obj, rail_endpoint(obj, interface_chainage_m)) for obj in retained_rails),
        key=lambda item: item[1]["center_x_m"],
    )
    if len(new_records) != 6 or len(retained_records) != 6:
        raise RuntimeError(
            "Interface alignment requires six new and six retained rails, "
            f"got {len(new_records)} and {len(retained_records)}"
        )

    results: list[dict[str, Any]] = []
    for (new_obj, before), (retained_obj, target) in zip(
        new_records, retained_records, strict=True
    ):
        target_chainage = target["chainage_m"] + cap_gap_m
        chainage_shift = target_chainage - before["chainage_m"]
        lateral_shift = target["center_x_m"] - before["center_x_m"]
        vertical_shift = target["rail_top_z_m"] - before["rail_top_z_m"]
        endpoint = before["chainage_m"]
        for vertex in new_obj.data.vertices:
            point = vertex.co
            distance_from_endpoint = float(point.z - endpoint)
            weight = max(0.0, min(1.0, 1.0 - distance_from_endpoint / blend_length_m))
            if weight <= 0.0:
                continue
            point.x += lateral_shift * weight
            point.y -= vertical_shift * weight
            point.z += chainage_shift * weight
        mesh = bmesh.new()
        mesh.from_mesh(new_obj.data)
        bmesh.ops.recalc_face_normals(mesh, faces=list(mesh.faces))
        mesh.to_mesh(new_obj.data)
        mesh.free()
        new_obj.data.validate(clean_customdata=False)
        new_obj.data.update()
        new_obj["interfaceAlignment"] = "two_metre_taper_with_cap_gap"
        after = rail_endpoint(new_obj, interface_chainage_m)
        results.append(
            {
                "new_asset_id": asset_id(new_obj),
                "retained_asset_id": asset_id(retained_obj),
                "before": before,
                "target": target,
                "applied_shift_m": {
                    "chainage": chainage_shift,
                    "lateral": lateral_shift,
                    "vertical": vertical_shift,
                },
                "after": after,
            }
        )
    return results


def chainage_bounds(obj: bpy.types.Object) -> tuple[float, float]:
    values = [
        blender_to_local_xyz(obj.matrix_world @ vertex.co).y
        for vertex in obj.data.vertices
    ]
    return float(min(values)), float(max(values))


def object_center_x(obj: bpy.types.Object) -> float:
    values = [
        blender_to_local_xyz(obj.matrix_world @ vertex.co).x
        for vertex in obj.data.vertices
    ]
    return float(sum(values) / len(values))


def align_retained_sleeper_phase(
    new_objects: list[bpy.types.Object],
    retained_objects: list[bpy.types.Object],
    retained_prefix: str,
    spacing_m: float = 0.6,
) -> dict[str, Any]:
    new_sleepers = sorted(
        (obj for obj in new_objects if obj.name.endswith("-SLEEPERS")),
        key=object_center_x,
    )
    groups: dict[str, list[bpy.types.Object]] = {}
    for obj in retained_objects:
        identifier = asset_id(obj)
        if identifier.startswith(retained_prefix) and "Sleepers" in identifier:
            groups.setdefault(identifier, []).append(obj)
    retained_groups = sorted(
        groups.items(),
        key=lambda item: sum(object_center_x(obj) for obj in item[1]) / len(item[1]),
    )
    if len(new_sleepers) != 3 or len(retained_groups) != 3:
        raise RuntimeError(
            "Sleeper phase alignment requires three new and three retained arrays, "
            f"got {len(new_sleepers)} and {len(retained_groups)}"
        )

    removed_names: list[str] = []
    results: list[dict[str, Any]] = []
    for new_obj, (retained_id, retained_group) in zip(
        new_sleepers, retained_groups, strict=True
    ):
        new_minimum, _ = chainage_bounds(new_obj)
        endpoint_points = [
            blender_to_local_xyz(new_obj.matrix_world @ vertex.co).y
            for vertex in new_obj.data.vertices
            if blender_to_local_xyz(new_obj.matrix_world @ vertex.co).y
            <= new_minimum + 0.4
        ]
        new_last_center = float(sum(endpoint_points) / len(endpoint_points))

        valid: list[tuple[bpy.types.Object, float]] = []
        fragments = []
        for obj in retained_group:
            minimum, maximum = chainage_bounds(obj)
            if maximum - minimum < 0.10 or len(obj.data.polygons) < 6:
                fragments.append(obj)
                continue
            valid.append((obj, (minimum + maximum) / 2.0))
        if not valid:
            raise RuntimeError(f"No full retained sleepers remain for {retained_id}")
        fragment_names = [fragment.name for fragment in fragments]
        for fragment in fragments:
            removed_names.append(fragment.name)
            bpy.data.objects.remove(fragment, do_unlink=True)

        valid.sort(key=lambda item: item[1], reverse=True)
        current_first_center = valid[0][1]
        segment_end_center = valid[-1][1]
        desired_first_center = new_last_center - spacing_m
        initial_shift = desired_first_center - current_first_center
        denominator = max(current_first_center - segment_end_center, 1e-6)
        maximum_applied_shift = 0.0
        for obj, center in valid:
            fade = max(0.0, min(1.0, (center - segment_end_center) / denominator))
            shift = initial_shift * fade
            maximum_applied_shift = max(maximum_applied_shift, abs(shift))
            inverse = obj.matrix_world.inverted()
            for vertex in obj.data.vertices:
                world = obj.matrix_world @ vertex.co
                world.z += shift
                vertex.co = inverse @ world
            obj.data.update()
            obj["interfaceSleeperPhaseRepair"] = "linear_fade_to_segment_end"
        results.append(
            {
                "new_asset_id": asset_id(new_obj),
                "retained_asset_id": retained_id,
                "new_last_center_chainage_m": new_last_center,
                "retained_first_center_before_m": current_first_center,
                "retained_first_center_after_m": current_first_center + initial_shift,
                "desired_spacing_m": spacing_m,
                "initial_shift_m": initial_shift,
                "maximum_applied_shift_m": maximum_applied_shift,
                "fragment_objects_removed": fragment_names,
                "full_sleeper_object_count": len(valid),
            }
        )
    return {
        "status": "applied",
        "spacing_m": spacing_m,
        "removed_object_names": removed_names,
        "removed_fragment_count": len(removed_names),
        "tracks": results,
    }


def aggregate_geometry_record(objects: list[bpy.types.Object]) -> dict[str, Any]:
    points = [
        blender_to_local_xyz(obj.matrix_world @ vertex.co)
        for obj in objects
        for vertex in obj.data.vertices
    ]
    minimum = [min(point[axis] for point in points) for axis in range(3)]
    maximum = [max(point[axis] for point in points) for axis in range(3)]
    triangles = sum(
        max(len(polygon.vertices) - 2, 0)
        for obj in objects
        for polygon in obj.data.polygons
    )
    return {
        "bounds": {"min": minimum, "max": maximum},
        "center": [(minimum[axis] + maximum[axis]) / 2.0 for axis in range(3)],
        "triangles": triangles,
    }


def object_geometry_record(obj: bpy.types.Object) -> dict[str, Any]:
    points = [blender_to_local_xyz(obj.matrix_world @ vertex.co) for vertex in obj.data.vertices]
    minimum = [min(point[axis] for point in points) for axis in range(3)]
    maximum = [max(point[axis] for point in points) for axis in range(3)]
    triangles = sum(max(len(polygon.vertices) - 2, 0) for polygon in obj.data.polygons)
    return {
        "bounds": {"min": minimum, "max": maximum},
        "center": [(minimum[axis] + maximum[axis]) / 2.0 for axis in range(3)],
        "triangles": triangles,
    }


def convert_track_registry(
    registry_path: Path,
    objects: list[bpy.types.Object],
    layer: str,
    profile: str,
    start_m: float,
    end_m: float,
) -> list[dict[str, Any]]:
    source = read_json(registry_path)
    source_by_id = {str(asset["id"]): asset for asset in source.get("assets", [])}
    result: list[dict[str, Any]] = []
    for obj in objects:
        identifier = obj.name
        if identifier not in source_by_id:
            raise RuntimeError(f"Track OBJ node is absent from TrackGraph registry: {identifier}")
        asset = source_by_id[identifier]
        parameters = json.loads(json.dumps(asset.get("parameters", {})))
        parameters.update(
            {
                "trackgraph_registry_status": asset.get("status"),
                "trackgraph_evidence_level": asset.get("evidence_level"),
                "ue_render_profile": profile,
                "engine_object_name": identifier,
                "explicit_normals": True,
            }
        )
        record = {
            "id": identifier,
            "type": TYPE_MAP[str(asset["type"])],
            "name": identifier,
            "mode": f"trackgraph_{asset.get('evidence_level', 'unknown')}",
            "confidence": float(asset.get("confidence", 0.0)),
            "parentId": "Corridor_s000_200m",
            **object_geometry_record(obj),
            "parameters": parameters,
            "source": asset.get("sources", []),
            "sourceObjects": [identifier],
            "layers": [layer],
            "professional": "01_track",
            "evidenceScope": asset.get("evidence_level"),
            "precisionGrade": (
                "internal_fit_only"
                if asset.get("evidence_level") == "observed"
                else "display_grade"
            ),
            "integrationLayer": layer,
            "chainageRangeM": [start_m, end_m],
            "limitations": asset.get("limitations", []),
        }
        result.append(record)
        obj["assetId"] = identifier
        obj["v63Layer"] = layer
        obj["v63_revision"] = profile
        obj["ueRenderProfile"] = "ue_clean_main"
        obj["trackGraphEvidenceLevel"] = str(asset.get("evidence_level", "unknown"))
        for material in obj.data.materials:
            if material is not None:
                material.use_backface_culling = True
    return result


def geometry_summary(objects: list[bpy.types.Object]) -> dict[str, Any]:
    vertices = polygons = triangles = invalid = nonfinite = 0
    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    for obj in objects:
        vertices += len(obj.data.vertices)
        polygons += len(obj.data.polygons)
        for polygon in obj.data.polygons:
            triangles += max(len(polygon.vertices) - 2, 0)
            if len(polygon.vertices) < 3 or polygon.area <= 1e-12:
                invalid += 1
        for vertex in obj.data.vertices:
            point = obj.matrix_world @ vertex.co
            if not all(math.isfinite(value) for value in point):
                nonfinite += 1
                continue
            for axis in range(3):
                minimum[axis] = min(minimum[axis], float(point[axis]))
                maximum[axis] = max(maximum[axis], float(point[axis]))
    return {
        "object_count": len(objects),
        "asset_id_count": len({asset_id(obj) for obj in objects}),
        "vertex_count": vertices,
        "polygon_count": polygons,
        "triangle_count": triangles,
        "invalid_or_zero_area_polygon_count": invalid,
        "nonfinite_vertex_count": nonfinite,
        "bounds_min_blender": minimum,
        "bounds_max_blender": maximum,
    }


def export_outputs(
    objects: list[bpy.types.Object],
    prefix: Path,
    registry: list[dict[str, Any]],
    profile: str,
    integration: dict[str, Any],
) -> dict[str, Path]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    # ``Path.with_suffix`` treats the release marker in names such as ``v9.1``
    # as an existing suffix and silently truncates the requested basename to
    # ``v9``.  The CLI accepts an extension-free output *prefix*, so append the
    # export suffixes explicitly and preserve dotted release identifiers.
    glb = Path(f"{prefix}.glb")
    fbx = Path(f"{prefix}.fbx")
    obj = Path(f"{prefix}.obj")
    registry_path = prefix.with_name(prefix.name + "_registry.json")
    select_only(objects)
    result = bpy.ops.export_scene.gltf(
        filepath=str(glb),
        check_existing=False,
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_materials="EXPORT",
        export_normals=True,
        export_yup=True,
        export_extras=True,
    )
    if "FINISHED" not in result:
        raise RuntimeError(f"GLB export failed: {result}")
    patch_glb_metadata(glb, registry, profile, integration)
    select_only(objects)
    result = bpy.ops.export_scene.fbx(
        filepath=str(fbx),
        check_existing=False,
        use_selection=True,
        global_scale=1.0,
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_UNITS",
        use_space_transform=True,
        bake_space_transform=False,
        object_types={"MESH"},
        use_mesh_modifiers=True,
        use_mesh_modifiers_render=True,
        mesh_smooth_type="FACE",
        use_tspace=False,
        use_triangles=True,
        use_custom_props=True,
        add_leaf_bones=False,
        bake_anim=False,
        path_mode="AUTO",
        embed_textures=False,
        use_metadata=True,
        axis_forward="-Y",
        axis_up="Z",
    )
    if "FINISHED" not in result:
        raise RuntimeError(f"FBX export failed: {result}")
    select_only(objects)
    result = bpy.ops.wm.obj_export(
        filepath=str(obj),
        check_existing=False,
        export_animation=False,
        forward_axis="NEGATIVE_Y",
        up_axis="Z",
        global_scale=1.0,
        apply_modifiers=True,
        export_eval_mode="DAG_EVAL_RENDER",
        export_selected_objects=True,
        export_uv=True,
        export_normals=True,
        export_colors=False,
        export_materials=True,
        export_pbr_extensions=False,
        path_mode="AUTO",
        export_triangulated_mesh=True,
        export_object_groups=True,
        export_material_groups=True,
        export_smooth_groups=True,
    )
    if "FINISHED" not in result:
        raise RuntimeError(f"OBJ export failed: {result}")
    write_json(registry_path, registry)
    return {"glb": glb, "fbx": fbx, "obj": obj, "registry": registry_path}


def main() -> None:
    args = arguments()
    source_glb = args.source_glb.resolve()
    track_obj = args.track_obj.resolve()
    source_document = read_glb_json(source_glb)
    source_registry = list(source_document.get("extras", {}).get("assetRegistry", []))
    if not source_registry:
        raise RuntimeError("Source GLB has no embedded asset registry")
    replacement_assets = [
        asset
        for asset in source_registry
        if str(asset.get("type")) in REPLACE_TYPES
        and str(asset.get("integrationLayer")) == args.replace_layer
    ]
    replacement_ids = {str(asset["id"]) for asset in replacement_assets}
    if not replacement_ids:
        raise RuntimeError("No replaceable source track assets matched the requested layer")

    reset_scene()
    bpy.ops.import_scene.gltf(filepath=str(source_glb))
    source_objects = mesh_objects()
    remove_objects = [obj for obj in source_objects if asset_id(obj) in replacement_ids]
    if {asset_id(obj) for obj in remove_objects} != replacement_ids:
        missing = sorted(replacement_ids - {asset_id(obj) for obj in remove_objects})
        raise RuntimeError(f"Registry replacement assets have no mesh objects: {missing}")
    for obj in remove_objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    preserved_objects = mesh_objects()
    source_registry_by_id = {str(asset["id"]): asset for asset in source_registry}
    retained_orientation_repairs = repair_negative_closed_assets(
        preserved_objects,
        source_registry_by_id,
        {"Rail"},
    )

    before_import = set(bpy.context.scene.objects)
    bpy.ops.wm.obj_import(filepath=str(track_obj), forward_axis="NEGATIVE_Y", up_axis="Z")
    track_objects = [
        obj
        for obj in bpy.context.scene.objects
        if obj not in before_import and obj.type == "MESH"
    ]
    if len(track_objects) != 12:
        raise RuntimeError(f"Expected 12 TrackGraph mesh objects, got {len(track_objects)}")
    transform = transform_track_objects(
        track_objects,
        raw_obj_bounds(track_obj),
        Vector(read_json(args.track_origin)["origin_xyz"]),
        Vector(read_json(args.full_origin)["global_origin"]),
        args.chainage_origin_local_y_m,
    )
    interface_chainage_m = args.chainage_origin_local_y_m - (
        args.chainage_end_m - args.chainage_start_m
    )
    interface_alignment = align_track_interface(
        track_objects,
        preserved_objects,
        interface_chainage_m,
        args.retained_interface_prefix,
        args.interface_blend_length_m,
        args.interface_cap_gap_m,
    )
    sleeper_phase_alignment = align_retained_sleeper_phase(
        track_objects,
        preserved_objects,
        args.retained_interface_prefix,
    )
    track_object_set = set(track_objects)
    preserved_objects = [obj for obj in mesh_objects() if obj not in track_object_set]
    layer = f"trackgraph_continuous_s000_200m_{args.profile}"
    new_assets = convert_track_registry(
        args.track_registry.resolve(),
        track_objects,
        layer,
        args.profile,
        args.chainage_start_m,
        args.chainage_end_m,
    )
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
    repaired_retained_ids = {
        str(repair["asset_id"]) for repair in retained_orientation_repairs
    }
    for asset in combined_registry:
        if str(asset["id"]) not in repaired_retained_ids:
            continue
        parameters = asset.setdefault("parameters", {})
        if isinstance(parameters, dict):
            parameters["face_orientation_repair"] = "bmesh_recalculate_outside"
    aligned_ids = {str(item["new_asset_id"]) for item in interface_alignment}
    for asset in combined_registry:
        if str(asset["id"]) not in aligned_ids:
            continue
        parameters = asset.setdefault("parameters", {})
        if isinstance(parameters, dict):
            parameters["interface_alignment"] = "two_metre_taper_with_cap_gap"
            parameters["interface_cap_gap_m"] = args.interface_cap_gap_m
    repaired_sleeper_ids = {
        str(item["retained_asset_id"])
        for item in sleeper_phase_alignment["tracks"]
    }
    for asset in combined_registry:
        identifier = str(asset["id"])
        if identifier not in repaired_sleeper_ids:
            continue
        members = [obj for obj in combined_objects if asset_id(obj) == identifier]
        asset.update(aggregate_geometry_record(members))
        asset["sourceObjects"] = [obj.name for obj in members]
        parameters = asset.setdefault("parameters", {})
        if isinstance(parameters, dict):
            parameters["interface_sleeper_phase_repair"] = (
                "linear_fade_to_segment_end"
            )
    object_ids = {asset_id(obj) for obj in combined_objects}
    registry_ids = {str(asset["id"]) for asset in combined_registry}
    if object_ids != registry_ids:
        raise RuntimeError(
            f"Object/registry mismatch: missing={sorted(object_ids-registry_ids)[:10]}, "
            f"unused={sorted(registry_ids-object_ids)[:10]}"
        )
    index_by_id = {str(asset["id"]): int(asset["index"]) for asset in combined_registry}
    for obj in combined_objects:
        obj["assetId"] = asset_id(obj)
        obj["assetIndex"] = index_by_id[asset_id(obj)]
        obj["ueRenderProfile"] = "ue_clean_main"
        obj["v63_revision"] = args.profile

    summary = geometry_summary(combined_objects)
    if summary["invalid_or_zero_area_polygon_count"] or summary["nonfinite_vertex_count"]:
        raise RuntimeError(f"Integrated scene failed pre-export geometry checks: {summary}")
    integration = {
        "profile": args.profile,
        "source_glb_sha256": sha256(source_glb),
        "track_obj_sha256": sha256(track_obj),
        "replace_layer": args.replace_layer,
        "removed_asset_count": len(replacement_ids),
        "removed_object_count": len(remove_objects),
        "added_asset_count": len(new_assets),
        "added_object_count": len(track_objects),
        "chainage_range_m": [args.chainage_start_m, args.chainage_end_m],
        "transform": transform,
        "interface_alignment": {
            "interface_chainage_m": interface_chainage_m,
            "retained_prefix": args.retained_interface_prefix,
            "blend_length_m": args.interface_blend_length_m,
            "cap_gap_m": args.interface_cap_gap_m,
            "aligned_rail_count": len(interface_alignment),
            "rails": interface_alignment,
        },
        "sleeper_phase_alignment": sleeper_phase_alignment,
        "retained_orientation_repair": {
            "asset_types": ["Rail"],
            "repaired_object_count": len(retained_orientation_repairs),
            "repairs": retained_orientation_repairs,
        },
    }
    paths = export_outputs(
        combined_objects,
        args.output_prefix.resolve(),
        combined_registry,
        args.profile,
        integration,
    )
    report = {
        "schema_version": "railway.trackgraph-full-scene-integration.v1",
        "status": "automatic_checks_passed_visual_review_required",
        "profile": args.profile,
        "source": str(source_glb),
        "integration": integration,
        "geometry": summary,
        "registry_asset_count": len(combined_registry),
        "removed_asset_ids": sorted(replacement_ids),
        "added_asset_ids": sorted(str(asset["id"]) for asset in new_assets),
        "outputs": {
            role: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for role, path in paths.items()
        },
    }
    write_json(args.report.resolve(), report)
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
