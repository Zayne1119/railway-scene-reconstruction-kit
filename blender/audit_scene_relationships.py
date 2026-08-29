"""Audit structural contacts directly from an exported complete-scene GLB.

Run with Blender:

    blender --background --python blender/audit_scene_relationships.py -- \
      --source exports/scene.glb --registry exports/registry.json \
      --config configs/templates/scene_relationships.default.json \
      --output reports/scene-relationships.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(values)


def asset_id(obj: bpy.types.Object) -> str:
    return str(obj.get("assetId") or obj.name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def engineering_point(point: Vector) -> Vector:
    """Decode the established full-scene GLB axes to x/chainage/elevation."""
    return Vector((point.x, point.z, -point.y))


def group_vertices(objects: list[bpy.types.Object]) -> list[Vector]:
    return [obj.matrix_world @ vertex.co for obj in objects for vertex in obj.data.vertices]


def engineering_bounds(objects: list[bpy.types.Object]) -> dict[str, list[float]]:
    points = [engineering_point(point) for point in group_vertices(objects)]
    return {
        "min": [min(point[axis] for point in points) for axis in range(3)],
        "max": [max(point[axis] for point in points) for axis in range(3)],
    }


def group_bvh(objects: list[bpy.types.Object]) -> BVHTree:
    vertices: list[Vector] = []
    polygons: list[tuple[int, ...]] = []
    for obj in objects:
        offset = len(vertices)
        vertices.extend(obj.matrix_world @ vertex.co for vertex in obj.data.vertices)
        polygons.extend(tuple(offset + index for index in polygon.vertices) for polygon in obj.data.polygons)
    return BVHTree.FromPolygons(vertices, polygons, all_triangles=False)


def ray_elevation(
    bvh: BVHTree,
    raw_bounds: dict[str, list[float]],
    lateral_x: float,
    chainage_y: float,
    *,
    surface: str,
) -> float | None:
    margin = 10.0
    if surface == "top":
        origin_y = raw_bounds["min"][1] - margin
        direction = Vector((0.0, 1.0, 0.0))
    else:
        origin_y = raw_bounds["max"][1] + margin
        direction = Vector((0.0, -1.0, 0.0))
    hit, _normal, _face, _distance = bvh.ray_cast(
        Vector((lateral_x, origin_y, chainage_y)), direction
    )
    return None if hit is None else float(-hit.y)


def raw_bounds(objects: list[bpy.types.Object]) -> dict[str, list[float]]:
    points = group_vertices(objects)
    return {
        "min": [min(point[axis] for point in points) for axis in range(3)],
        "max": [max(point[axis] for point in points) for axis in range(3)],
    }


def nearest_distance(points: list[Vector], target: BVHTree) -> float | None:
    distances = []
    for point in points:
        _location, _normal, _face, distance = target.find_nearest(point)
        if distance is not None and math.isfinite(distance):
            distances.append(float(distance))
    return min(distances) if distances else None


def signed_contact_status(gap: float | None, maximum_gap: float, maximum_penetration: float) -> str:
    if gap is None or not math.isfinite(gap):
        return "no_measurement"
    if gap > maximum_gap:
        return "gap"
    if gap < -maximum_penetration:
        return "excessive_penetration"
    return "pass"


def percentile(values: list[float], quantile: float) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    position = (len(finite) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return finite[lower]
    weight = position - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def main() -> None:
    args = arguments()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    registry_by_id = {str(asset["id"]): asset for asset in registry}
    selectors = config["selectors"]
    thresholds = config["thresholds_m"]
    scope = config["scope"]

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.source.resolve()), import_shading="NORMALS")
    groups: dict[str, list[bpy.types.Object]] = {}
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            groups.setdefault(asset_id(obj), []).append(obj)

    def ids_by_type(type_name: str) -> list[str]:
        identifiers = []
        for identifier, asset in registry_by_id.items():
            if asset.get("type") != type_name or identifier not in groups:
                continue
            bounds = engineering_bounds(groups[identifier])
            center_chainage = (bounds["min"][1] + bounds["max"][1]) / 2.0
            if not scope["chainage_min_m"] <= center_chainage <= scope["chainage_max_m"]:
                continue
            identifiers.append(identifier)
        return sorted(identifiers)

    relationships: list[dict[str, Any]] = []
    platform_id = str(selectors["platform_asset_id"])
    platform_objects = groups[platform_id]
    platform_bvh = group_bvh(platform_objects)
    platform_raw_bounds = raw_bounds(platform_objects)

    column_pattern = re.compile(str(selectors["column_id_regex"]))
    column_ids = ids_by_type(str(selectors["column_type"]))
    for column_id in column_ids:
        match = column_pattern.match(column_id)
        if match is None:
            relationships.append(
                {
                    "kind": "column_pairing",
                    "source": column_id,
                    "status": "unmatched_identifier",
                }
            )
            continue
        index = match.group(2)
        beam_id = str(selectors["cross_beam_id_template"]).format(index=index)
        column_objects = groups[column_id]
        bounds = engineering_bounds(column_objects)
        center_x = (bounds["min"][0] + bounds["max"][0]) / 2.0
        center_chainage = (bounds["min"][1] + bounds["max"][1]) / 2.0
        platform_top = ray_elevation(
            platform_bvh,
            platform_raw_bounds,
            center_x,
            center_chainage,
            surface="top",
        )
        base_gap = None if platform_top is None else bounds["min"][2] - platform_top
        relationships.append(
            {
                "kind": "column_to_platform",
                "source": column_id,
                "target": platform_id,
                "signed_gap_m": base_gap,
                "status": signed_contact_status(
                    base_gap,
                    thresholds["column_platform_maximum_gap"],
                    thresholds["column_platform_maximum_penetration"],
                ),
            }
        )
        if beam_id not in groups:
            relationships.append(
                {
                    "kind": "column_to_cross_beam",
                    "source": column_id,
                    "target": beam_id,
                    "status": "missing_target",
                }
            )
            continue
        beam_objects = groups[beam_id]
        beam_bottom = ray_elevation(
            group_bvh(beam_objects),
            raw_bounds(beam_objects),
            center_x,
            center_chainage,
            surface="bottom",
        )
        top_gap = None if beam_bottom is None else beam_bottom - bounds["max"][2]
        relationships.append(
            {
                "kind": "column_to_cross_beam",
                "source": column_id,
                "target": beam_id,
                "signed_gap_m": top_gap,
                "status": signed_contact_status(
                    top_gap,
                    thresholds["column_beam_maximum_gap"],
                    thresholds["column_beam_maximum_penetration"],
                ),
            }
        )

    roof_ids = ids_by_type(str(selectors["roof_type"]))
    roof_objects = [obj for identifier in roof_ids for obj in groups[identifier]]
    roof_bvh = group_bvh(roof_objects)
    roof_raw_bounds = raw_bounds(roof_objects)
    for beam_id in ids_by_type(str(selectors["cross_beam_type"])):
        beam_objects = groups[beam_id]
        beam_bounds = engineering_bounds(beam_objects)
        beam_bvh = group_bvh(beam_objects)
        beam_raw_bounds = raw_bounds(beam_objects)
        chainage = (beam_bounds["min"][1] + beam_bounds["max"][1]) / 2.0
        width = beam_bounds["max"][0] - beam_bounds["min"][0]
        gaps = []
        for sample in range(1, 10):
            lateral = beam_bounds["min"][0] + width * sample / 10.0
            beam_top = ray_elevation(
                beam_bvh, beam_raw_bounds, lateral, chainage, surface="top"
            )
            roof_bottom = ray_elevation(
                roof_bvh, roof_raw_bounds, lateral, chainage, surface="bottom"
            )
            if beam_top is not None and roof_bottom is not None:
                gaps.append(roof_bottom - beam_top)
        p90_gap = percentile(gaps, 0.9)
        relationships.append(
            {
                "kind": "cross_beam_to_roof",
                "source": beam_id,
                "target": roof_ids,
                "sample_count": len(gaps),
                "signed_gap_p90_m": p90_gap,
                "signed_gap_max_m": max(gaps) if gaps else None,
                "status": signed_contact_status(
                    p90_gap,
                    thresholds["beam_roof_p90_maximum_gap"],
                    thresholds["beam_roof_maximum_penetration"],
                ),
            }
        )

    for beam_id in ids_by_type(str(selectors["longitudinal_beam_type"])):
        beam_objects = groups[beam_id]
        beam_bounds = engineering_bounds(beam_objects)
        beam_bvh = group_bvh(beam_objects)
        beam_raw_bounds = raw_bounds(beam_objects)
        lateral = (beam_bounds["min"][0] + beam_bounds["max"][0]) / 2.0
        length = beam_bounds["max"][1] - beam_bounds["min"][1]
        gaps = []
        beam_hit_count = 0
        sample_total = 17
        for sample in range(1, sample_total + 1):
            chainage = beam_bounds["min"][1] + length * sample / (sample_total + 1)
            beam_top = ray_elevation(
                beam_bvh, beam_raw_bounds, lateral, chainage, surface="top"
            )
            roof_bottom = ray_elevation(
                roof_bvh, roof_raw_bounds, lateral, chainage, surface="bottom"
            )
            if beam_top is not None:
                beam_hit_count += 1
            if beam_top is not None and roof_bottom is not None:
                gaps.append(roof_bottom - beam_top)
        p90_gap = percentile(gaps, 0.9)
        coverage = len(gaps) / beam_hit_count if beam_hit_count else 0.0
        contact = signed_contact_status(
            p90_gap,
            thresholds["beam_roof_p90_maximum_gap"],
            thresholds["beam_roof_maximum_penetration"],
        )
        relationships.append(
            {
                "kind": "longitudinal_beam_to_roof",
                "source": beam_id,
                "target": roof_ids,
                "sample_count": len(gaps),
                "beam_sample_count": beam_hit_count,
                "sample_coverage": coverage,
                "signed_gap_p90_m": p90_gap,
                "signed_gap_max_m": max(gaps) if gaps else None,
                "status": "pass"
                if contact == "pass"
                and coverage >= thresholds["beam_roof_minimum_sample_coverage"]
                else "insufficient_contact",
            }
        )

    stair_ids = [
        identifier
        for identifier in ids_by_type(str(selectors["stair_type"]))
        if identifier.startswith(str(selectors["visible_stair_id_prefix"]))
    ]
    for stair_id in stair_ids:
        stair_objects = groups[stair_id]
        platform_distance = nearest_distance(group_vertices(stair_objects), platform_bvh)
        stair_suffix = stair_id.removeprefix(str(selectors["visible_stair_id_prefix"]))
        guard_id = f"{selectors['stair_guard_prefix']}{stair_suffix}"
        guard_distance = None
        if guard_id in groups:
            guard_distance = nearest_distance(
                group_vertices(stair_objects), group_bvh(groups[guard_id])
            )
        relationships.extend(
            [
                {
                    "kind": "stair_to_platform",
                    "source": stair_id,
                    "target": platform_id,
                    "surface_distance_m": platform_distance,
                    "status": "pass"
                    if platform_distance is not None
                    and platform_distance <= thresholds["stair_platform_surface_distance"]
                    else "gap",
                },
                {
                    "kind": "stair_to_guard",
                    "source": stair_id,
                    "target": guard_id,
                    "surface_distance_m": guard_distance,
                    "status": "pass"
                    if guard_distance is not None
                    and guard_distance <= thresholds["stair_guard_surface_distance"]
                    else "gap",
                },
            ]
        )

    for enclosure_id in ids_by_type(str(selectors["enclosure_type"])):
        enclosure_objects = groups[enclosure_id]
        bounds = engineering_bounds(enclosure_objects)
        minimum_z = bounds["min"][2]
        bottom_points = [
            point
            for point in group_vertices(enclosure_objects)
            if engineering_point(point).z <= minimum_z + 0.15
        ]
        platform_distance = nearest_distance(bottom_points, platform_bvh)
        roof_interval_gap = min(
            max(
                0.0,
                roof_bounds["min"][2] - bounds["max"][2],
                bounds["min"][2] - roof_bounds["max"][2],
            )
            for roof_id in roof_ids
            for roof_bounds in [engineering_bounds(groups[roof_id])]
            if max(bounds["min"][0], roof_bounds["min"][0])
            <= min(bounds["max"][0], roof_bounds["max"][0])
            and max(bounds["min"][1], roof_bounds["min"][1])
            <= min(bounds["max"][1], roof_bounds["max"][1])
        )
        relationships.extend(
            [
                {
                    "kind": "enclosure_to_platform",
                    "source": enclosure_id,
                    "target": platform_id,
                    "surface_distance_m": platform_distance,
                    "status": "pass"
                    if platform_distance is not None
                    and platform_distance
                    <= thresholds["enclosure_platform_surface_distance"]
                    else "gap",
                },
                {
                    "kind": "enclosure_to_roof_envelope",
                    "source": enclosure_id,
                    "target": roof_ids,
                    "interval_gap_m": roof_interval_gap,
                    "status": "pass"
                    if roof_interval_gap <= thresholds["enclosure_roof_interval_gap"]
                    else "gap",
                },
            ]
        )

    failures = [item for item in relationships if item["status"] != "pass"]
    report = {
        "schema_version": "railway.scene-relationships-audit.v1",
        "source": str(args.source.resolve()),
        "source_sha256": sha256(args.source.resolve()),
        "registry": str(args.registry.resolve()),
        "registry_sha256": sha256(args.registry.resolve()),
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config.resolve()),
        "status": "pass" if not failures and relationships else "fail",
        "relationship_count": len(relationships),
        "failure_count": len(failures),
        "failures": failures,
        "relationships": relationships,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if report["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
