"""Audit sleeper phase and ballast-bed cross-sections at a track interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from railway_recon.track_assembly import (
    bed_interface_metrics,
    metrics_pass,
    sleeper_interface_metrics,
)


def arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interface-chainage-m", type=float, default=-100.0)
    parser.add_argument("--nominal-sleeper-spacing-m", type=float, default=0.6)
    parser.add_argument("--max-sleeper-spacing-residual-m", type=float, default=0.03)
    parser.add_argument("--max-sleeper-lateral-offset-m", type=float, default=0.12)
    parser.add_argument("--max-sleeper-top-offset-m", type=float, default=0.08)
    parser.add_argument("--max-bed-longitudinal-gap-m", type=float, default=0.05)
    parser.add_argument("--max-bed-endpoint-skew-m", type=float, default=0.05)
    parser.add_argument("--max-bed-center-offset-m", type=float, default=0.12)
    parser.add_argument("--max-bed-top-offset-m", type=float, default=0.08)
    parser.add_argument("--max-bed-bottom-offset-m", type=float, default=0.12)
    parser.add_argument("--max-bed-top-width-offset-m", type=float, default=0.20)
    parser.add_argument("--max-bed-bottom-width-offset-m", type=float, default=0.30)
    parser.add_argument("--max-material-color-distance", type=float, default=0.12)
    return parser.parse_args(values)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_id(obj: bpy.types.Object) -> str:
    return str(obj.get("assetId") or obj.name)


def local_point(obj: bpy.types.Object, index: int) -> tuple[float, float, float]:
    point = obj.matrix_world @ obj.data.vertices[index].co
    return float(point.x), float(point.z), float(-point.y)


def material_record(objects: list[bpy.types.Object]) -> dict[str, Any]:
    materials = [
        slot.material
        for obj in objects
        for slot in obj.material_slots
        if slot.material is not None
    ]
    if not materials:
        return {"material_name": None, "material_base_color": [0.0, 0.0, 0.0, 1.0]}
    material = materials[0]
    return {
        "material_name": material.name,
        "material_base_color": [float(value) for value in material.diffuse_color],
    }


def connected_components(obj: bpy.types.Object) -> list[dict[str, float]]:
    coordinate_by_index = {
        index: tuple(round(value, 6) for value in local_point(obj, index))
        for index in range(len(obj.data.vertices))
    }
    adjacency: dict[tuple[float, float, float], set[tuple[float, float, float]]] = (
        defaultdict(set)
    )
    used: set[tuple[float, float, float]] = set()
    for polygon in obj.data.polygons:
        coordinates = [coordinate_by_index[index] for index in polygon.vertices]
        used.update(coordinates)
        for index, first in enumerate(coordinates):
            second = coordinates[(index + 1) % len(coordinates)]
            adjacency[first].add(second)
            adjacency[second].add(first)
    components: list[dict[str, float]] = []
    unseen = set(used)
    while unseen:
        root = unseen.pop()
        queue = deque([root])
        coordinates = [root]
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor not in unseen:
                    continue
                unseen.remove(neighbor)
                queue.append(neighbor)
                coordinates.append(neighbor)
        points = list(coordinates)
        minimum = [min(point[axis] for point in points) for axis in range(3)]
        maximum = [max(point[axis] for point in points) for axis in range(3)]
        components.append(
            {
                "center_x_m": (minimum[0] + maximum[0]) / 2.0,
                "center_y_m": (minimum[1] + maximum[1]) / 2.0,
                "center_z_m": (minimum[2] + maximum[2]) / 2.0,
                "minimum_x_m": minimum[0],
                "maximum_x_m": maximum[0],
                "minimum_y_m": minimum[1],
                "maximum_y_m": maximum[1],
                "minimum_z_m": minimum[2],
                "maximum_z_m": maximum[2],
                "vertex_count": float(len(coordinates)),
            }
        )
    return components


def chainage_clusters(
    objects: list[bpy.types.Object], maximum_internal_gap_m: float = 0.30
) -> list[dict[str, float]]:
    """Recover sleeper boxes even when GLB hard normals split face vertices."""
    points = [
        local_point(obj, index)
        for obj in objects
        for index in range(len(obj.data.vertices))
    ]
    points.sort(key=lambda point: point[1])
    groups: list[list[tuple[float, float, float]]] = []
    for point in points:
        if not groups or point[1] - groups[-1][-1][1] > maximum_internal_gap_m:
            groups.append([point])
        else:
            groups[-1].append(point)
    components: list[dict[str, float]] = []
    for group in groups:
        minimum = [min(point[axis] for point in group) for axis in range(3)]
        maximum = [max(point[axis] for point in group) for axis in range(3)]
        components.append(
            {
                "center_x_m": (minimum[0] + maximum[0]) / 2.0,
                "center_y_m": (minimum[1] + maximum[1]) / 2.0,
                "center_z_m": (minimum[2] + maximum[2]) / 2.0,
                "minimum_x_m": minimum[0],
                "maximum_x_m": maximum[0],
                "minimum_y_m": minimum[1],
                "maximum_y_m": maximum[1],
                "minimum_z_m": minimum[2],
                "maximum_z_m": maximum[2],
                "vertex_count": float(len(group)),
            }
        )
    return components


def sleeper_endpoint(
    objects: list[bpy.types.Object], interface_chainage_m: float
) -> dict[str, Any]:
    components = [component for obj in objects for component in connected_components(obj)]
    plausible = [component for component in components if component["vertex_count"] >= 8]
    if not plausible:
        plausible = [
            component
            for component in chainage_clusters(objects)
            if component["vertex_count"] >= 8
            and component["maximum_y_m"] - component["minimum_y_m"] <= 0.45
        ]
    if not plausible:
        raise RuntimeError(f"No full sleeper components for {asset_id(objects[0])}")
    endpoint = min(
        plausible,
        key=lambda component: abs(component["center_y_m"] - interface_chainage_m),
    )
    local = [
        component
        for component in plausible
        if abs(component["center_y_m"] - interface_chainage_m) <= 5.0
    ]
    if len(local) < 2:
        local = sorted(
            plausible,
            key=lambda component: abs(component["center_y_m"] - interface_chainage_m),
        )[:10]
    first, last = min(local, key=lambda value: value["center_y_m"]), max(
        local, key=lambda value: value["center_y_m"]
    )
    delta_x = last["center_x_m"] - first["center_x_m"]
    delta_y = last["center_y_m"] - first["center_y_m"]
    length = math.hypot(delta_x, delta_y)
    direction = [delta_x / length, delta_y / length]
    return {
        "asset_id": asset_id(objects[0]),
        "component_count": len(plausible),
        "longitudinal_direction_xy": direction,
        **endpoint,
        **material_record(objects),
    }


def bed_endpoint(
    objects: list[bpy.types.Object],
    interface_chainage_m: float,
    direction_xy: list[float],
) -> dict[str, Any]:
    points = [
        local_point(obj, index)
        for obj in objects
        for index in range(len(obj.data.vertices))
    ]
    minimum_y = min(point[1] for point in points)
    maximum_y = max(point[1] for point in points)
    direction_length = math.hypot(direction_xy[0], direction_xy[1])
    direction = [
        direction_xy[0] / direction_length,
        direction_xy[1] / direction_length,
    ]
    lateral = [-direction[1], direction[0]]

    projected = [
        (
            point,
            point[0] * direction[0] + point[1] * direction[1],
            point[0] * lateral[0] + point[1] * lateral[1],
        )
        for point in points
    ]
    minimum_longitudinal = min(value[1] for value in projected)
    maximum_longitudinal = max(value[1] for value in projected)
    mean_chainage = sum(point[1] for point in points) / len(points)
    use_minimum = mean_chainage > interface_chainage_m
    cap_depth_m = 1.25
    if use_minimum:
        projected_section = [
            value for value in projected if value[1] <= minimum_longitudinal + cap_depth_m
        ]
    else:
        projected_section = [
            value for value in projected if value[1] >= maximum_longitudinal - cap_depth_m
        ]
    local_top = max(value[0][2] for value in projected_section)
    local_bottom = min(value[0][2] for value in projected_section)
    midpoint_z = (local_top + local_bottom) / 2.0
    top_values = [value for value in projected_section if value[0][2] > midpoint_z]
    bottom_values = [value for value in projected_section if value[0][2] <= midpoint_z]

    def profile_corner(
        values: list[tuple[tuple[float, float, float], float, float]],
        choose_minimum_lateral: bool,
    ) -> tuple[tuple[float, float, float], float, float]:
        extreme_lateral = (
            min(value[2] for value in values)
            if choose_minimum_lateral
            else max(value[2] for value in values)
        )
        band = [
            value for value in values if abs(value[2] - extreme_lateral) <= 0.12
        ]
        return (
            min(band, key=lambda value: value[1])
            if use_minimum
            else max(band, key=lambda value: value[1])
        )

    top_left = profile_corner(top_values, True)
    top_right = profile_corner(top_values, False)
    bottom_left = profile_corner(bottom_values, True)
    bottom_right = profile_corner(bottom_values, False)
    corners = [top_left, top_right, bottom_left, bottom_right]
    endpoint_longitudinal = sum(value[1] for value in corners) / len(corners)
    endpoint_y = sum(value[0][1] for value in corners) / len(corners)
    endpoint_skew = max(value[1] for value in corners) - min(
        value[1] for value in corners
    )
    top_min_x, top_max_x = sorted((top_left[2], top_right[2]))
    bottom_min_x, bottom_max_x = sorted((bottom_left[2], bottom_right[2]))
    top_z = (top_left[0][2] + top_right[0][2]) / 2.0
    bottom_z = (bottom_left[0][2] + bottom_right[0][2]) / 2.0
    return {
        "asset_id": asset_id(objects[0]),
        "chainage_range_m": [minimum_y, maximum_y],
        "endpoint_chainage_m": endpoint_y,
        "longitudinal_range_m": [minimum_longitudinal, maximum_longitudinal],
        "endpoint_longitudinal_m": endpoint_longitudinal,
        "endpoint_skew_m": endpoint_skew,
        "longitudinal_direction_xy": direction,
        "lateral_direction_xy": lateral,
        "section_tolerance_m": cap_depth_m,
        "section_vertex_count": len(projected_section),
        "top_center_x_m": (top_min_x + top_max_x) / 2.0,
        "bottom_center_x_m": (bottom_min_x + bottom_max_x) / 2.0,
        "top_width_m": top_max_x - top_min_x,
        "bottom_width_m": bottom_max_x - bottom_min_x,
        "top_z_m": top_z,
        "bottom_z_m": bottom_z,
        **material_record(objects),
    }


def track_number(identifier: str) -> int:
    match = re.search(r"(?:TRACK-000|Track_00)([123])", identifier)
    if not match:
        raise ValueError(f"Cannot recover track number from {identifier}")
    return int(match.group(1))


def grouped_objects(pattern: re.Pattern[str]) -> dict[int, list[bpy.types.Object]]:
    result: dict[int, list[bpy.types.Object]] = defaultdict(list)
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or not pattern.fullmatch(asset_id(obj)):
            continue
        result[track_number(asset_id(obj))].append(obj)
    return dict(result)


def main() -> None:
    args = arguments()
    source = args.source.resolve()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source), import_shading="NORMALS")

    new_sleepers = grouped_objects(re.compile(r"TRACK-000[123]-SLEEPERS"))
    retained_sleepers = grouped_objects(
        re.compile(r"s200_250m\.Track_00[123]_Sleepers")
    )
    new_beds = grouped_objects(re.compile(r"TRACK-000[123]-BED"))
    retained_beds = grouped_objects(
        re.compile(r"s200_250m\.Track_00[123]_TrackBed")
    )
    groups = (new_sleepers, retained_sleepers, new_beds, retained_beds)
    if any(set(group) != {1, 2, 3} for group in groups):
        raise RuntimeError("Expected three new and three retained sleeper/bed groups")

    sleeper_thresholds = {
        "spacing_residual_m": args.max_sleeper_spacing_residual_m,
        "lateral_offset_m": args.max_sleeper_lateral_offset_m,
        "top_offset_m": args.max_sleeper_top_offset_m,
    }
    bed_thresholds = {
        "longitudinal_gap_m": args.max_bed_longitudinal_gap_m,
        "endpoint_skew_m": args.max_bed_endpoint_skew_m,
        "top_center_lateral_offset_m": args.max_bed_center_offset_m,
        "top_elevation_offset_m": args.max_bed_top_offset_m,
        "bottom_elevation_offset_m": args.max_bed_bottom_offset_m,
        "top_width_offset_m": args.max_bed_top_width_offset_m,
        "bottom_width_offset_m": args.max_bed_bottom_width_offset_m,
        "material_color_distance": args.max_material_color_distance,
    }
    tracks = []
    for number in (1, 2, 3):
        new_sleeper = sleeper_endpoint(
            new_sleepers[number], args.interface_chainage_m
        )
        retained_sleeper = sleeper_endpoint(
            retained_sleepers[number], args.interface_chainage_m
        )
        sleeper_metrics = sleeper_interface_metrics(
            new_sleeper, retained_sleeper, args.nominal_sleeper_spacing_m
        )
        combined_direction = [
            new_sleeper["longitudinal_direction_xy"][axis]
            + retained_sleeper["longitudinal_direction_xy"][axis]
            for axis in range(2)
        ]
        new_bed = bed_endpoint(
            new_beds[number], args.interface_chainage_m, combined_direction
        )
        retained_bed = bed_endpoint(
            retained_beds[number], args.interface_chainage_m, combined_direction
        )
        bed_metrics = bed_interface_metrics(new_bed, retained_bed)
        sleeper_status = (
            "pass" if metrics_pass(sleeper_metrics, sleeper_thresholds) else "fail"
        )
        bed_status = "pass" if metrics_pass(bed_metrics, bed_thresholds) else "fail"
        tracks.append(
            {
                "track_number": number,
                "sleeper_interface": {
                    "new": new_sleeper,
                    "retained": retained_sleeper,
                    "metrics": sleeper_metrics,
                    "status": sleeper_status,
                },
                "bed_interface": {
                    "new": new_bed,
                    "retained": retained_bed,
                    "metrics": bed_metrics,
                    "status": bed_status,
                },
                "status": (
                    "pass" if sleeper_status == bed_status == "pass" else "fail"
                ),
            }
        )

    report = {
        "schema_version": "railway.track-assembly-interface-audit.v1",
        "source": str(source),
        "source_sha256": sha256(source),
        "registry": str(args.registry.resolve()),
        "registry_sha256": sha256(args.registry.resolve()),
        "interface_chainage_m": args.interface_chainage_m,
        "status": "pass" if all(item["status"] == "pass" for item in tracks) else "fail",
        "thresholds": {
            "sleeper": sleeper_thresholds,
            "bed": bed_thresholds,
        },
        "tracks": tracks,
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
