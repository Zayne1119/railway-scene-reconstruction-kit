"""Audit rail continuity across a new-to-retained model interface.

Run with Blender in background mode, for example::

    blender --background --python blender/audit_track_interface.py -- \
      --source exports/scene.glb --output reports/track-interface.json \
      --interface-chainage-m -100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import bpy


def arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interface-chainage-m", type=float, required=True)
    parser.add_argument("--new-prefix", default="TRACK-")
    parser.add_argument("--retained-prefix", default="s200_250m.Track")
    parser.add_argument("--max-longitudinal-gap-m", type=float, default=0.10)
    parser.add_argument("--max-lateral-offset-m", type=float, default=0.12)
    parser.add_argument("--max-rail-top-offset-m", type=float, default=0.08)
    return parser.parse_args(values)


def asset_id(obj: bpy.types.Object) -> str:
    return str(obj.get("assetId") or obj.name)


def local_points(obj: bpy.types.Object) -> list[tuple[float, float, float]]:
    """Recover project local (x, chainage, elevation) from Blender encoding."""

    result = []
    for vertex in obj.data.vertices:
        point = obj.matrix_world @ vertex.co
        result.append((float(point.x), float(point.z), float(-point.y)))
    return result


def endpoint_record(
    obj: bpy.types.Object, interface_chainage_m: float
) -> dict[str, Any]:
    points = local_points(obj)
    minimum = min(point[1] for point in points)
    maximum = max(point[1] for point in points)
    endpoint = min((minimum, maximum), key=lambda value: abs(value - interface_chainage_m))
    distances = sorted(abs(point[1] - endpoint) for point in points)
    tolerance = max(0.02, distances[min(len(distances) - 1, 31)] + 1e-6)
    section = [point for point in points if abs(point[1] - endpoint) <= tolerance]
    return {
        "asset_id": asset_id(obj),
        "chainage_range_m": [minimum, maximum],
        "endpoint_chainage_m": endpoint,
        "section_tolerance_m": tolerance,
        "section_vertex_count": len(section),
        "center_x_m": sum(point[0] for point in section) / len(section),
        "rail_top_z_m": max(point[2] for point in section),
    }


def main() -> None:
    args = arguments()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.source.resolve()), import_shading="NORMALS")
    rails = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and "rail" in asset_id(obj).lower()
    ]
    new = sorted(
        (
            endpoint_record(obj, args.interface_chainage_m)
            for obj in rails
            if asset_id(obj).startswith(args.new_prefix)
        ),
        key=lambda item: item["center_x_m"],
    )
    retained = sorted(
        (
            endpoint_record(obj, args.interface_chainage_m)
            for obj in rails
            if asset_id(obj).startswith(args.retained_prefix)
        ),
        key=lambda item: item["center_x_m"],
    )
    if len(new) != 6 or len(retained) != 6:
        raise RuntimeError(
            f"Expected six new and six retained rails, got {len(new)} and {len(retained)}"
        )

    pairs = []
    for new_rail, retained_rail in zip(new, retained, strict=True):
        metrics = {
            "longitudinal_gap_m": abs(
                new_rail["endpoint_chainage_m"]
                - retained_rail["endpoint_chainage_m"]
            ),
            "lateral_offset_m": abs(
                new_rail["center_x_m"] - retained_rail["center_x_m"]
            ),
            "rail_top_offset_m": abs(
                new_rail["rail_top_z_m"] - retained_rail["rail_top_z_m"]
            ),
        }
        passed = (
            metrics["longitudinal_gap_m"] <= args.max_longitudinal_gap_m
            and metrics["lateral_offset_m"] <= args.max_lateral_offset_m
            and metrics["rail_top_offset_m"] <= args.max_rail_top_offset_m
        )
        pairs.append(
            {
                "new": new_rail,
                "retained": retained_rail,
                "metrics": metrics,
                "status": "pass" if passed else "fail",
            }
        )

    maxima = {
        key: max(pair["metrics"][key] for pair in pairs)
        for key in ("longitudinal_gap_m", "lateral_offset_m", "rail_top_offset_m")
    }
    report = {
        "schema_version": "railway.track-interface-audit.v1",
        "source": str(args.source.resolve()),
        "interface_chainage_m": args.interface_chainage_m,
        "status": "pass" if all(pair["status"] == "pass" for pair in pairs) else "fail",
        "thresholds_m": {
            "longitudinal_gap": args.max_longitudinal_gap_m,
            "lateral_offset": args.max_lateral_offset_m,
            "rail_top_offset": args.max_rail_top_offset_m,
        },
        "maxima_m": maxima,
        "pairs": pairs,
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
