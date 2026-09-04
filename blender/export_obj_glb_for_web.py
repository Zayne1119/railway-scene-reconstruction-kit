"""Convert an integrated OBJ/MTL candidate scene to a fast-loading Web GLB.

This is a format conversion only: asset objects and materials remain separate,
and no weld, decimation, normal recalculation, or geometry repair is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(values)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mesh_summary(objects: list[bpy.types.Object]) -> dict[str, object]:
    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    vertex_count = 0
    polygon_count = 0
    for obj in objects:
        vertex_count += len(obj.data.vertices)
        polygon_count += len(obj.data.polygons)
        for corner in obj.bound_box:
            point = obj.matrix_world @ Vector(corner)
            for axis in range(3):
                minimum[axis] = min(minimum[axis], float(point[axis]))
                maximum[axis] = max(maximum[axis], float(point[axis]))
    return {
        "object_count": len(objects),
        "vertex_count": vertex_count,
        "polygon_count": polygon_count,
        "bounds_minimum": minimum,
        "bounds_maximum": maximum,
    }


def main() -> None:
    args = arguments()
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() != ".obj" or output.suffix.lower() != ".glb":
        raise ValueError("Expected an OBJ input and GLB output")
    if source == output:
        raise ValueError("Input and output must differ")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 1.0
    bpy.ops.wm.obj_import(
        filepath=str(source),
        forward_axis="Y",
        up_axis="Z",
        use_split_objects=True,
        use_split_groups=False,
    )
    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not objects:
        raise RuntimeError("OBJ import produced no mesh objects")
    before = mesh_summary(objects)

    output.parent.mkdir(parents=True, exist_ok=True)
    result = bpy.ops.export_scene.gltf(
        filepath=str(output),
        export_format="GLB",
        export_apply=True,
        export_materials="EXPORT",
        export_normals=True,
        export_yup=False,
        export_extras=True,
        export_cameras=False,
        export_lights=False,
    )
    if "FINISHED" not in result or not output.is_file():
        raise RuntimeError(f"GLB export failed: {result}")

    report = {
        "schema_version": "railway.web-glb-conversion.v1",
        "status": "pass",
        "input": str(source),
        "output": str(output),
        "source_sha256": sha256(source),
        "output_sha256": sha256(output),
        "source_size_bytes": source.stat().st_size,
        "output_size_bytes": output.stat().st_size,
        "blender_version": bpy.app.version_string,
        "geometry_operations": [],
        "axis": "Z-up",
        "summary": before,
        "limitations": [
            "This report confirms format conversion, object count and aggregate bounds.",
            "The existing OBJ mesh audit remains the topology acceptance source.",
        ],
    }
    report_path = output.with_suffix(".conversion.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
