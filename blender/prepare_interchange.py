"""Blender 4/5 headless mesh cleanup that preserves per-asset objects."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bmesh
import bpy


def arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--merge-distance", type=float, default=0.0001)
    parser.add_argument("--triangulate", action="store_true")
    return parser.parse_args(values)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def import_scene(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        raise ValueError(f"Unsupported input format: {suffix}")


def clean_mesh_object(obj: bpy.types.Object, merge_distance: float, triangulate: bool) -> dict:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    mesh = obj.data
    before = {"vertices": len(mesh.vertices), "edges": len(mesh.edges), "polygons": len(mesh.polygons)}
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_distance)
    bmesh.ops.dissolve_degenerate(bm, dist=merge_distance, edges=bm.edges)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    if triangulate:
        bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.validate(verbose=False, clean_customdata=True)
    mesh.update(calc_edges=True)
    after = {"vertices": len(mesh.vertices), "edges": len(mesh.edges), "polygons": len(mesh.polygons)}
    obj.select_set(False)
    return {"name": obj.name, "before": before, "after": after}


def export_scene(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".glb", ".gltf"}:
        bpy.ops.export_scene.gltf(
            filepath=str(path),
            export_format="GLB" if suffix == ".glb" else "GLTF_SEPARATE",
            export_yup=True,
            export_apply=True,
        )
    elif suffix == ".fbx":
        bpy.ops.export_scene.fbx(
            filepath=str(path),
            use_selection=False,
            axis_forward="-Y",
            axis_up="Z",
            apply_unit_scale=True,
            bake_space_transform=False,
        )
    elif suffix == ".obj":
        bpy.ops.wm.obj_export(filepath=str(path), export_selected_objects=False, export_materials=True)
    else:
        raise ValueError(f"Unsupported output format: {suffix}")


def main() -> None:
    args = arguments()
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == output:
        raise ValueError("Input and output must be different files")
    if args.merge_distance <= 0 or not math.isfinite(args.merge_distance):
        raise ValueError("merge-distance must be a finite positive value")

    clear_scene()
    import_scene(source)
    records = [
        clean_mesh_object(obj, args.merge_distance, args.triangulate)
        for obj in list(bpy.context.scene.objects)
        if obj.type == "MESH"
    ]
    if not records:
        raise ValueError("Imported scene contains no mesh objects")
    export_scene(output)
    report = {
        "schema_version": "railway.blender-interchange-report.v1",
        "input": str(source),
        "output": str(output),
        "blender_version": bpy.app.version_string,
        "merge_distance_m": args.merge_distance,
        "triangulated": args.triangulate,
        "mesh_object_count": len(records),
        "objects": records,
        "limitations": [
            "Cross-object coplanar overlaps require separate visual/topology review.",
            "Materials and thin catenary lines must be accepted again in the target renderer.",
        ],
    }
    report_path = output.with_suffix(output.suffix + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

