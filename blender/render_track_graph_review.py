"""Render six fixed TrackGraph views for visual continuity review.

Run with Blender in background mode, for example:

    blender --background --python blender/render_track_graph_review.py -- \
      --obj path/to/track_graph_track.obj --output-dir path/to/review
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution-x", type=int, default=960)
    parser.add_argument("--resolution-y", type=int, default=540)
    parser.add_argument(
        "--focus-chainage",
        action="append",
        type=float,
        default=[],
        help="Add a close oblique review view at this model chainage",
    )
    return parser.parse_args(values)


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("ReviewWorld")
    scene.world.color = (0.004, 0.012, 0.018)


def import_obj(path: Path) -> list[bpy.types.Object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    before = set(bpy.context.scene.objects)
    bpy.ops.wm.obj_import(filepath=str(path), forward_axis="NEGATIVE_Y", up_axis="Z")
    objects = [obj for obj in bpy.context.scene.objects if obj not in before and obj.type == "MESH"]
    if not objects:
        raise RuntimeError(f"OBJ import produced no mesh objects: {path}")
    return objects


def world_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    minimum = Vector(tuple(min(point[axis] for point in corners) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in corners) for axis in range(3)))
    return minimum, maximum


def look_at(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def add_lighting(center: Vector, span: float) -> None:
    world = bpy.context.scene.world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.004, 0.014, 0.023, 1.0)
    background.inputs["Strength"].default_value = 0.28
    for name, offset, energy, size in (
        ("Key", Vector((0.25 * span, -0.20 * span, 0.45 * span)), 1800.0, 0.35 * span),
        ("Fill", Vector((-0.20 * span, 0.25 * span, 0.25 * span)), 1100.0, 0.25 * span),
    ):
        data = bpy.data.lights.new(name, "AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = size
        light = bpy.data.objects.new(name, data)
        light.location = center + offset
        bpy.context.collection.objects.link(light)
        look_at(light, center)
    sun_data = bpy.data.lights.new("Sun", "SUN")
    sun_data.energy = 2.0
    sun = bpy.data.objects.new("Sun", sun_data)
    sun.rotation_euler = (math.radians(28), math.radians(-22), math.radians(35))
    bpy.context.collection.objects.link(sun)


def render_view(
    camera: bpy.types.Object,
    output: Path,
    eye: Vector,
    target: Vector,
    lens_mm: float,
) -> None:
    camera.location = eye
    camera.data.lens = lens_mm
    look_at(camera, target)
    bpy.context.scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)


def main() -> None:
    args = arguments()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reset_scene()
    objects = import_obj(args.obj.resolve())
    minimum, maximum = world_bounds(objects)
    center = (minimum + maximum) / 2.0
    spans = maximum - minimum
    long_axis = Vector((1.0, 0.0, 0.0)) if spans.x >= spans.y else Vector((0.0, 1.0, 0.0))
    cross_axis = Vector((-long_axis.y, long_axis.x, 0.0))
    long_span = max(spans.x, spans.y)
    cross_span = max(min(spans.x, spans.y), 4.0)
    vertical_span = max(spans.z, 1.0)
    z_axis = Vector((0.0, 0.0, 1.0))

    scene = bpy.context.scene
    scene.render.resolution_x = args.resolution_x
    scene.render.resolution_y = args.resolution_y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.render.resolution_percentage = 100

    camera_data = bpy.data.cameras.new("ReviewCamera")
    camera_data.clip_start = 0.05
    camera_data.clip_end = max(1000.0, long_span * 6.0)
    camera = bpy.data.objects.new("ReviewCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    add_lighting(center, long_span)

    z_target = minimum.z + vertical_span * 0.55
    overview_eye = (
        center
        + long_axis * long_span * 0.28
        + cross_axis * long_span * 0.42
        + z_axis * long_span * 0.30
    )
    views: list[tuple[str, Vector, Vector, float]] = [
        ("01_overall_oblique", overview_eye, Vector((center.x, center.y, z_target)), 52.0),
    ]
    for index, fraction in enumerate((0.03, 0.25, 0.50, 0.75, 0.97), start=2):
        target = center + long_axis * ((fraction - 0.5) * long_span)
        target.z = z_target
        eye = (
            target
            + cross_axis * max(cross_span * 1.8, 18.0)
            - long_axis * min(long_span * 0.05, 10.0)
            + z_axis * max(cross_span * 0.8, 8.0)
        )
        label = (
            "start"
            if fraction < 0.10
            else "end"
            if fraction > 0.90
            else f"seam_{round(fraction * long_span):03d}m"
        )
        views.append((f"{index:02d}_{label}", eye, target, 58.0))

    for chainage in args.focus_chainage:
        fraction = min(1.0, max(0.0, float(chainage) / max(long_span, 1e-9)))
        target = center + long_axis * ((fraction - 0.5) * long_span)
        target.z = z_target
        eye = (
            target
            + cross_axis * max(cross_span * 1.35, 14.0)
            - long_axis * min(long_span * 0.02, 8.0)
            + z_axis * max(cross_span * 0.55, 6.0)
        )
        views.append(
            (
                f"{len(views) + 1:02d}_focus_{round(chainage):04d}m",
                eye,
                target,
                62.0,
            )
        )

    outputs: list[str] = []
    for name, eye, target, lens in views:
        output = output_dir / f"{name}.png"
        render_view(camera, output, eye, target, lens)
        outputs.append(str(output))

    report = {
        "schema_version": "railway.track-graph-fixed-view-review.v1",
        "source_obj": str(args.obj.resolve()),
        "status": "visual_review_required",
        "object_count": len(objects),
        "bounds_min": list(minimum),
        "bounds_max": list(maximum),
        "long_span_m": long_span,
        "views": outputs,
    }
    (output_dir / "fixed-view-review.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
