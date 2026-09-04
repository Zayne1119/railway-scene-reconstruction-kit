"""Render deterministic OBJ review views without a DCC dependency.

The renderer is intentionally small: it parses OBJ/MTL, applies back-face
culling, shades triangles, and writes fixed-view PNGs plus a JSON manifest.
It is meant for geometry acceptance in environments where Blender/UE is not
available, not for photorealistic delivery renders.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BACKGROUND = (5, 18, 26)
GRID = (22, 56, 68)
TEXT = (221, 240, 244)


@dataclass(frozen=True)
class Triangle:
    indexes: tuple[int, int, int]
    object_name: str
    material_name: str


@dataclass(frozen=True)
class View:
    name: str
    title: str
    eye_offset: tuple[float, float, float]
    target: tuple[float, float, float] | None = None
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    object_prefix: str | None = None
    roof_surface: str | None = None
    crop_radius_xy: float | None = None


def _normalise(value: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(value))
    if length <= 1.0e-12:
        raise ValueError("Cannot normalise a zero vector")
    return value / length


def parse_mtl(path: Path) -> dict[str, tuple[int, int, int]]:
    colours: dict[str, tuple[int, int, int]] = {}
    if not path.is_file():
        return colours
    current = "default"
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("newmtl "):
            current = line.split(maxsplit=1)[1]
        elif line.startswith("Kd "):
            values = [float(item) for item in line.split()[1:4]]
            colours[current] = tuple(round(max(0.0, min(1.0, item)) * 255) for item in values)
    return colours


def parse_obj(path: Path) -> tuple[np.ndarray, list[Triangle], dict[str, tuple[int, int, int]]]:
    vertices: list[list[float]] = []
    triangles: list[Triangle] = []
    object_name = "default"
    material_name = "default"
    material_library: Path | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "v" and len(parts) >= 4:
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] in {"o", "g"} and len(parts) >= 2:
            object_name = " ".join(parts[1:])
        elif parts[0] == "usemtl" and len(parts) >= 2:
            material_name = " ".join(parts[1:])
        elif parts[0] == "mtllib" and len(parts) >= 2:
            material_library = path.parent / " ".join(parts[1:])
        elif parts[0] == "f" and len(parts) >= 4:
            indexes: list[int] = []
            for token in parts[1:]:
                raw_index = int(token.split("/")[0])
                indexes.append(raw_index - 1 if raw_index > 0 else len(vertices) + raw_index)
            for index in range(1, len(indexes) - 1):
                triangles.append(
                    Triangle(
                        (indexes[0], indexes[index], indexes[index + 1]),
                        object_name,
                        material_name,
                    )
                )
    if not vertices or not triangles:
        raise ValueError(f"OBJ has no renderable geometry: {path}")
    if material_library is None:
        material_library = path.with_suffix(".mtl")
    return np.asarray(vertices, dtype=np.float64), triangles, parse_mtl(material_library)


def _shade(colour: tuple[int, int, int], normal: np.ndarray) -> tuple[int, int, int]:
    light = _normalise(np.asarray([0.35, -0.55, 0.76], dtype=np.float64))
    amount = 0.44 + 0.56 * abs(float(np.dot(normal, light)))
    return tuple(int(max(0, min(255, round(channel * amount)))) for channel in colour)


def _select_triangles(
    vertices: np.ndarray,
    triangles: Iterable[Triangle],
    view: View,
) -> list[Triangle]:
    selected: list[Triangle] = []
    target = np.asarray(view.target, dtype=np.float64) if view.target is not None else None
    node_prefixes = tuple(view.object_prefix.split("|")) if view.object_prefix else ()
    roof_prefixes = tuple(view.roof_surface.split("|")) if view.roof_surface else ()
    for triangle in triangles:
        is_node = bool(node_prefixes) and triangle.object_name.startswith(node_prefixes)
        is_roof = bool(roof_prefixes) and triangle.object_name.startswith(roof_prefixes)
        if view.object_prefix is not None and not (is_node or is_roof):
            continue
        crop_selected = view.object_prefix is None or is_node or is_roof
        if target is not None and view.crop_radius_xy is not None and crop_selected:
            points_xy = vertices[list(triangle.indexes), :2]
            if not np.any(
                np.linalg.norm(points_xy - target[:2], axis=1) <= view.crop_radius_xy
            ):
                continue
        selected.append(triangle)
    return selected


def render_view(
    vertices: np.ndarray,
    triangles: list[Triangle],
    colours: dict[str, tuple[int, int, int]],
    view: View,
    output: Path,
    width: int = 1600,
    height: int = 900,
) -> dict[str, object]:
    selected = _select_triangles(vertices, triangles, view)
    if not selected:
        raise ValueError(f"View {view.name} selected no triangles")
    selected_indexes = sorted({index for item in selected for index in item.indexes})
    selected_vertices = vertices[selected_indexes]
    target = (
        np.asarray(view.target, dtype=np.float64)
        if view.target is not None
        else (selected_vertices.min(axis=0) + selected_vertices.max(axis=0)) * 0.5
    )
    eye = target + np.asarray(view.eye_offset, dtype=np.float64)
    forward = _normalise(target - eye)
    right = _normalise(np.cross(forward, np.asarray(view.up, dtype=np.float64)))
    camera_up = _normalise(np.cross(right, forward))
    relative = vertices - target
    projected = np.column_stack((relative @ right, relative @ camera_up))
    framing_indexes = selected_indexes
    if view.target is not None and view.crop_radius_xy is not None:
        nearby = [
            index
            for index in selected_indexes
            if float(np.linalg.norm(vertices[index, :2] - target[:2]))
            <= view.crop_radius_xy
        ]
        if nearby:
            framing_indexes = nearby
    selected_projected = projected[framing_indexes]
    minimum = selected_projected.min(axis=0)
    maximum = selected_projected.max(axis=0)
    extent = np.maximum(maximum - minimum, 1.0e-6)
    scale = min((width - 120) / extent[0], (height - 150) / extent[1])
    centre = (minimum + maximum) * 0.5

    canvas = Image.new("RGB", (width * 2, height * 2), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    for x in range(0, width * 2, 120):
        draw.line([(x, 0), (x, height * 2)], fill=GRID, width=1)
    for y in range(0, height * 2, 120):
        draw.line([(0, y), (width * 2, y)], fill=GRID, width=1)

    visible: list[tuple[float, Triangle, np.ndarray]] = []
    culled_count = 0
    for triangle in selected:
        points = vertices[list(triangle.indexes)]
        normal_raw = np.cross(points[1] - points[0], points[2] - points[0])
        area_twice = float(np.linalg.norm(normal_raw))
        if area_twice <= 1.0e-12:
            continue
        normal = normal_raw / area_twice
        centroid = points.mean(axis=0)
        if float(np.dot(normal, eye - centroid)) <= 0.0:
            culled_count += 1
            continue
        depth = float(np.dot(centroid - eye, forward))
        visible.append((depth, triangle, normal))
    visible.sort(key=lambda item: item[0], reverse=True)

    for _, triangle, normal in visible:
        points_2d = projected[list(triangle.indexes)]
        pixel_points = [
            (
                round((width * 0.5 + (point[0] - centre[0]) * scale) * 2),
                round((height * 0.53 - (point[1] - centre[1]) * scale) * 2),
            )
            for point in points_2d
        ]
        base = colours.get(triangle.material_name, (150, 172, 178))
        is_conductor = triangle.object_name.startswith(
            (
                "CONTACT-WIRE-",
                "MESSENGER-WIRE-",
                "SUPPLEMENTAL-AUX-CONDUCTOR-",
                "SUPPLEMENTAL-BOUNDARY-CONDUCTOR-",
                "ADJACENT-NEXT-BOUNDARY-CONDUCTOR-",
                "ADJACENT-PREVIOUS-BOUNDARY-CONDUCTOR-",
            )
        )
        fill = (255, 158, 28) if is_conductor else _shade(base, normal)
        draw.polygon(pixel_points, fill=fill)
        if not is_conductor:
            draw.line(pixel_points + [pixel_points[0]], fill=(12, 30, 38), width=2)

    font = ImageFont.load_default(size=28)
    small = ImageFont.load_default(size=20)
    draw.rectangle([(30, 25), (width * 2 - 30, 120)], fill=(3, 14, 21), outline=(79, 187, 205), width=2)
    draw.text((55, 45), view.title, fill=TEXT, font=font)
    draw.text(
        (55, 83),
        f"back-face culling ON | visible {len(visible)} / selected {len(selected)} triangles",
        fill=(143, 194, 204),
        font=small,
    )
    draw.text(
        (width * 2 - 700, height * 2 - 55),
        "candidate geometry only - not merged to formal model",
        fill=(174, 255, 81),
        font=small,
    )
    canvas = canvas.resize((width, height), Image.Resampling.LANCZOS)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return {
        "name": view.name,
        "title": view.title,
        "output": str(output),
        "selected_triangle_count": len(selected),
        "visible_triangle_count": len(visible),
        "backface_culled_triangle_count": culled_count,
        "camera_eye": eye.tolist(),
        "camera_target": target.tolist(),
    }


def _object_centres(vertices: np.ndarray, triangles: list[Triangle]) -> dict[str, list[float]]:
    indexes: dict[str, set[int]] = {}
    for triangle in triangles:
        indexes.setdefault(triangle.object_name, set()).update(triangle.indexes)
    return {
        name: vertices[sorted(items)].mean(axis=0).tolist()
        for name, items in indexes.items()
    }


def _horizontal_principal_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centred = points[:, :2] - points[:, :2].mean(axis=0)
    covariance = np.cov(centred, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    along_xy = vectors[:, int(np.argmax(values))]
    if along_xy[0] < 0.0:
        along_xy *= -1.0
    along = np.asarray([along_xy[0], along_xy[1], 0.0], dtype=np.float64)
    cross = np.asarray([-along_xy[1], along_xy[0], 0.0], dtype=np.float64)
    return _normalise(along), _normalise(cross)


def build_views(vertices: np.ndarray, triangles: list[Triangle]) -> list[View]:
    centres = _object_centres(vertices, triangles)
    column_names = sorted(
        name
        for name in centres
        if len(name.split("-")) == 4
        and name.startswith("RIGHT-CANOPY-GRID-")
        and name.rsplit("-", 1)[-1].isdigit()
    )
    if not column_names:
        raise ValueError("Canopy preset requires at least one emitted column object")
    roof_names = sorted(
        {
            triangle.object_name.split("-SURFACE-", 1)[0]
            for triangle in triangles
            if triangle.object_name.startswith("RIGHT-CANOPY-ROOF-")
        }
    )
    if not roof_names:
        raise ValueError("Canopy preset requires at least one emitted roof object")
    all_roof_indexes = sorted(
        {
            vertex_index
            for triangle in triangles
            if triangle.object_name.startswith(tuple(roof_names))
            for vertex_index in triangle.indexes
        }
    )
    along, cross = _horizontal_principal_axes(vertices[all_roof_indexes])
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    views = [
        View(
            "01_overall_oblique",
            "01 Overall oblique / candidate scope",
            tuple((70.0 * along - 120.0 * cross + 58.0 * up).tolist()),
        ),
        View(
            "02_top_plan",
            "02 Top plan / roof segmentation",
            (0.0, 0.0, 160.0),
            up=tuple(along.tolist()),
        ),
        View(
            "03_long_side",
            "03 Long side / column-roof continuity",
            tuple((95.0 * cross + 10.0 * up).tolist()),
        ),
    ]
    selected_indexes = np.linspace(
        0, len(column_names) - 1, min(3, len(column_names)), dtype=np.int64
    )
    node_specs = [
        (
            column_names[int(column_index)],
            roof_names[0],
            f"{view_index:02d} Confirmed node / {column_names[int(column_index)]}",
        )
        for view_index, column_index in enumerate(selected_indexes, start=4)
    ]
    for index, (prefix, roof, title) in enumerate(node_specs, start=4):
        shaft_centre = np.asarray(centres[prefix], dtype=np.float64)
        target = shaft_centre.copy()
        roof_indexes = sorted(
            {
                vertex_index
                for triangle in triangles
                if triangle.object_name.startswith(roof)
                for vertex_index in triangle.indexes
            }
        )
        if not roof_indexes:
            raise ValueError(f"No roof object found for prefix: {roof}")
        target[2] = float(vertices[roof_indexes, 2].max()) - 1.1
        views.append(
            View(
                f"{index:02d}_{prefix.lower().replace('-', '_')}",
                title,
                tuple((-13.0 * along + 4.0 * cross + 5.5 * up).tolist()),
                target=tuple(target.tolist()),
                object_prefix=prefix,
                roof_surface=roof,
                crop_radius_xy=4.2,
            )
        )
    return views


def _prefix_centre(
    vertices: np.ndarray, triangles: list[Triangle], prefixes: tuple[str, ...]
) -> np.ndarray:
    indexes = sorted(
        {
            index
            for triangle in triangles
            if triangle.object_name.startswith(prefixes)
            for index in triangle.indexes
        }
    )
    if not indexes:
        raise ValueError(f"No objects found for prefixes: {prefixes}")
    return vertices[indexes].mean(axis=0)


def build_track_views(vertices: np.ndarray, triangles: list[Triangle]) -> list[View]:
    track_indexes = sorted(
        {
            index
            for triangle in triangles
            if triangle.object_name.startswith("TRACK-")
            for index in triangle.indexes
        }
    )
    if not track_indexes:
        raise ValueError("Track preset requires TRACK-* objects")
    track_vertices = vertices[track_indexes]
    track_centre = track_vertices.mean(axis=0)
    centred_xy = track_vertices[:, :2] - track_centre[:2]
    _, _, vh = np.linalg.svd(centred_xy, full_matrices=False)
    along = _normalise(np.asarray([vh[0, 0], vh[0, 1], 0.0]))
    cross = np.asarray([-along[1], along[0], 0.0])
    up = np.asarray([0.0, 0.0, 1.0])

    track2 = _prefix_centre(vertices, triangles, ("TRACK-0002",))
    track3 = _prefix_centre(vertices, triangles, ("TRACK-0003",))
    interface = _prefix_centre(
        vertices, triangles, ("TRACK-0003", "S2100_2150M-PLATFORM")
    )
    return [
        View(
            "01_track_overall",
            "01 Accepted tracks / overall oblique",
            tuple((-55.0 * along + 55.0 * cross + 34.0 * up).tolist()),
            target=tuple(track_centre.tolist()),
            object_prefix="TRACK-",
        ),
        View(
            "02_track_top_plan",
            "02 Accepted tracks / top plan and separation",
            (0.0, 0.0, 120.0),
            target=tuple(track_centre.tolist()),
            up=tuple(along.tolist()),
            object_prefix="TRACK-",
        ),
        View(
            "03_track_0002_oblique",
            "03 TRACK-0002 / continuous rails, sleepers and bed",
            tuple((-32.0 * along + 22.0 * cross + 15.0 * up).tolist()),
            target=tuple(track2.tolist()),
            object_prefix="TRACK-0002",
        ),
        View(
            "04_track_0003_oblique",
            "04 TRACK-0003 / continuous rails, sleepers and bed",
            tuple((32.0 * along - 22.0 * cross + 15.0 * up).tolist()),
            target=tuple(track3.tolist()),
            object_prefix="TRACK-0003",
        ),
        View(
            "05_track_platform_interface",
            "05 Nearest accepted track / platform relationship",
            tuple((-35.0 * along + 36.0 * cross + 20.0 * up).tolist()),
            target=tuple(interface.tolist()),
            object_prefix="TRACK-0003",
            roof_surface="S2100_2150M-PLATFORM",
        ),
        View(
            "06_track_end_profile",
            "06 Track end / gauge, rail and bed profile",
            tuple((58.0 * along + 5.0 * up).tolist()),
            target=tuple(track3.tolist()),
            object_prefix="TRACK-0003",
        ),
    ]


def build_track_boundary_views(
    vertices: np.ndarray, triangles: list[Triangle]
) -> list[View]:
    """Render the handoff between reviewed and adjacent track namespaces."""
    current_prefix = "TRACKGRAPH--TRACK-"
    adjacent_prefix = "TRACK-"
    current_indexes = sorted(
        {
            index
            for triangle in triangles
            if triangle.object_name.startswith(current_prefix)
            for index in triangle.indexes
        }
    )
    adjacent_indexes = sorted(
        {
            index
            for triangle in triangles
            if triangle.object_name.startswith(adjacent_prefix)
            for index in triangle.indexes
        }
    )
    if not current_indexes or not adjacent_indexes:
        raise ValueError(
            "Track-boundary preset requires TRACKGRAPH--TRACK-* and TRACK-* objects"
        )
    current = vertices[np.asarray(current_indexes, dtype=np.int64)]
    adjacent = vertices[np.asarray(adjacent_indexes, dtype=np.int64)]
    along, cross = _horizontal_principal_axes(current)
    if float(np.dot(adjacent[:, :2].mean(axis=0) - current[:, :2].mean(axis=0), along[:2])) < 0.0:
        along *= -1.0
        cross *= -1.0
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    reference_xy = current[:, :2].mean(axis=0)
    current_station = (current[:, :2] - reference_xy) @ along[:2]
    adjacent_station = (adjacent[:, :2] - reference_xy) @ along[:2]
    current_edge = current_station >= float(current_station.max()) - 0.5
    adjacent_edge = adjacent_station <= float(adjacent_station.min()) + 0.5
    seam_vertices = np.vstack((current[current_edge], adjacent[adjacent_edge]))
    seam_target = seam_vertices.mean(axis=0)
    prefixes = f"{current_prefix}|{adjacent_prefix}"
    return [
        View(
            "01_track_boundary_overall",
            "01 Track handoff / reviewed and adjacent geometry",
            tuple((-32.0 * along + 30.0 * cross + 18.0 * up).tolist()),
            target=tuple(seam_target.tolist()),
            object_prefix=prefixes,
            crop_radius_xy=38.0,
        ),
        View(
            "02_track_boundary_top",
            "02 Track handoff / top plan continuity",
            tuple((55.0 * up).tolist()),
            target=tuple(seam_target.tolist()),
            up=tuple(along.tolist()),
            object_prefix=prefixes,
            crop_radius_xy=24.0,
        ),
        View(
            "03_track_boundary_side",
            "03 Track handoff / longitudinal elevation",
            tuple((32.0 * cross + 5.0 * up).tolist()),
            target=tuple(seam_target.tolist()),
            object_prefix=prefixes,
            crop_radius_xy=24.0,
        ),
        View(
            "04_track_boundary_low_forward",
            "04 Track handoff / low forward inspection",
            tuple((-23.0 * along + 12.0 * cross + 4.0 * up).tolist()),
            target=tuple(seam_target.tolist()),
            object_prefix=prefixes,
            crop_radius_xy=22.0,
        ),
        View(
            "05_track_boundary_low_reverse",
            "05 Track handoff / low reverse inspection",
            tuple((23.0 * along - 12.0 * cross + 4.0 * up).tolist()),
            target=tuple(seam_target.tolist()),
            object_prefix=prefixes,
            crop_radius_xy=22.0,
        ),
        View(
            "06_track_boundary_profile",
            "06 Track handoff / rail, sleeper and bed profile",
            tuple((34.0 * along + 3.0 * up).tolist()),
            target=tuple(seam_target.tolist()),
            object_prefix=prefixes,
            crop_radius_xy=15.0,
        ),
    ]


def build_corridor_views(vertices: np.ndarray, triangles: list[Triangle]) -> list[View]:
    """Render a complete namespaced multi-segment corridor without asset filtering."""
    centre = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    centred_xy = vertices[:, :2] - vertices[:, :2].mean(axis=0)
    _, _, vh = np.linalg.svd(centred_xy, full_matrices=False)
    along = _normalise(np.asarray([vh[0, 0], vh[0, 1], 0.0]))
    cross = np.asarray([-along[1], along[0], 0.0])
    up = np.asarray([0.0, 0.0, 1.0])
    station = (vertices[:, :2] - centre[:2]) @ along[:2]
    length = max(float(np.ptp(station)), 50.0)
    first_seam = centre - along * length / 6.0
    second_seam = centre + along * length / 6.0
    first_seam[2] = centre[2]
    second_seam[2] = centre[2]
    return [
        View(
            "01_corridor_overall",
            "01 Complete corridor / overall oblique",
            tuple((-0.62 * length * along + 0.48 * length * cross + 0.28 * length * up).tolist()),
            target=tuple(centre.tolist()),
        ),
        View(
            "02_corridor_top_plan",
            "02 Complete corridor / top plan",
            tuple((1.05 * length * up).tolist()),
            target=tuple(centre.tolist()),
            up=tuple(along.tolist()),
        ),
        View(
            "03_corridor_low_forward",
            "03 Complete corridor / low forward",
            tuple((-0.58 * length * along + 0.10 * length * cross + 0.08 * length * up).tolist()),
            target=tuple(centre.tolist()),
        ),
        View(
            "04_corridor_low_reverse",
            "04 Complete corridor / low reverse",
            tuple((0.58 * length * along - 0.10 * length * cross + 0.08 * length * up).tolist()),
            target=tuple(centre.tolist()),
        ),
        View(
            "05_first_segment_seam",
            "05 First ownership seam / interface review",
            tuple((-18.0 * along + 20.0 * cross + 13.0 * up).tolist()),
            target=tuple(first_seam.tolist()),
            crop_radius_xy=22.0,
        ),
        View(
            "06_second_segment_seam",
            "06 Second ownership seam / interface review",
            tuple((18.0 * along - 20.0 * cross + 13.0 * up).tolist()),
            target=tuple(second_seam.tolist()),
            crop_radius_xy=22.0,
        ),
    ]


def build_catenary_views(vertices: np.ndarray, triangles: list[Triangle]) -> list[View]:
    centre = _prefix_centre(vertices, triangles, ("CATENARY-MAST-",))
    return [
        View(
            "01_catenary_overall",
            "01 Reviewed catenary mast / overall",
            (10.0, -10.0, 5.0),
            target=tuple(centre.tolist()),
            object_prefix="CATENARY-MAST-",
        ),
        View(
            "02_catenary_opposite",
            "02 Reviewed catenary mast / opposite side",
            (-10.0, 10.0, 5.0),
            target=tuple(centre.tolist()),
            object_prefix="CATENARY-MAST-",
        ),
        View(
            "03_catenary_x_elevation",
            "03 Mast elevation / global X view",
            (14.0, 0.0, 0.0),
            target=tuple(centre.tolist()),
            object_prefix="CATENARY-MAST-",
        ),
        View(
            "04_catenary_y_elevation",
            "04 Mast elevation / global Y view",
            (0.0, 14.0, 0.0),
            target=tuple(centre.tolist()),
            object_prefix="CATENARY-MAST-",
        ),
        View(
            "05_catenary_top_section",
            "05 H-section / top plan",
            (0.0, 0.0, 18.0),
            target=tuple(centre.tolist()),
            up=(0.0, 1.0, 0.0),
            object_prefix="CATENARY-MAST-",
            crop_radius_xy=1.5,
        ),
        View(
            "06_catenary_foundation",
            "06 Mast-foundation contact",
            (4.0, -4.0, 2.0),
            target=(float(centre[0]), float(centre[1]), float(vertices[:, 2].min() + 0.4)),
            object_prefix="CATENARY-MAST-",
            crop_radius_xy=2.0,
        ),
    ]


def build_catenary_top_views(vertices: np.ndarray, triangles: list[Triangle]) -> list[View]:
    prefixes = (
        "SUPPLEMENTAL-CATENARY-MAST-016",
        "SUPPLEMENTAL-CATENARY-MAST-034",
        "SUPPLEMENTAL-CATENARY-MAST-024",
        "SUPPLEMENTAL-CATENARY-MAST-029",
        "SEG2100-CATENARY--CATENARY-MAST-0002",
        "SUPPLEMENTAL-CATENARY-MAST-030",
        "SUPPLEMENTAL-CATENARY-MAST-037",
    )
    selected_indexes = sorted(
        {
            index
            for triangle in triangles
            if triangle.object_name.startswith(prefixes)
            for index in triangle.indexes
        }
    )
    if not selected_indexes:
        raise ValueError("Catenary-top preset requires point-supported mast assemblies")
    selected = vertices[selected_indexes]
    centre = selected.mean(axis=0)
    along, cross = _horizontal_principal_axes(selected)
    up = np.asarray([0.0, 0.0, 1.0])
    station = (selected[:, :2] - centre[:2]) @ along[:2]
    length = max(float(np.ptp(station)), 50.0)
    prefix_text = "|".join(prefixes)
    first = _prefix_centre(vertices, triangles, ("SUPPLEMENTAL-CATENARY-MAST-016",))
    middle = _prefix_centre(vertices, triangles, ("SUPPLEMENTAL-CATENARY-MAST-029",))
    last = _prefix_centre(vertices, triangles, ("SUPPLEMENTAL-CATENARY-MAST-037",))
    return [
        View(
            "01_catenary_top_overall",
            "01 Seven point-supported mast assemblies / overall",
            tuple((-0.55 * length * along + 0.32 * length * cross + 0.18 * length * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=prefix_text,
        ),
        View(
            "02_catenary_top_reverse",
            "02 Seven assemblies / reverse side",
            tuple((0.55 * length * along - 0.32 * length * cross + 0.18 * length * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=prefix_text,
        ),
        View(
            "03_catenary_top_plan",
            "03 Mast rows and inward arm directions / top plan",
            tuple((0.75 * length * up).tolist()),
            target=tuple(centre.tolist()),
            up=tuple(along.tolist()),
            object_prefix=prefix_text,
        ),
        View(
            "04_catenary_top_first_detail",
            "04 First paired assemblies / fitted hardware detail",
            tuple((-8.0 * along + 9.0 * cross + 4.0 * up).tolist()),
            target=tuple(first.tolist()),
            object_prefix=prefix_text,
            crop_radius_xy=8.0,
        ),
        View(
            "05_catenary_top_middle_detail",
            "05 Middle paired assemblies / fitted hardware detail",
            tuple((8.0 * along - 9.0 * cross + 4.0 * up).tolist()),
            target=tuple(middle.tolist()),
            object_prefix=prefix_text,
            crop_radius_xy=8.0,
        ),
        View(
            "06_catenary_top_last_detail",
            "06 Last paired assemblies / fitted hardware detail",
            tuple((-7.0 * along + 8.0 * cross + 3.0 * up).tolist()),
            target=tuple(last.tolist()),
            object_prefix=prefix_text,
            crop_radius_xy=8.0,
        ),
    ]


def build_conductor_views(vertices: np.ndarray, triangles: list[Triangle]) -> list[View]:
    supplemental_prefix = "SUPPLEMENTAL-AUX-CONDUCTOR-"
    if any(item.object_name.startswith(supplemental_prefix) for item in triangles):
        wire_centre = _prefix_centre(vertices, triangles, (supplemental_prefix,))
        wire1 = _prefix_centre(vertices, triangles, (f"{supplemental_prefix}01",))
        wire2 = _prefix_centre(vertices, triangles, (f"{supplemental_prefix}02",))
        track_vertices_indexes = sorted(
            {
                index
                for triangle in triangles
                if triangle.object_name.startswith("TRACK")
                for index in triangle.indexes
            }
        )
        track_vertices = vertices[track_vertices_indexes]
        centred_xy = track_vertices[:, :2] - track_vertices[:, :2].mean(axis=0)
        _, _, vh = np.linalg.svd(centred_xy, full_matrices=False)
        along = _normalise(np.asarray([vh[0, 0], vh[0, 1], 0.0]))
        cross = np.asarray([-along[1], along[0], 0.0])
        up = np.asarray([0.0, 0.0, 1.0])
        return [
            View(
                "01_supplemental_conductor_overall",
                "01 Supplemental observed conductors / corridor context",
                tuple((-45.0 * along + 42.0 * cross + 25.0 * up).tolist()),
                target=tuple(wire_centre.tolist()),
                object_prefix=supplemental_prefix,
                roof_surface="TRACK",
            ),
            View(
                "02_supplemental_conductor_top",
                "02 Supplemental conductors / lateral alignment",
                (0.0, 0.0, 85.0),
                target=tuple(wire_centre.tolist()),
                up=tuple(along.tolist()),
                object_prefix=supplemental_prefix,
                roof_surface="TRACK",
            ),
            View(
                "03_supplemental_conductor_01",
                "03 Auxiliary conductor 01 / three support-constrained spans",
                tuple((-28.0 * along + 18.0 * cross + 11.0 * up).tolist()),
                target=tuple(wire1.tolist()),
                object_prefix=f"{supplemental_prefix}01",
            ),
            View(
                "04_supplemental_conductor_02",
                "04 Auxiliary conductor 02 / three support-constrained spans",
                tuple((25.0 * along - 16.0 * cross + 10.0 * up).tolist()),
                target=tuple(wire2.tolist()),
                object_prefix=f"{supplemental_prefix}02",
            ),
            View(
                "05_supplemental_conductor_side",
                "05 Supplemental conductors / sag elevation",
                tuple((48.0 * cross + 4.0 * up).tolist()),
                target=tuple(wire_centre.tolist()),
                object_prefix=supplemental_prefix,
                roof_surface="TRACK",
            ),
            View(
                "06_supplemental_conductor_end",
                "06 Supplemental conductors / track cross-section relationship",
                tuple((58.0 * along + 4.0 * up).tolist()),
                target=tuple(wire_centre.tolist()),
                object_prefix=supplemental_prefix,
                roof_surface="TRACK",
            ),
        ]
    wire_centre = _prefix_centre(vertices, triangles, ("CONTACT-WIRE-",))
    track_vertices_indexes = sorted(
        {
            index
            for triangle in triangles
            if triangle.object_name.startswith("TRACK-")
            for index in triangle.indexes
        }
    )
    track_vertices = vertices[track_vertices_indexes]
    centred_xy = track_vertices[:, :2] - track_vertices[:, :2].mean(axis=0)
    _, _, vh = np.linalg.svd(centred_xy, full_matrices=False)
    along = _normalise(np.asarray([vh[0, 0], vh[0, 1], 0.0]))
    cross = np.asarray([-along[1], along[0], 0.0])
    up = np.asarray([0.0, 0.0, 1.0])
    wire2 = _prefix_centre(vertices, triangles, ("CONTACT-WIRE-TRACK-0002",))
    wire3 = _prefix_centre(vertices, triangles, ("CONTACT-WIRE-TRACK-0003",))
    return [
        View(
            "01_conductor_overall",
            "01 Approved contact-wire fragments / accepted tracks",
            tuple((-45.0 * along + 42.0 * cross + 25.0 * up).tolist()),
            target=tuple(wire_centre.tolist()),
            object_prefix="CONTACT-WIRE-",
            roof_surface="TRACK-",
        ),
        View(
            "02_conductor_top",
            "02 Contact-wire lateral alignment / top plan",
            (0.0, 0.0, 85.0),
            target=tuple(wire_centre.tolist()),
            up=tuple(along.tolist()),
            object_prefix="CONTACT-WIRE-",
            roof_surface="TRACK-",
        ),
        View(
            "03_track_0002_wire",
            "03 TRACK-0002 / 40.29 m observed contact-wire span",
            tuple((-28.0 * along + 18.0 * cross + 11.0 * up).tolist()),
            target=tuple(wire2.tolist()),
            object_prefix="CONTACT-WIRE-TRACK-0002",
            roof_surface="TRACK-0002",
        ),
        View(
            "04_track_0003_wire",
            "04 TRACK-0003 / 24.71 m observed contact-wire span",
            tuple((25.0 * along - 16.0 * cross + 10.0 * up).tolist()),
            target=tuple(wire3.tolist()),
            object_prefix="CONTACT-WIRE-TRACK-0003",
            roof_surface="TRACK-0003",
        ),
        View(
            "05_conductor_side",
            "05 Contact-wire elevation / no unsupported gap fill",
            tuple((48.0 * cross + 4.0 * up).tolist()),
            target=tuple(wire_centre.tolist()),
            object_prefix="CONTACT-WIRE-",
            roof_surface="TRACK-",
        ),
        View(
            "06_conductor_end",
            "06 Contact-wire to track-centre relationship / end view",
            tuple((58.0 * along + 4.0 * up).tolist()),
            target=tuple(wire_centre.tolist()),
            object_prefix="CONTACT-WIRE-",
            roof_surface="TRACK-",
        ),
    ]


def build_observed_roof_views(vertices: np.ndarray, triangles: list[Triangle]) -> list[View]:
    prefix = "LEFT-CANOPY-ROOF-OBS-"
    roof_indexes = sorted(
        {
            index
            for triangle in triangles
            if triangle.object_name.startswith(prefix)
            for index in triangle.indexes
        }
    )
    if not roof_indexes:
        raise ValueError("Observed-roof preset requires LEFT-CANOPY-ROOF-OBS-* objects")
    roof_vertices = vertices[roof_indexes]
    centre = roof_vertices.mean(axis=0)
    centred_xy = roof_vertices[:, :2] - centre[:2]
    _, _, vh = np.linalg.svd(centred_xy, full_matrices=False)
    along = _normalise(np.asarray([vh[0, 0], vh[0, 1], 0.0]))
    cross = np.asarray([-along[1], along[0], 0.0])
    up = np.asarray([0.0, 0.0, 1.0])
    return [
        View(
            "01_observed_roof_track_context",
            "01 Opposite canopy main slope / accepted track context",
            tuple((-38.0 * along + 34.0 * cross + 18.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=prefix,
            roof_surface="TRACK-",
        ),
        View(
            "02_observed_roof_top",
            "02 Observed 50 m roof strip / top plan",
            tuple((70.0 * up).tolist()),
            target=tuple(centre.tolist()),
            up=tuple(along.tolist()),
            object_prefix=prefix,
        ),
        View(
            "03_observed_roof_long_side",
            "03 Long side / slab continuity and longitudinal fit",
            tuple((42.0 * cross + 6.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=prefix,
        ),
        View(
            "04_observed_roof_cross_section",
            "04 End section / reviewed transverse slope",
            tuple((42.0 * along + 3.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=prefix,
        ),
        View(
            "05_observed_roof_underside",
            "05 Underside / back-face and closed-slab check",
            tuple((-22.0 * cross - 14.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=prefix,
        ),
        View(
            "06_observed_roof_opposite",
            "06 Opposite oblique / no forced adjacent fascia",
            tuple((25.0 * along - 30.0 * cross + 12.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=prefix,
        ),
    ]


def build_opposite_platform_views(vertices: np.ndarray, triangles: list[Triangle]) -> list[View]:
    platform_prefix = "S2100_2150M-LEFT-PLATFORM-"
    platform_indexes = sorted(
        {
            index
            for triangle in triangles
            if triangle.object_name.startswith(platform_prefix)
            for index in triangle.indexes
        }
    )
    if not platform_indexes:
        raise ValueError(
            "Opposite-platform preset requires S2100_2150M-LEFT-PLATFORM-* objects"
        )
    platform_vertices = vertices[platform_indexes]
    centre = platform_vertices.mean(axis=0)
    centred_xy = platform_vertices[:, :2] - centre[:2]
    _, _, vh = np.linalg.svd(centred_xy, full_matrices=False)
    along = _normalise(np.asarray([vh[0, 0], vh[0, 1], 0.0]))
    cross = np.asarray([-along[1], along[0], 0.0])
    up = np.asarray([0.0, 0.0, 1.0])
    return [
        View(
            "01_opposite_platform_context",
            "01 Opposite platform / observed canopy and track context",
            tuple((-40.0 * along + 35.0 * cross + 18.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=platform_prefix,
            roof_surface="LEFT-CANOPY-ROOF-OBS-",
        ),
        View(
            "02_opposite_platform_top",
            "02 Segmented top / surveyed extent and boundaries",
            tuple((68.0 * up).tolist()),
            target=tuple(centre.tolist()),
            up=tuple(along.tolist()),
            object_prefix=platform_prefix,
        ),
        View(
            "03_opposite_platform_rail_edge",
            "03 Rail-side edge / clearance and longitudinal continuity",
            tuple((38.0 * cross + 6.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=platform_prefix,
            roof_surface="TRACK-0001",
        ),
        View(
            "04_opposite_platform_outer_edge",
            "04 Outer edge / closed volume and inferred thickness",
            tuple((-38.0 * cross + 5.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=platform_prefix,
        ),
        View(
            "05_opposite_platform_end",
            "05 End section / platform-track-roof vertical relationship",
            tuple((42.0 * along + 5.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=platform_prefix,
            roof_surface="LEFT-CANOPY-ROOF-OBS-",
        ),
        View(
            "06_opposite_platform_underside",
            "06 Underside / winding, closure and duplicate-face check",
            tuple((-22.0 * cross - 12.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=platform_prefix,
        ),
    ]


def build_platform_sign_views(vertices: np.ndarray, triangles: list[Triangle]) -> list[View]:
    right = "S2100_2150M-RIGHT-PLATFORM-SIGN-001"
    left = "S2100_2150M-LEFT-PLATFORM-SIGN-001"
    sign_prefixes = f"{right}|{left}"
    sign_vertices = vertices[
        sorted(
            {
                index
                for triangle in triangles
                if triangle.object_name.startswith((right, left))
                for index in triangle.indexes
            }
        )
    ]
    if not len(sign_vertices):
        raise ValueError("Platform-sign preset requires both photo-confirmed sign objects")
    centre = sign_vertices.mean(axis=0)
    right_centre = _prefix_centre(vertices, triangles, (right,))
    left_centre = _prefix_centre(vertices, triangles, (left,))
    along_xy = right_centre[:2] - left_centre[:2]
    if float(np.linalg.norm(along_xy)) < 1.0e-6:
        along_xy = np.asarray([1.0, 0.0])
    along = _normalise(np.asarray([along_xy[0], along_xy[1], 0.0]))
    cross = np.asarray([-along[1], along[0], 0.0])
    up = np.asarray([0.0, 0.0, 1.0])
    platform_context = "S2100_2150M-PLATFORM-001-|S2100_2150M-LEFT-PLATFORM-001-"
    return [
        View(
            "01_platform_signs_context",
            "01 Bilateral platform signs / independent evidence",
            tuple((-18.0 * along + 32.0 * cross + 14.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=sign_prefixes,
            roof_surface=platform_context,
        ),
        View(
            "02_right_sign_front",
            "02 Right sign / frame, panel and support contact",
            tuple((-1.0 * along + 9.0 * cross + 2.0 * up).tolist()),
            target=tuple(right_centre.tolist()),
            object_prefix=right,
            roof_surface="S2100_2150M-PLATFORM-001-",
            crop_radius_xy=5.0,
        ),
        View(
            "03_right_sign_side",
            "03 Right sign / thickness and closed volume",
            tuple((8.0 * along + 2.0 * cross + 1.5 * up).tolist()),
            target=tuple(right_centre.tolist()),
            object_prefix=right,
        ),
        View(
            "04_left_sign_front",
            "04 Left sign / independent reconstruction",
            tuple((1.0 * along - 9.0 * cross + 2.0 * up).tolist()),
            target=tuple(left_centre.tolist()),
            object_prefix=left,
            roof_surface="S2100_2150M-LEFT-PLATFORM-001-",
            crop_radius_xy=5.0,
        ),
        View(
            "05_left_sign_side",
            "05 Left sign / thickness and closed volume",
            tuple((-8.0 * along - 2.0 * cross + 1.5 * up).tolist()),
            target=tuple(left_centre.tolist()),
            object_prefix=left,
        ),
        View(
            "06_platform_signs_top",
            "06 Top plan / bilateral placement and platform interfaces",
            tuple((42.0 * up).tolist()),
            target=tuple(centre.tolist()),
            up=tuple(along.tolist()),
            object_prefix=sign_prefixes,
            roof_surface=platform_context,
        ),
    ]


def build_platform_fence_views(vertices: np.ndarray, triangles: list[Triangle]) -> list[View]:
    fence = "S2100_2150M-RIGHT-PLATFORM-OUTER-FENCE-001"
    fence_indexes = sorted(
        {
            index
            for triangle in triangles
            if triangle.object_name.startswith(fence)
            for index in triangle.indexes
        }
    )
    if not fence_indexes:
        raise ValueError("Platform-fence preset requires the right outer fence object")
    fence_vertices = vertices[fence_indexes]
    centre = fence_vertices.mean(axis=0)
    centred_xy = fence_vertices[:, :2] - centre[:2]
    _, _, vh = np.linalg.svd(centred_xy, full_matrices=False)
    along = _normalise(np.asarray([vh[0, 0], vh[0, 1], 0.0]))
    cross = np.asarray([-along[1], along[0], 0.0])
    up = np.asarray([0.0, 0.0, 1.0])
    platform = "S2100_2150M-PLATFORM-001-"
    return [
        View(
            "01_platform_fence_context",
            "01 Outer platform fence / 50 m clipped context",
            tuple((-35.0 * along + 26.0 * cross + 14.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=fence,
            roof_surface=platform,
        ),
        View(
            "02_platform_fence_front",
            "02 Long elevation / support continuity",
            tuple((32.0 * cross + 3.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=fence,
        ),
        View(
            "03_platform_fence_mid_detail",
            "03 Midspan / vertical members and horizontal rails",
            tuple((7.0 * cross + 1.5 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=fence,
            roof_surface=platform,
            crop_radius_xy=3.0,
        ),
        View(
            "04_platform_fence_top",
            "04 Top plan / outer-edge alignment",
            tuple((45.0 * up).tolist()),
            target=tuple(centre.tolist()),
            up=tuple(along.tolist()),
            object_prefix=fence,
            roof_surface=platform,
        ),
        View(
            "05_platform_fence_end",
            "05 End section / thickness and platform support",
            tuple((30.0 * along + 5.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=fence,
            roof_surface=platform,
            crop_radius_xy=3.0,
        ),
        View(
            "06_platform_fence_underside",
            "06 Low oblique / closed members and contact",
            tuple((-10.0 * cross - 3.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=fence,
        ),
    ]


def build_inner_column_views(vertices: np.ndarray, triangles: list[Triangle]) -> list[View]:
    prefix = "RIGHT-CANOPY-INNER-GRID-CORE-"
    roof = "RIGHT-CANOPY-ROOF-CANDIDATE-001-SURFACE-02-RUN-01"
    platform = "S2100_2150M-PLATFORM-001-"
    indexes = sorted(
        {
            index
            for triangle in triangles
            if triangle.object_name.startswith(prefix)
            for index in triangle.indexes
        }
    )
    if not indexes:
        raise ValueError("Inner-column preset requires the five recovered inner-row columns")
    row_vertices = vertices[indexes]
    centre = row_vertices.mean(axis=0)
    centred_xy = row_vertices[:, :2] - centre[:2]
    _, _, vh = np.linalg.svd(centred_xy, full_matrices=False)
    along = _normalise(np.asarray([vh[0, 0], vh[0, 1], 0.0]))
    cross = np.asarray([-along[1], along[0], 0.0])
    up = np.asarray([0.0, 0.0, 1.0])
    views = [
        View(
            "01_inner_column_row_context",
            "01 Recovered inner column row / platform-roof context",
            tuple((-30.0 * along + 24.0 * cross + 12.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=prefix,
            roof_surface=f"{roof}|{platform}",
        ),
        View(
            "02_inner_column_row_elevation",
            "02 Five-column elevation / 9 m topology",
            tuple((28.0 * cross + 3.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=prefix,
            roof_surface=roof,
        ),
        View(
            "03_inner_column_row_top",
            "03 Top plan / distinct inner and outer support rows",
            tuple((48.0 * up).tolist()),
            target=tuple(centre.tolist()),
            up=tuple(along.tolist()),
            object_prefix=prefix,
            roof_surface=roof,
        ),
    ]
    for index, suffix in enumerate(("001", "003", "005"), start=4):
        node = f"{prefix}{suffix}"
        node_centre = _prefix_centre(vertices, triangles, (node,))
        views.append(
            View(
                f"{index:02d}_inner_column_{suffix}_interface",
                f"{index:02d} Inner column {suffix} / base, capital and roof contact",
                tuple((7.0 * cross + 2.8 * up).tolist()),
                target=tuple(node_centre.tolist()),
                object_prefix=node,
                roof_surface=f"{roof}|{platform}",
                crop_radius_xy=3.4,
            )
        )
    return views


def build_left_column_views(vertices: np.ndarray, triangles: list[Triangle]) -> list[View]:
    prefix = "LEFT-CANOPY-GRID-CORE-"
    roof = "LEFT-CANOPY-ROOF-OBS-001-RUN-01"
    platform = "S2100_2150M-LEFT-PLATFORM-001-"
    indexes = sorted(
        {
            index
            for triangle in triangles
            if triangle.object_name.startswith(prefix)
            for index in triangle.indexes
        }
    )
    if not indexes:
        raise ValueError("Left-column preset requires five opposite-platform columns")
    row_vertices = vertices[indexes]
    centre = row_vertices.mean(axis=0)
    centred_xy = row_vertices[:, :2] - centre[:2]
    _, _, vh = np.linalg.svd(centred_xy, full_matrices=False)
    along = _normalise(np.asarray([vh[0, 0], vh[0, 1], 0.0]))
    cross = np.asarray([-along[1], along[0], 0.0])
    up = np.asarray([0.0, 0.0, 1.0])
    views = [
        View(
            "01_left_column_row_context",
            "01 Opposite-platform column row / independent evidence",
            tuple((-30.0 * along - 24.0 * cross + 12.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=prefix,
            roof_surface=f"{roof}|{platform}",
        ),
        View(
            "02_left_column_row_elevation",
            "02 Five-column elevation / 9 m topology",
            tuple((-28.0 * cross + 3.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=prefix,
            roof_surface=roof,
        ),
        View(
            "03_left_column_row_top",
            "03 Top plan / local-only roof boundary connectors",
            tuple((48.0 * up).tolist()),
            target=tuple(centre.tolist()),
            up=tuple(along.tolist()),
            object_prefix=prefix,
            roof_surface=roof,
        ),
    ]
    for index, suffix in enumerate(("001", "003", "005"), start=4):
        node = f"{prefix}{suffix}"
        node_centre = _prefix_centre(vertices, triangles, (node,))
        views.append(
            View(
                f"{index:02d}_left_column_{suffix}_interface",
                f"{index:02d} Left column {suffix} / base, capital and local connector",
                tuple((-7.0 * cross + 2.8 * up).tolist()),
                target=tuple(node_centre.tolist()),
                object_prefix=node,
                roof_surface=f"{roof}|{platform}",
                crop_radius_xy=3.4,
            )
        )
    return views


def build_endpoint_column_views(vertices: np.ndarray, triangles: list[Triangle]) -> list[View]:
    roots = (
        "SUPPLEMENTAL-RIGHT-CANOPY-END-COLUMN-START",
        "SUPPLEMENTAL-RIGHT-CANOPY-END-COLUMN-END",
        "SUPPLEMENTAL-OPPOSITE-CANOPY-END-COLUMN",
    )
    for root in roots:
        _prefix_centre(vertices, triangles, (root,))
    track_indexes = sorted(
        {
            index
            for triangle in triangles
            if triangle.object_name.startswith("TRACK")
            for index in triangle.indexes
        }
    )
    track_vertices = vertices[track_indexes]
    centred_xy = track_vertices[:, :2] - track_vertices[:, :2].mean(axis=0)
    _, _, vh = np.linalg.svd(centred_xy, full_matrices=False)
    along = _normalise(np.asarray([vh[0, 0], vh[0, 1], 0.0]))
    cross = np.asarray([-along[1], along[0], 0.0])
    up = np.asarray([0.0, 0.0, 1.0])
    roof_context = (
        "SEG2050-CANOPY|SEG2150-CANOPY|S2050-OPPOSITE-CANOPY-ROOF|"
        "S2100-OPPOSITE-CANOPY-ROOF|S2150-OPPOSITE-CANOPY-ROOF"
    )
    views: list[View] = []
    for index, root in enumerate(roots, start=1):
        centre = _prefix_centre(vertices, triangles, (root,))
        label = root.removeprefix("SUPPLEMENTAL-").replace("-", " ").title()
        views.extend(
            [
                View(
                    f"{index * 2 - 1:02d}_{root.lower()}_side",
                    f"{index * 2 - 1:02d} {label} / row elevation and roof contact",
                    tuple((7.0 * cross + 2.8 * up).tolist()),
                    target=tuple(centre.tolist()),
                    object_prefix=root,
                    roof_surface=roof_context,
                    crop_radius_xy=3.4,
                ),
                View(
                    f"{index * 2:02d}_{root.lower()}_oblique",
                    f"{index * 2:02d} {label} / base, section and capital",
                    tuple((-5.0 * along + 5.0 * cross + 3.2 * up).tolist()),
                    target=tuple(centre.tolist()),
                    object_prefix=root,
                    roof_surface=roof_context,
                    crop_radius_xy=3.4,
                ),
            ]
        )
    return views


def build_lateral_refit_column_views(
    vertices: np.ndarray,
    triangles: list[Triangle],
) -> list[View]:
    opposite_prefix = (
        "S2050-OPPOSITE-CANOPY-COLUMN-|"
        "S2100-OPPOSITE-CANOPY-COLUMN-|"
        "S2150-OPPOSITE-CANOPY-COLUMN-"
    )
    right_column = "SUPPLEMENTAL-RIGHT-CANOPY-COLUMN-006"
    opposite_roofs = (
        "S2050-OPPOSITE-CANOPY-ROOF-|"
        "S2100-OPPOSITE-CANOPY-ROOF-|"
        "S2150-OPPOSITE-CANOPY-ROOF-"
    )
    row_prefixes = tuple(opposite_prefix.split("|"))
    indexes = sorted(
        {
            index
            for triangle in triangles
            if triangle.object_name.startswith(row_prefixes)
            for index in triangle.indexes
        }
    )
    if not indexes:
        raise ValueError("Lateral-refit preset requires opposite-platform columns")
    _prefix_centre(vertices, triangles, (right_column,))
    row_vertices = vertices[indexes]
    centre = row_vertices.mean(axis=0)
    centred_xy = row_vertices[:, :2] - centre[:2]
    _, _, vh = np.linalg.svd(centred_xy, full_matrices=False)
    along = _normalise(np.asarray([vh[0, 0], vh[0, 1], 0.0]))
    cross = np.asarray([-along[1], along[0], 0.0])
    up = np.asarray([0.0, 0.0, 1.0])
    representative_nodes = (
        "S2050-OPPOSITE-CANOPY-COLUMN-01",
        "S2100-OPPOSITE-CANOPY-COLUMN-03",
    )
    views = [
        View(
            "01_lateral_refit_row_context",
            "01 Refit opposite column row / platform and roof context",
            tuple((-35.0 * along - 28.0 * cross + 13.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=opposite_prefix,
            roof_surface=opposite_roofs,
        ),
        View(
            "02_lateral_refit_row_elevation",
            "02 Refit column-row elevation / longitudinal positions retained",
            tuple((-32.0 * cross + 3.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=opposite_prefix,
            roof_surface=opposite_roofs,
        ),
        View(
            "03_lateral_refit_row_top",
            "03 Top plan / lateral centres and roof relation",
            tuple((52.0 * up).tolist()),
            target=tuple(centre.tolist()),
            up=tuple(along.tolist()),
            object_prefix=opposite_prefix,
            roof_surface=opposite_roofs,
        ),
    ]
    for index, node in enumerate(representative_nodes, start=4):
        node_centre = _prefix_centre(vertices, triangles, (node,))
        views.append(
            View(
                f"{index:02d}_{node.lower()}_interface",
                f"{index:02d} {node} / shaft, capital and local under-roof",
                tuple((-7.0 * cross + 2.8 * up).tolist()),
                target=tuple(node_centre.tolist()),
                object_prefix=node,
                roof_surface=opposite_roofs,
                crop_radius_xy=3.4,
            )
        )
    right_centre = _prefix_centre(vertices, triangles, (right_column,))
    views.append(
        View(
            "06_right_supplemental_column_interface",
            "06 Right supplemental column / lateral refit and roof contact",
            tuple((7.0 * cross + 2.8 * up).tolist()),
            target=tuple(right_centre.tolist()),
            object_prefix=right_column,
            roof_surface="SEG2100-CANOPY|SEG2050-CANOPY|SEG2150-CANOPY",
            crop_radius_xy=3.4,
        )
    )
    return views


def build_gap_mast_views(
    vertices: np.ndarray,
    triangles: list[Triangle],
) -> list[View]:
    roots = (
        "SUPPLEMENTAL-CATENARY-MAST-0010",
        "SUPPLEMENTAL-CATENARY-MAST-0011",
    )
    for root in roots:
        _prefix_centre(vertices, triangles, (root,))
    track_indexes = sorted(
        {
            index
            for triangle in triangles
            if triangle.object_name.startswith("TRACKGRAPH--TRACK-")
            for index in triangle.indexes
        }
    )
    if not track_indexes:
        raise ValueError("Gap-mast preset requires track context")
    track_vertices = vertices[track_indexes]
    centred_xy = track_vertices[:, :2] - track_vertices[:, :2].mean(axis=0)
    _, _, vh = np.linalg.svd(centred_xy, full_matrices=False)
    along = _normalise(np.asarray([vh[0, 0], vh[0, 1], 0.0]))
    cross = np.asarray([-along[1], along[0], 0.0])
    up = np.asarray([0.0, 0.0, 1.0])
    local_context = (
        "TRACKGRAPH--TRACK-|SEG2050-CATENARY--|SEG2100-CATENARY--|"
        "SEG2150-CATENARY--|S2050-OPPOSITE-|S2100-OPPOSITE-|S2150-OPPOSITE-"
    )
    views: list[View] = []
    for sequence, root in enumerate(roots):
        centre = _prefix_centre(vertices, triangles, (root,))
        first = sequence * 3 + 1
        views.extend(
            [
                View(
                    f"{first:02d}_{root.lower()}_elevation",
                    f"{first:02d} {root} / full-height railway-side elevation",
                    tuple((8.0 * cross + 2.5 * up).tolist()),
                    target=tuple(centre.tolist()),
                    object_prefix=root,
                    roof_surface=local_context,
                    crop_radius_xy=5.5,
                ),
                View(
                    f"{first + 1:02d}_{root.lower()}_oblique",
                    f"{first + 1:02d} {root} / section and local clearance",
                    tuple((-6.0 * along + 6.0 * cross + 3.5 * up).tolist()),
                    target=tuple(centre.tolist()),
                    object_prefix=root,
                    roof_surface=local_context,
                    crop_radius_xy=5.5,
                ),
                View(
                    f"{first + 2:02d}_{root.lower()}_top",
                    f"{first + 2:02d} {root} / top plan and track relation",
                    tuple((13.0 * up).tolist()),
                    target=tuple(centre.tolist()),
                    up=tuple(along.tolist()),
                    object_prefix=root,
                    roof_surface=local_context,
                    crop_radius_xy=5.5,
                ),
            ]
        )
    return views


def build_gap_canopy_support_views(
    vertices: np.ndarray,
    triangles: list[Triangle],
) -> list[View]:
    root = "SUPPLEMENTAL-OPPOSITE-CANOPY-OUTER-COLUMN-001"
    centre = _prefix_centre(vertices, triangles, (root,))
    indexes = sorted(
        {
            index
            for triangle in triangles
            if triangle.object_name.startswith(root)
            for index in triangle.indexes
        }
    )
    if not indexes:
        raise ValueError("Gap-canopy-support preset requires the supplemental support")
    support_vertices = vertices[indexes]
    track_indexes = sorted(
        {
            index
            for triangle in triangles
            if triangle.object_name.startswith("TRACKGRAPH--TRACK-")
            for index in triangle.indexes
        }
    )
    track_vertices = vertices[track_indexes]
    centred_xy = track_vertices[:, :2] - track_vertices[:, :2].mean(axis=0)
    _, _, vh = np.linalg.svd(centred_xy, full_matrices=False)
    along = _normalise(np.asarray([vh[0, 0], vh[0, 1], 0.0]))
    cross = np.asarray([-along[1], along[0], 0.0])
    up = np.asarray([0.0, 0.0, 1.0])
    context = (
        "S2150-OPPOSITE-PLATFORM-|S2150-OPPOSITE-CANOPY-ROOF-|"
        "S2150-OPPOSITE-CANOPY-COLUMN-"
    )
    base_target = centre.copy()
    base_target[2] = float(np.min(support_vertices[:, 2]) + 0.8)
    roof_target = centre.copy()
    roof_target[2] = float(np.max(support_vertices[:, 2]) - 0.35)
    return [
        View(
            "01_gap_canopy_support_context",
            "01 Photo-confirmed outer support / same-station column pair",
            tuple((-14.0 * along - 14.0 * cross + 7.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=12.0,
        ),
        View(
            "02_gap_canopy_support_elevation",
            "02 Outer support elevation / platform-to-roof continuity",
            tuple((-8.0 * cross + 2.5 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=4.5,
        ),
        View(
            "03_gap_canopy_support_oblique",
            "03 Outer support oblique / shaft, capital and connector",
            tuple((-5.0 * along - 5.0 * cross + 3.2 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=4.5,
        ),
        View(
            "04_gap_canopy_support_top",
            "04 Top plan / inner-outer row alignment",
            tuple((14.0 * up).tolist()),
            target=tuple(centre.tolist()),
            up=tuple(along.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=5.0,
        ),
        View(
            "05_gap_canopy_support_base",
            "05 Base interface / observed platform landing",
            tuple((-4.0 * along - 4.0 * cross + 1.5 * up).tolist()),
            target=tuple(base_target.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=2.5,
        ),
        View(
            "06_gap_canopy_support_roof",
            "06 Roof interface / capital and local under-roof closure",
            tuple((3.5 * along - 4.0 * cross - 0.8 * up).tolist()),
            target=tuple(roof_target.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=2.8,
        ),
    ]


def build_station_entry_views(
    vertices: np.ndarray,
    triangles: list[Triangle],
) -> list[View]:
    root = "SUPPLEMENTAL-STATION-ENTRY-01-"
    centre = _prefix_centre(vertices, triangles, (root,))
    track_indexes = sorted(
        {
            index
            for triangle in triangles
            if triangle.object_name.startswith("TRACKGRAPH--TRACK-")
            for index in triangle.indexes
        }
    )
    if not track_indexes:
        raise ValueError("Station-entry preset requires track context")
    track_vertices = vertices[track_indexes]
    centred_xy = track_vertices[:, :2] - track_vertices[:, :2].mean(axis=0)
    _, _, vh = np.linalg.svd(centred_xy, full_matrices=False)
    along = _normalise(np.asarray([vh[0, 0], vh[0, 1], 0.0]))
    cross = np.asarray([-along[1], along[0], 0.0])
    up = np.asarray([0.0, 0.0, 1.0])
    context = (
        "SEG2150-PLATFORM--|SEG2150-CANOPY--|S2150-RIGHT-PLATFORM-|"
        "S2150-RIGHT-CANOPY-|TRACKGRAPH--TRACK-|SUPPLEMENTAL-STATION-ENTRY-01-"
    )
    door_centre = _prefix_centre(
        vertices,
        triangles,
        ("SUPPLEMENTAL-STATION-ENTRY-01-BLUE-DOOR-",),
    )
    glazing_centre = _prefix_centre(
        vertices,
        triangles,
        (
            "SUPPLEMENTAL-STATION-ENTRY-01-GLASS-",
            "SUPPLEMENTAL-STATION-ENTRY-01-OUTER-GLASS-",
        ),
    )
    return [
        View(
            "01_station_entry_context",
            "01 Supplemental station entry / railway-side context",
            tuple((-20.0 * along - 22.0 * cross + 10.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=28.0,
        ),
        View(
            "02_station_entry_railway_elevation",
            "02 Railway-side elevation / doors, cladding and sloped glazing",
            tuple((-18.0 * cross + 3.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=17.0,
        ),
        View(
            "03_station_entry_outer_elevation",
            "03 Outer elevation / observed rear-side surfaces",
            tuple((18.0 * cross + 3.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=17.0,
        ),
        View(
            "04_station_entry_top",
            "04 Top plan / two observed facade bands and roof limits",
            tuple((25.0 * up).tolist()),
            target=tuple(centre.tolist()),
            up=tuple(along.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=18.0,
        ),
        View(
            "05_station_entry_doors",
            "05 Photo-interpreted blue doors / no rear-volume inference",
            tuple((-10.0 * cross + 1.2 * up).tolist()),
            target=tuple(door_centre.tolist()),
            object_prefix="SUPPLEMENTAL-STATION-ENTRY-01-BLUE-DOOR-",
            roof_surface="SUPPLEMENTAL-STATION-ENTRY-01-UPPER-CLADDING|SUPPLEMENTAL-STATION-ENTRY-01-END-WALL",
            crop_radius_xy=7.0,
        ),
        View(
            "06_station_entry_glazing",
            "06 Sloped glazed enclosure / segmented observed outline",
            tuple((-8.0 * along - 10.0 * cross + 4.0 * up).tolist()),
            target=tuple(glazing_centre.tolist()),
            object_prefix="SUPPLEMENTAL-STATION-ENTRY-01-GLASS-|SUPPLEMENTAL-STATION-ENTRY-01-OUTER-GLASS-",
            roof_surface="SUPPLEMENTAL-STATION-ENTRY-01-SLOPED-ROOF",
            crop_radius_xy=9.0,
        ),
    ]


def build_station_end_wall_views(
    vertices: np.ndarray,
    triangles: list[Triangle],
) -> list[View]:
    root = "SUPPLEMENTAL-STATION-END-WALL-01"
    centre = _prefix_centre(vertices, triangles, (root,))
    track_indexes = sorted(
        {
            index
            for triangle in triangles
            if triangle.object_name.startswith("TRACKGRAPH--TRACK-")
            for index in triangle.indexes
        }
    )
    if not track_indexes:
        raise ValueError("Station-end-wall preset requires track context")
    track_vertices = vertices[track_indexes]
    centred_xy = track_vertices[:, :2] - track_vertices[:, :2].mean(axis=0)
    _, _, vh = np.linalg.svd(centred_xy, full_matrices=False)
    along = _normalise(np.asarray([vh[0, 0], vh[0, 1], 0.0]))
    cross = np.asarray([-along[1], along[0], 0.0])
    up = np.asarray([0.0, 0.0, 1.0])
    context = (
        "SEG2150-PLATFORM--|SEG2150-CANOPY--|S2150-RIGHT-PLATFORM-|"
        "S2150-RIGHT-CANOPY-|TRACKGRAPH--TRACK-|SUPPLEMENTAL-STATION-ENTRY-01-|"
        "SUPPLEMENTAL-STATION-END-WALL-01"
    )
    lower = centre.copy()
    lower[2] -= 1.4
    upper = centre.copy()
    upper[2] += 1.9
    return [
        View(
            "01_station_end_wall_context",
            "01 Observed station end wall / railway and entrance context",
            tuple((-20.0 * along - 20.0 * cross + 9.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=26.0,
        ),
        View(
            "02_station_end_wall_front",
            "02 End-wall front / dense-cloud fitted visible tiled face",
            tuple((-16.0 * along + 1.5 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=10.0,
        ),
        View(
            "03_station_end_wall_reverse",
            "03 Reverse face / explicit thickness and no inferred volume",
            tuple((16.0 * along + 1.5 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=10.0,
        ),
        View(
            "04_station_end_wall_top",
            "04 Top plan / fitted station plane and cross extent",
            tuple((18.0 * up).tolist()),
            target=tuple(centre.tolist()),
            up=tuple(cross.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=10.0,
        ),
        View(
            "05_station_end_wall_base",
            "05 Lower interface / observed ground-side wall limit",
            tuple((-7.0 * along - 4.0 * cross + 1.5 * up).tolist()),
            target=tuple(lower.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=6.0,
        ),
        View(
            "06_station_end_wall_top_limit",
            "06 Upper limit / wall stops below unsupported roof overhang",
            tuple((-7.0 * along - 4.0 * cross + 2.0 * up).tolist()),
            target=tuple(upper.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=6.0,
        ),
    ]


def build_platform_clock_views(
    vertices: np.ndarray,
    triangles: list[Triangle],
) -> list[View]:
    root = "SUPPLEMENTAL-OPPOSITE-PLATFORM-CLOCK-001"
    centre = _prefix_centre(vertices, triangles, (root,))
    track_indexes = sorted(
        {
            index
            for triangle in triangles
            if triangle.object_name.startswith("TRACKGRAPH--TRACK-")
            for index in triangle.indexes
        }
    )
    if not track_indexes:
        raise ValueError("Platform-clock preset requires track context")
    track_vertices = vertices[track_indexes]
    centred_xy = track_vertices[:, :2] - track_vertices[:, :2].mean(axis=0)
    _, _, vh = np.linalg.svd(centred_xy, full_matrices=False)
    along = _normalise(np.asarray([vh[0, 0], vh[0, 1], 0.0]))
    cross = np.asarray([-along[1], along[0], 0.0])
    up = np.asarray([0.0, 0.0, 1.0])
    context = (
        "S2100-OPPOSITE-CANOPY-|S2100-OPPOSITE-PLATFORM-|TRACKGRAPH--TRACK-|"
        "SUPPLEMENTAL-OPPOSITE-PLATFORM-CLOCK-001"
    )
    hanger_target = centre.copy()
    hanger_target[2] += 0.45
    return [
        View(
            "01_platform_clock_context",
            "01 Opposite-platform clock / canopy and track context",
            tuple((-10.0 * along - 12.0 * cross + 5.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=13.0,
        ),
        View(
            "02_platform_clock_front",
            "02 Clock front / observed round point-cloud fit",
            tuple((-5.0 * along + 0.2 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=3.0,
        ),
        View(
            "03_platform_clock_reverse",
            "03 Clock reverse / explicit disc thickness",
            tuple((5.0 * along + 0.2 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=3.0,
        ),
        View(
            "04_platform_clock_side",
            "04 Side profile / disc and vertical hanger",
            tuple((-5.0 * cross + 0.5 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=3.0,
        ),
        View(
            "05_platform_clock_top",
            "05 Top view / fitted station plane and thickness",
            tuple((6.0 * up).tolist()),
            target=tuple(centre.tolist()),
            up=tuple(along.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=3.0,
        ),
        View(
            "06_platform_clock_hanger",
            "06 Hanger interface / conservative canopy connection",
            tuple((-3.0 * along - 3.0 * cross + 0.5 * up).tolist()),
            target=tuple(hanger_target.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=2.5,
        ),
    ]


def build_remote_context_facade_views(
    vertices: np.ndarray,
    triangles: list[Triangle],
) -> list[View]:
    root = "SUPPLEMENTAL-REMOTE-CONTEXT-FACADE-001"
    centre = _prefix_centre(vertices, triangles, (root,))
    track_indexes = sorted(
        {
            index
            for triangle in triangles
            if triangle.object_name.startswith("TRACKGRAPH--TRACK-")
            for index in triangle.indexes
        }
    )
    if not track_indexes:
        raise ValueError("Remote-facade preset requires track context")
    track_vertices = vertices[track_indexes]
    centred_xy = track_vertices[:, :2] - track_vertices[:, :2].mean(axis=0)
    _, _, vh = np.linalg.svd(centred_xy, full_matrices=False)
    along = _normalise(np.asarray([vh[0, 0], vh[0, 1], 0.0]))
    cross = np.asarray([-along[1], along[0], 0.0])
    up = np.asarray([0.0, 0.0, 1.0])
    context = (
        "SUPPLEMENTAL-REMOTE-CONTEXT-FACADE-001|SUPPLEMENTAL-STATION-ENTRY-|"
        "S2100-RIGHT-PLATFORM-|TRACKGRAPH--TRACK-"
    )
    return [
        View(
            "01_remote_facade_context",
            "01 Observed remote facade / station and track context",
            tuple((-15.0 * along - 28.0 * cross + 9.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=32.0,
        ),
        View(
            "02_remote_facade_front",
            "02 Railway-facing elevation / dense cloud plane fit",
            tuple((-10.0 * cross + 1.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=7.0,
        ),
        View(
            "03_remote_facade_reverse",
            "03 Reverse face / explicit thin UE-safe closure",
            tuple((10.0 * cross + 1.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=7.0,
        ),
        View(
            "04_remote_facade_side",
            "04 Side profile / observed-plane thickness",
            tuple((-9.0 * along + 1.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=7.0,
        ),
        View(
            "05_remote_facade_top",
            "05 Top view / station extent and cross-plane position",
            tuple((10.0 * up).tolist()),
            target=tuple(centre.tolist()),
            up=tuple(along.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=7.0,
        ),
        View(
            "06_remote_facade_lower_edge",
            "06 Lower edge / conservative observed surface limit",
            tuple((-7.0 * along - 7.0 * cross + 2.0 * up).tolist()),
            target=tuple((centre - 2.0 * up).tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=7.0,
        ),
    ]


def build_boundary_conductor_views(
    vertices: np.ndarray,
    triangles: list[Triangle],
) -> list[View]:
    root = "SUPPLEMENTAL-BOUNDARY-CONDUCTOR-"
    centre = _prefix_centre(vertices, triangles, (root,))
    track_indexes = sorted(
        {
            index
            for triangle in triangles
            if triangle.object_name.startswith("TRACKGRAPH--TRACK-")
            for index in triangle.indexes
        }
    )
    if not track_indexes:
        raise ValueError("Boundary-conductor preset requires track context")
    track_vertices = vertices[track_indexes]
    centred_xy = track_vertices[:, :2] - track_vertices[:, :2].mean(axis=0)
    _, _, vh = np.linalg.svd(centred_xy, full_matrices=False)
    along = _normalise(np.asarray([vh[0, 0], vh[0, 1], 0.0]))
    cross = np.asarray([-along[1], along[0], 0.0])
    up = np.asarray([0.0, 0.0, 1.0])
    line_01 = _prefix_centre(vertices, triangles, (f"{root}01",))
    line_02 = _prefix_centre(vertices, triangles, (f"{root}02",))
    line_03 = _prefix_centre(vertices, triangles, (f"{root}03",))
    context = (
        "TRACKGRAPH--TRACK-|SUPPLEMENTAL-CATENARY-MAST-030|"
        "SUPPLEMENTAL-CATENARY-MAST-037"
    )
    return [
        View(
            "01_boundary_conductors_context",
            "01 Outbound conductors / terminal support and track context",
            tuple((-15.0 * along + 16.0 * cross + 8.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=18.0,
        ),
        View(
            "02_boundary_conductors_top",
            "02 Top view / three evidence-separated lines",
            tuple((20.0 * up).tolist()),
            target=tuple(centre.tolist()),
            up=tuple(along.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=18.0,
        ),
        View(
            "03_boundary_conductors_side",
            "03 Side elevation / observed short-fragment curvature",
            tuple((18.0 * cross + 2.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=18.0,
        ),
        View(
            "04_boundary_conductor_01",
            "04 Boundary conductor 01 / mast 037 anchor",
            tuple((-6.0 * along + 7.0 * cross + 3.0 * up).tolist()),
            target=tuple(line_01.tolist()),
            object_prefix=f"{root}01",
            roof_surface=context,
            crop_radius_xy=9.0,
        ),
        View(
            "05_boundary_conductors_02_03",
            "05 Boundary conductors 02+03 / mast 030 pair",
            tuple((-6.0 * along - 7.0 * cross + 3.0 * up).tolist()),
            target=tuple(((line_02 + line_03) / 2.0).tolist()),
            object_prefix=f"{root}02|{root}03",
            roof_surface=context,
            crop_radius_xy=9.0,
        ),
        View(
            "06_boundary_conductors_end",
            "06 Segment-end view / explicit continuation boundary",
            tuple((10.0 * along + 2.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=root,
            roof_surface=context,
            crop_radius_xy=18.0,
        ),
    ]


def build_adjacent_handoff_views(
    vertices: np.ndarray,
    triangles: list[Triangle],
) -> list[View]:
    current = "SUPPLEMENTAL-BOUNDARY-CONDUCTOR-"
    next_root = "ADJACENT-NEXT-BOUNDARY-CONDUCTOR-"
    previous = "ADJACENT-PREVIOUS-BOUNDARY-CONDUCTOR-"
    right_start = "SUPPLEMENTAL-RIGHT-CANOPY-END-COLUMN-START"
    right_end = "SUPPLEMENTAL-RIGHT-CANOPY-END-COLUMN-END"
    opposite_end = "SUPPLEMENTAL-OPPOSITE-CANOPY-END-COLUMN"
    observed_surfaces = "ADJACENT-NEXT-RIGHT-|ADJACENT-NEXT-OPPOSITE-"
    all_added = (
        f"{next_root}|{previous}|{right_start}|{right_end}|{opposite_end}|"
        f"{observed_surfaces}"
    )
    next_pair = (
        f"{current}|{next_root}|{right_end}|{opposite_end}|{observed_surfaces}"
    )
    centre = _prefix_centre(
        vertices,
        triangles,
        (current, next_root, right_end, opposite_end, "ADJACENT-NEXT-RIGHT-", "ADJACENT-NEXT-OPPOSITE-"),
    )
    previous_centre = _prefix_centre(
        vertices, triangles, (previous, right_start)
    )
    right_centre = _prefix_centre(vertices, triangles, (right_end,))
    opposite_centre = _prefix_centre(vertices, triangles, (opposite_end,))
    track_indexes = sorted(
        {
            index
            for triangle in triangles
            if triangle.object_name.startswith("TRACKGRAPH--TRACK-")
            for index in triangle.indexes
        }
    )
    if len(track_indexes) >= 2:
        track = vertices[np.asarray(track_indexes, dtype=np.int64)]
        covariance = np.cov(track[:, :2].T)
        values, vectors = np.linalg.eigh(covariance)
        along = np.asarray([*vectors[:, int(np.argmax(values))], 0.0])
    else:
        along = np.asarray([1.0, 0.0, 0.0])
    cross = np.asarray([-along[1], along[0], 0.0])
    up = np.asarray([0.0, 0.0, 1.0])
    context = (
        "TRACKGRAPH--TRACK-|CANOPY-ROOF-|SUPPLEMENTAL-CATENARY-MAST-|"
        "TRACK--|TRACK-|STATION--|S2200_2250M-PLATFORM-|CATENARY--|CONDUCTOR--|"
        "ADJACENT-NEXT-"
    )
    return [
        View(
            "01_adjacent_handoff_overview",
            "01 Adjacent handoff / supported tracks, surfaces and conductors",
            tuple((-20.0 * along + 24.0 * cross + 13.0 * up).tolist()),
            target=tuple(((centre + previous_centre) / 2.0).tolist()),
            object_prefix=all_added,
            roof_surface=context,
            crop_radius_xy=45.0,
        ),
        View(
            "02_next_boundary_top",
            "02 Next boundary / current and evidence-owned continuation",
            tuple((24.0 * up).tolist()),
            target=tuple(centre.tolist()),
            up=tuple(along.tolist()),
            object_prefix=next_pair,
            roof_surface=context,
            crop_radius_xy=18.0,
        ),
        View(
            "03_next_boundary_side",
            "03 Next boundary / platform, roof and conductor seams",
            tuple((18.0 * cross + 4.0 * up).tolist()),
            target=tuple(centre.tolist()),
            object_prefix=next_pair,
            roof_surface=context,
            crop_radius_xy=18.0,
        ),
        View(
            "04_right_endpoint_column",
            "04 Right canopy / next grid column",
            tuple((-7.0 * along + 7.0 * cross + 3.0 * up).tolist()),
            target=tuple(right_centre.tolist()),
            object_prefix=right_end,
            roof_surface=context,
            crop_radius_xy=8.0,
        ),
        View(
            "05_opposite_endpoint_column",
            "05 Opposite canopy / next grid column",
            tuple((-7.0 * along - 7.0 * cross + 3.0 * up).tolist()),
            target=tuple(opposite_centre.tolist()),
            object_prefix=opposite_end,
            roof_surface=context,
            crop_radius_xy=8.0,
        ),
        View(
            "06_previous_boundary_evidence",
            "06 Previous boundary / owned fragment without invented bridge",
            tuple((8.0 * along + 9.0 * cross + 4.0 * up).tolist()),
            target=tuple(previous_centre.tolist()),
            object_prefix=f"{previous}|{right_start}",
            roof_surface=context,
            crop_radius_xy=12.0,
        ),
    ]


def build_montage(view_records: list[dict[str, object]], output: Path) -> None:
    images = [Image.open(str(record["output"])).convert("RGB") for record in view_records]
    thumb_width = 800
    thumb_height = 450
    canvas = Image.new("RGB", (thumb_width * 2, thumb_height * 3), BACKGROUND)
    for index, item in enumerate(images):
        thumb = item.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        canvas.paste(thumb, ((index % 2) * thumb_width, (index // 2) * thumb_height))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--preset",
        choices=(
            "canopy",
            "track",
            "track_boundary",
            "catenary",
            "catenary_top",
            "conductor",
            "observed_roof",
            "opposite_platform",
            "platform_signs",
            "platform_fence",
            "inner_columns",
            "left_columns",
            "endpoint_columns",
            "lateral_refit_columns",
            "gap_masts",
            "gap_canopy_support",
            "station_entry",
            "station_end_wall",
            "platform_clock",
            "remote_context_facade",
            "boundary_conductor",
            "adjacent_handoff",
            "corridor",
        ),
        default="canopy",
    )
    args = parser.parse_args()
    vertices, triangles, colours = parse_obj(args.obj.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    view_builders = {
        "canopy": build_views,
        "track": build_track_views,
        "track_boundary": build_track_boundary_views,
        "catenary": build_catenary_views,
        "catenary_top": build_catenary_top_views,
        "conductor": build_conductor_views,
        "observed_roof": build_observed_roof_views,
        "opposite_platform": build_opposite_platform_views,
        "platform_signs": build_platform_sign_views,
        "platform_fence": build_platform_fence_views,
        "inner_columns": build_inner_column_views,
        "left_columns": build_left_column_views,
        "endpoint_columns": build_endpoint_column_views,
        "lateral_refit_columns": build_lateral_refit_column_views,
        "gap_masts": build_gap_mast_views,
        "gap_canopy_support": build_gap_canopy_support_views,
        "station_entry": build_station_entry_views,
        "station_end_wall": build_station_end_wall_views,
        "platform_clock": build_platform_clock_views,
        "remote_context_facade": build_remote_context_facade_views,
        "boundary_conductor": build_boundary_conductor_views,
        "adjacent_handoff": build_adjacent_handoff_views,
        "corridor": build_corridor_views,
    }
    views = view_builders[args.preset](vertices, triangles)
    for view in views:
        output = args.output_dir / f"{view.name}.png"
        records.append(render_view(vertices, triangles, colours, view, output))
    montage = args.output_dir / f"{args.preset}_fixed_views_montage.jpg"
    build_montage(records, montage)
    manifest = {
        "schema_version": "railway.obj-fixed-view-review.v1",
        "source_obj": str(args.obj.resolve()),
        "renderer": "deterministic_python_backface_culled",
        "preset": args.preset,
        "view_count": len(records),
        "views": records,
        "montage": str(montage),
        "limitations": [
            "This renderer checks mesh visibility and gross interfaces, not PBR materials.",
            "Final UE acceptance remains required before production delivery.",
        ],
    }
    manifest_path = args.output_dir / "fixed_view_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
