from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ProjectConfig
from ..io import load_json, write_json
from ..registry import new_registry, summarize_registry, validate_registry_value
from .mesh import ObjWriter

MATERIAL_BY_EVIDENCE = {
    "observed": "Observed",
    "photo_interpreted": "PhotoInterpreted",
    "rule_inferred": "RuleInferred",
    "unsupported": "Unsupported",
}


def _box(center: list[float], size: list[float]) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    c = np.asarray(center, dtype=np.float64)
    half = np.asarray(size, dtype=np.float64) / 2.0
    if c.shape != (3,) or half.shape != (3,) or np.any(half <= 0):
        raise ValueError("box geometry needs positive three-value center and size")
    vertices = np.asarray(
        [
            c + half * signs
            for signs in (
                (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
            )
        ]
    )
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return vertices, faces


def _beam(
    start: list[float], end: list[float], width: float, height: float
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    a = np.asarray(start, dtype=np.float64)
    b = np.asarray(end, dtype=np.float64)
    tangent = b - a
    length = float(np.linalg.norm(tangent))
    if length <= 1e-9 or width <= 0 or height <= 0:
        raise ValueError("beam needs distinct endpoints and positive width/height")
    tangent /= length
    reference = np.asarray([0.0, 0.0, 1.0])
    if abs(float(np.dot(tangent, reference))) > 0.95:
        reference = np.asarray([1.0, 0.0, 0.0])
    lateral = np.cross(tangent, reference)
    lateral /= np.linalg.norm(lateral)
    vertical = np.cross(lateral, tangent)
    vertical /= np.linalg.norm(vertical)
    offsets = [
        -lateral * width / 2 - vertical * height / 2,
        lateral * width / 2 - vertical * height / 2,
        lateral * width / 2 + vertical * height / 2,
        -lateral * width / 2 + vertical * height / 2,
    ]
    vertices = np.asarray([*(a + offset for offset in offsets), *(b + offset for offset in offsets)])
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return vertices, faces


def _extruded_polygon_xy(
    points: list[list[float]], bottom_z: float, top_z: float
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    xy = np.asarray(points, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or top_z <= bottom_z:
        raise ValueError("extruded_polygon_xy needs >=3 XY points and top_z > bottom_z")
    signed_area = 0.5 * float(
        np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1])
    )
    if abs(signed_area) <= 1e-12:
        raise ValueError("extruded polygon has zero area")
    if signed_area < 0:
        xy = xy[::-1]
    bottom = np.column_stack((xy, np.full(len(xy), bottom_z)))
    top = np.column_stack((xy, np.full(len(xy), top_z)))
    vertices = np.vstack((bottom, top))
    size = len(xy)
    faces: list[tuple[int, ...]] = [
        tuple(reversed(range(size))),
        tuple(size + index for index in range(size)),
    ]
    for index in range(size):
        following = (index + 1) % size
        faces.append((index, following, size + following, size + index))
    return vertices, faces


def _panel(
    points: list[list[float]], thickness: float
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    polygon = np.asarray(points, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 3 or len(polygon) < 3 or thickness <= 0:
        raise ValueError("panel needs >=3 XYZ points and positive thickness")
    normal = np.zeros(3, dtype=np.float64)
    for index in range(1, len(polygon) - 1):
        normal = np.cross(polygon[index] - polygon[0], polygon[index + 1] - polygon[0])
        if np.linalg.norm(normal) > 1e-9:
            break
    magnitude = float(np.linalg.norm(normal))
    if magnitude <= 1e-9:
        raise ValueError("panel points are collinear")
    normal /= magnitude
    lower = polygon - normal * thickness / 2.0
    upper = polygon + normal * thickness / 2.0
    vertices = np.vstack((lower, upper))
    size = len(polygon)
    faces: list[tuple[int, ...]] = [
        tuple(reversed(range(size))),
        tuple(size + index for index in range(size)),
    ]
    for index in range(size):
        following = (index + 1) % size
        faces.append((index, following, size + following, size + index))
    return vertices, faces


def _tube(
    points: list[list[float]], radius: float, sides: int
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    centerline = np.asarray(points, dtype=np.float64)
    if centerline.ndim != 2 or centerline.shape[1] != 3 or len(centerline) < 2:
        raise ValueError("tube needs at least two XYZ centerline points")
    if radius <= 0 or sides < 3:
        raise ValueError("tube radius must be positive and sides >= 3")
    tangents = np.gradient(centerline, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)
    rings = []
    previous_normal: np.ndarray | None = None
    for center, tangent in zip(centerline, tangents):
        reference = np.asarray([0.0, 0.0, 1.0])
        if abs(float(np.dot(tangent, reference))) > 0.92:
            reference = np.asarray([1.0, 0.0, 0.0])
        normal = np.cross(tangent, reference)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        if previous_normal is not None and np.dot(normal, previous_normal) < 0:
            normal = -normal
        binormal = np.cross(tangent, normal)
        binormal /= max(float(np.linalg.norm(binormal)), 1e-12)
        angles = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
        rings.append(
            np.asarray(
                [center + radius * (math.cos(angle) * normal + math.sin(angle) * binormal) for angle in angles]
            )
        )
        previous_normal = normal
    vertices = np.vstack(rings)
    faces: list[tuple[int, ...]] = []
    for ring in range(len(rings) - 1):
        a = ring * sides
        b = (ring + 1) * sides
        for index in range(sides):
            following = (index + 1) % sides
            faces.append((a + index, a + following, b + following, b + index))
    faces.append(tuple(reversed(range(sides))))
    final = (len(rings) - 1) * sides
    faces.append(tuple(final + index for index in range(sides)))
    return vertices, faces


def _mesh_for_geometry(value: dict[str, Any]) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    kind = value.get("kind")
    if kind == "box":
        return _box(value["center"], value["size"])
    if kind == "beam":
        return _beam(value["start"], value["end"], float(value["width"]), float(value["height"]))
    if kind == "extruded_polygon_xy":
        return _extruded_polygon_xy(value["points"], float(value["bottom_z"]), float(value["top_z"]))
    if kind == "panel":
        return _panel(value["points"], float(value["thickness"]))
    if kind == "tube":
        return _tube(value["points"], float(value["radius"]), int(value.get("sides", 8)))
    raise ValueError(f"Unsupported reviewed geometry kind: {kind}")


def _write_materials(path: Path) -> None:
    value = """# Evidence-aware scene materials
newmtl Observed
Kd 0.55 0.68 0.72
Ks 0.12 0.12 0.12
Ns 18

newmtl PhotoInterpreted
Kd 0.25 0.72 0.82
Ks 0.10 0.10 0.10
Ns 14

newmtl RuleInferred
Kd 0.96 0.43 0.12
Ks 0.05 0.05 0.05
Ns 6

newmtl Unsupported
Kd 0.55 0.42 0.72
Ks 0.03 0.03 0.03
Ns 4
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def build_reviewed_scene(
    project: ProjectConfig,
    layout_path: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    layout = load_json(layout_path)
    if layout.get("schema_version") != "railway.reviewed-scene-layout.v1":
        raise ValueError("Unsupported reviewed scene layout schema")
    layout_assets = list(layout.get("assets", []))
    if not layout_assets:
        raise ValueError("Reviewed scene layout contains no assets")
    ids = [str(item.get("id")) for item in layout_assets]
    if len(ids) != len(set(ids)):
        raise ValueError("Reviewed scene layout contains duplicate asset ids")

    generated: list[tuple[dict[str, Any], np.ndarray, list[tuple[int, ...]]]] = []
    for asset in layout_assets:
        if asset.get("evidence_level") not in MATERIAL_BY_EVIDENCE:
            raise ValueError(f"Invalid evidence level for {asset.get('id')}")
        vertices, faces = _mesh_for_geometry(asset["geometry"])
        generated.append((asset, vertices, faces))
    all_vertices = np.vstack([item[1] for item in generated])
    requested_origin = layout.get("origin_xyz")
    origin = (
        np.asarray(requested_origin, dtype=np.float64)
        if requested_origin is not None
        else np.asarray([all_vertices[:, 0].min(), all_vertices[:, 1].min(), 0.0])
    )
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("origin_xyz must contain three finite numbers")

    output_dir = project.workspace_path("exports") / "reviewed_scene"
    obj_path = output_dir / "reviewed_scene.obj"
    mtl_path = output_dir / "reviewed_scene.mtl"
    origin_path = output_dir / "model_origin.json"
    report_path = project.workspace_path("reports") / "reviewed_scene_build.json"
    for path in (obj_path, mtl_path, origin_path, report_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite: {path}")

    writer = ObjWriter(origin, material_library=mtl_path.name)
    registry_assets = []
    for asset, vertices, faces in generated:
        writer.add_mesh(
            asset["id"],
            vertices,
            faces,
            MATERIAL_BY_EVIDENCE[asset["evidence_level"]],
        )
        registry_assets.append(
            {
                "id": asset["id"],
                "type": asset["type"],
                "subtype": asset.get("subtype"),
                "status": asset.get("status", "reviewed"),
                "chainage_m": asset.get("chainage_m"),
                "evidence_level": asset["evidence_level"],
                "confidence": float(asset["confidence"]),
                "sources": asset["sources"],
                "parameters": asset.get("parameters", {}),
                "geometry": {"node": asset["id"], "file": str(obj_path)},
                "limitations": asset.get("limitations", []),
            }
        )
    writer.write(obj_path)
    _write_materials(mtl_path)
    write_json(origin_path, {"origin_xyz": origin.tolist(), "units": "metre", "axis": "Z-up"})

    registry_path = project.workspace_path("asset_registry")
    registry = load_json(registry_path) if registry_path.is_file() else new_registry(project.project_id)
    incoming = {item["id"] for item in registry_assets}
    conflicts = incoming & {item["id"] for item in registry.get("assets", [])}
    if conflicts and not overwrite:
        raise ValueError(f"Registry already contains reviewed scene ids: {sorted(conflicts)}")
    registry["assets"] = [item for item in registry["assets"] if item["id"] not in incoming]
    registry["assets"].extend(registry_assets)
    layout_relations = list(layout.get("relations", []))
    incoming_relations = {item["id"] for item in layout_relations}
    registry["relations"] = [
        item for item in registry.get("relations", []) if item["id"] not in incoming_relations
    ]
    registry["relations"].extend(layout_relations)
    registry["summary"] = summarize_registry(registry)
    errors = validate_registry_value(registry)
    if errors:
        raise ValueError("Reviewed scene registry is invalid:\n- " + "\n- ".join(errors))
    write_json(registry_path, registry)

    report = {
        "schema_version": "railway.reviewed-scene-build.v1",
        "project_id": project.project_id,
        "layout": str(layout_path.resolve()),
        "output_obj": str(obj_path),
        "output_mtl": str(mtl_path),
        "origin": str(origin_path),
        "asset_count": len(registry_assets),
        "vertex_count": writer.vertex_count,
        "face_count": writer.face_count,
        "geometry_kinds": sorted({item[0]["geometry"]["kind"] for item in generated}),
        "status": "reviewed_layout_built_mesh_audit_and_visual_acceptance_required",
    }
    write_json(report_path, report)
    return report

