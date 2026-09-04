"""Envelope-preserving detail refinement for rule-generated sleeper meshes."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .obj_interchange import ObjFace, ObjObject, _parse_obj


@dataclass(frozen=True)
class SleeperDetailPolicy:
    """Geometric detail that stays inside the source sleeper envelope."""

    corner_chamfer_m: float = 0.04
    top_edge_bevel_m: float = 0.015

    def validate(self) -> None:
        if not np.isfinite(self.corner_chamfer_m) or self.corner_chamfer_m <= 0.0:
            raise ValueError("corner_chamfer_m must be finite and positive")
        if not np.isfinite(self.top_edge_bevel_m) or self.top_edge_bevel_m <= 0.0:
            raise ValueError("top_edge_bevel_m must be finite and positive")


def _chamfered_ring(corners: np.ndarray, distance_m: float) -> np.ndarray:
    value = np.asarray(corners, dtype=np.float64)
    if value.shape != (4, 3):
        raise ValueError("A sleeper ring must contain four XYZ corners")
    result: list[np.ndarray] = []
    for index, corner in enumerate(value):
        previous = value[(index - 1) % 4]
        following = value[(index + 1) % 4]
        previous_length = float(np.linalg.norm(previous[:2] - corner[:2]))
        following_length = float(np.linalg.norm(following[:2] - corner[:2]))
        if min(previous_length, following_length) <= 2.0 * distance_m:
            raise ValueError("Sleeper corner chamfer is too large for the source box")
        incoming = corner + (previous - corner) * (distance_m / previous_length)
        outgoing = corner + (following - corner) * (distance_m / following_length)
        result.extend((incoming, outgoing))
    return np.asarray(result, dtype=np.float64)


def beveled_sleeper_box(
    box_vertices: np.ndarray,
    policy: SleeperDetailPolicy | None = None,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Replace one eight-vertex sleeper box with a closed, inset detail mesh.

    The bottom outline retains the source XY envelope, the top elevation is
    unchanged, and all newly introduced vertices lie inside that envelope.
    Input vertex ordering must match :func:`railway_recon.algorithms.mesh.oriented_box`.
    """

    settings = policy or SleeperDetailPolicy()
    settings.validate()
    source = np.asarray(box_vertices, dtype=np.float64)
    if source.shape != (8, 3) or not np.all(np.isfinite(source)):
        raise ValueError("A sleeper source box must be a finite 8x3 array")
    bottom = source[:4]
    top = source[4:]
    bottom_z = float(np.median(bottom[:, 2]))
    top_z = float(np.median(top[:, 2]))
    height = top_z - bottom_z
    if height <= 2.0 * settings.top_edge_bevel_m:
        raise ValueError("Sleeper top-edge bevel is too large for the source height")
    if float(np.max(np.abs(bottom[:, 2] - bottom_z))) > 1.0e-6:
        raise ValueError("Sleeper bottom ring is not planar")
    if float(np.max(np.abs(top[:, 2] - top_z))) > 1.0e-6:
        raise ValueError("Sleeper top ring is not planar")

    bottom_outline = _chamfered_ring(bottom, settings.corner_chamfer_m)
    shoulder = bottom_outline.copy()
    shoulder[:, 2] = top_z - settings.top_edge_bevel_m

    inset_top = top.copy()
    for index, corner in enumerate(top):
        previous = top[(index - 1) % 4]
        following = top[(index + 1) % 4]
        toward_previous = previous[:2] - corner[:2]
        toward_following = following[:2] - corner[:2]
        toward_previous /= np.linalg.norm(toward_previous)
        toward_following /= np.linalg.norm(toward_following)
        inset_top[index, :2] += settings.top_edge_bevel_m * (
            toward_previous + toward_following
        )
    top_chamfer = min(settings.corner_chamfer_m, 0.45 * min(
        float(np.linalg.norm(inset_top[1, :2] - inset_top[0, :2])),
        float(np.linalg.norm(inset_top[2, :2] - inset_top[1, :2])),
    ))
    top_outline = _chamfered_ring(inset_top, top_chamfer)

    vertices = np.vstack((bottom_outline, shoulder, top_outline))
    ring_size = 8
    faces: list[tuple[int, ...]] = [tuple(reversed(range(ring_size)))]
    for first_ring, second_ring in ((0, 1), (1, 2)):
        first = first_ring * ring_size
        second = second_ring * ring_size
        for index in range(ring_size):
            following = (index + 1) % ring_size
            faces.append(
                (first + index, first + following, second + following, second + index)
            )
    top_offset = 2 * ring_size
    faces.append(tuple(top_offset + index for index in range(ring_size)))
    return vertices, faces


def refine_sleeper_group(
    vertices: np.ndarray,
    policy: SleeperDetailPolicy | None = None,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Refine a merged group of eight-vertex sleeper boxes."""

    source = np.asarray(vertices, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3 or len(source) % 8:
        raise ValueError("Sleeper group vertices must be an Nx3 array divisible by eight")
    output_vertices: list[np.ndarray] = []
    output_faces: list[tuple[int, ...]] = []
    offset = 0
    for start in range(0, len(source), 8):
        box_vertices, box_faces = beveled_sleeper_box(source[start : start + 8], policy)
        output_vertices.append(box_vertices)
        output_faces.extend(tuple(index + offset for index in face) for face in box_faces)
        offset += len(box_vertices)
    return np.vstack(output_vertices), output_faces


def _object_local_mesh(
    xyz: np.ndarray, obj: ObjObject
) -> tuple[np.ndarray, list[ObjFace]]:
    used = sorted({index for face in obj.faces for index in face.vertices})
    mapping = {source: local for local, source in enumerate(used)}
    faces = [
        ObjFace(tuple(mapping[index] for index in face.vertices), face.material)
        for face in obj.faces
    ]
    return xyz[used], faces


def _oriented_envelope(vertices: np.ndarray, source_box: np.ndarray) -> np.ndarray:
    first = source_box[1, :2] - source_box[0, :2]
    second = source_box[3, :2] - source_box[0, :2]
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    xy = np.asarray(vertices, dtype=np.float64)[:, :2]
    return np.asarray(
        [
            np.min(xy @ first),
            np.max(xy @ first),
            np.min(xy @ second),
            np.max(xy @ second),
            np.min(vertices[:, 2]),
            np.max(vertices[:, 2]),
        ]
    )


def _write_obj(
    path: Path,
    material_libraries: Iterable[str],
    meshes: list[tuple[str, np.ndarray, list[ObjFace]]],
) -> None:
    lines = [
        "# Railway candidate mesh with envelope-preserving sleeper detail",
        "# Coordinates remain local metres; no status promotion is implied.",
    ]
    lines.extend(f"mtllib {name}" for name in material_libraries)
    offset = 1
    for name, vertices, faces in meshes:
        lines.append(f"o {name}")
        for vertex in vertices:
            lines.append(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}")
        current_material: str | None = None
        for face in faces:
            if face.material != current_material:
                if face.material:
                    lines.append(f"usemtl {face.material}")
                current_material = face.material
            lines.append("f " + " ".join(str(offset + index) for index in face.vertices))
        offset += len(vertices)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def refine_sleeper_objects_in_obj(
    source_obj: str | Path,
    output_obj: str | Path,
    target_nodes: Iterable[str],
    *,
    material_library: str | None = None,
    policy: SleeperDetailPolicy | None = None,
) -> dict[str, object]:
    """Refine named sleeper objects while preserving every other OBJ object."""

    source = Path(source_obj).resolve()
    output = Path(output_obj).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite candidate OBJ: {output}")
    target = {str(value) for value in target_nodes}
    if not target:
        raise ValueError("At least one target sleeper node is required")
    xyz, objects, libraries = _parse_obj(source)
    available = {obj.name for obj in objects}
    missing = sorted(target - available)
    if missing:
        raise ValueError(f"Target sleeper nodes are absent from OBJ: {missing}")

    settings = policy or SleeperDetailPolicy()
    meshes: list[tuple[str, np.ndarray, list[ObjFace]]] = []
    records: list[dict[str, object]] = []
    for obj in objects:
        local_vertices, local_faces = _object_local_mesh(xyz, obj)
        if obj.name not in target:
            meshes.append((obj.name, local_vertices, local_faces))
            continue
        materials = {face.material for face in local_faces}
        if len(materials) != 1:
            raise ValueError(f"Sleeper object must use exactly one material: {obj.name}")
        refined_vertices, refined_faces = refine_sleeper_group(local_vertices, settings)
        material = next(iter(materials))
        meshes.append(
            (
                obj.name,
                refined_vertices,
                [ObjFace(face, material) for face in refined_faces],
            )
        )
        source_boxes = local_vertices.reshape(-1, 8, 3)
        refined_boxes = refined_vertices.reshape(-1, 24, 3)
        source_centers = np.asarray(
            [
                [*box[:, :2].mean(axis=0), (box[:, 2].min() + box[:, 2].max()) * 0.5]
                for box in source_boxes
            ]
        )
        refined_centers = np.asarray(
            [
                [*box[:, :2].mean(axis=0), (box[:, 2].min() + box[:, 2].max()) * 0.5]
                for box in refined_boxes
            ]
        )
        envelope_preserved = all(
            np.allclose(
                _oriented_envelope(refined_vertices[index * 24 : (index + 1) * 24], box),
                _oriented_envelope(box, box),
                atol=1.0e-6,
            )
            for index, box in enumerate(source_boxes)
        )
        records.append(
            {
                "node": obj.name,
                "sleeper_count": len(local_vertices) // 8,
                "source_vertex_count": len(local_vertices),
                "refined_vertex_count": len(refined_vertices),
                "source_face_count": len(local_faces),
                "refined_face_count": len(refined_faces),
                "oriented_envelope_preserved": bool(envelope_preserved),
                "center_preserved": bool(
                    np.allclose(source_centers, refined_centers, atol=1.0e-6)
                ),
            }
        )
    if not all(bool(record["oriented_envelope_preserved"]) for record in records):
        raise ValueError("Sleeper detail escaped a source geometry envelope")
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_obj(
        output,
        [material_library] if material_library else libraries,
        meshes,
    )
    return {
        "source_obj": str(source),
        "output_obj": str(output),
        "target_count": len(target),
        "refined_sleeper_count": sum(int(record["sleeper_count"]) for record in records),
        "corner_chamfer_m": settings.corner_chamfer_m,
        "top_edge_bevel_m": settings.top_edge_bevel_m,
        "records": records,
        "passed": len(records) == len(target)
        and all(
            bool(record["oriented_envelope_preserved"])
            and bool(record["center_preserved"])
            for record in records
        ),
    }
