from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class ObjWriter:
    def __init__(self, origin: np.ndarray, material_library: str = "railway_model.mtl") -> None:
        self.origin = np.asarray(origin, dtype=np.float64)
        self.lines = [
            "# Railway parametric model",
            "# Coordinates are local metres.",
            f"# Global origin: {origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f}",
            f"mtllib {material_library}",
        ]
        self.vertex_count = 0
        self.face_count = 0

    def add_mesh(
        self,
        name: str,
        vertices: np.ndarray,
        faces: list[tuple[int, ...]],
        material: str,
    ) -> None:
        local = np.asarray(vertices, dtype=np.float64) - self.origin
        self.lines.extend((f"o {name}", f"usemtl {material}"))
        for vertex in local:
            self.lines.append(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}")
        offset = self.vertex_count + 1
        for face in faces:
            self.lines.append("f " + " ".join(str(offset + index) for index in face))
        self.vertex_count += len(local)
        self.face_count += len(faces)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text("\n".join(self.lines) + "\n", encoding="utf-8", newline="\n")
        os.replace(temporary, path)


def rail_profile() -> np.ndarray:
    return np.asarray(
        [
            [-0.0375, 0.000], [0.0375, 0.000], [0.0375, -0.030],
            [0.0120, -0.045], [0.0100, -0.125], [0.0750, -0.135],
            [0.0750, -0.155], [-0.0750, -0.155], [-0.0750, -0.135],
            [-0.0100, -0.125], [-0.0120, -0.045], [-0.0375, -0.030],
        ],
        dtype=np.float64,
    )


def sweep_mesh(
    centerline: np.ndarray, profile: np.ndarray
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    if len(centerline) < 2:
        raise ValueError("A swept mesh needs at least two centerline points")
    tangent = np.gradient(centerline[:, :2], axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    lateral = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    rings = [
        np.column_stack(
            (
                center[0] + normal[0] * profile[:, 0],
                center[1] + normal[1] * profile[:, 0],
                center[2] + profile[:, 1],
            )
        )
        for center, normal in zip(centerline, lateral)
    ]
    vertices = np.vstack(rings)
    size = len(profile)
    faces: list[tuple[int, ...]] = []
    for ring in range(len(centerline) - 1):
        a = ring * size
        b = (ring + 1) * size
        for index in range(size):
            following = (index + 1) % size
            faces.append((a + index, a + following, b + following, b + index))
    faces.append(tuple(reversed(range(size))))
    final = (len(centerline) - 1) * size
    faces.append(tuple(final + index for index in range(size)))
    return vertices, faces


def interpolate_polyline(centerline: np.ndarray, distances: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(centerline[:, :2], axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    result = np.empty((len(distances), 3), dtype=np.float64)
    for axis in range(3):
        result[:, axis] = np.interp(distances, cumulative, centerline[:, axis])
    return result


def oriented_box(
    center: np.ndarray,
    tangent: np.ndarray,
    lateral_length: float,
    longitudinal_width: float,
    bottom_z: float,
    top_z: float,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    tangent_xy = tangent[:2] / max(float(np.linalg.norm(tangent[:2])), 1e-12)
    lateral_xy = np.asarray([-tangent_xy[1], tangent_xy[0]])
    vertices = []
    for z_value in (bottom_z, top_z):
        for longitudinal_sign, lateral_sign in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            xy = (
                center[:2]
                + tangent_xy * longitudinal_sign * longitudinal_width / 2.0
                + lateral_xy * lateral_sign * lateral_length / 2.0
            )
            vertices.append([xy[0], xy[1], z_value])
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return np.asarray(vertices), faces


def write_track_materials(path: Path) -> None:
    content = """# Generic railway materials
newmtl RailSteel
Kd 0.24 0.27 0.30
Ks 0.65 0.65 0.65
Ns 80

newmtl SleeperConcrete
Kd 0.58 0.57 0.54
Ks 0.08 0.08 0.08
Ns 8

newmtl Ballast
Kd 0.36 0.34 0.31
Ks 0.02 0.02 0.02
Ns 3
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)

