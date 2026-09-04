from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .camera import camera_trajectory, load_camera_rows
from .config import ProjectConfig
from .io import load_json
from .multi_source import effective_camera_csv_path


@dataclass(frozen=True)
class CorridorFrame:
    origin_xy: np.ndarray
    along_xy: np.ndarray
    cross_xy: np.ndarray

    def project(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        delta = np.column_stack((x, y)) - self.origin_xy
        return delta @ self.along_xy, delta @ self.cross_xy

    def world_xy(self, longitudinal: np.ndarray, cross: np.ndarray) -> np.ndarray:
        return (
            self.origin_xy
            + longitudinal[:, None] * self.along_xy
            + cross[:, None] * self.cross_xy
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "origin_xy": self.origin_xy.tolist(),
            "along_xy": self.along_xy.tolist(),
            "cross_xy": self.cross_xy.tolist(),
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> CorridorFrame:
        return cls(
            np.asarray(value["origin_xy"], dtype=np.float64),
            np.asarray(value["along_xy"], dtype=np.float64),
            np.asarray(value["cross_xy"], dtype=np.float64),
        )


def fit_corridor_frame(camera_trajectory: list[dict[str, Any]]) -> CorridorFrame:
    if len(camera_trajectory) < 2:
        raise ValueError("At least two camera positions are required to fit a corridor frame")
    xy = np.asarray([[item["x"], item["y"]] for item in camera_trajectory], dtype=np.float64)
    origin = np.mean(xy, axis=0)
    centered = xy - origin
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    along = vh[0]
    direction = xy[-1] - xy[0]
    if np.dot(along, direction) < 0:
        along = -along
    along = along / max(float(np.linalg.norm(along)), 1e-12)
    cross = np.asarray([-along[1], along[0]], dtype=np.float64)
    return CorridorFrame(origin, along, cross)


def fit_corridor_frame_dominant_axis_regression(
    camera_trajectory: list[dict[str, Any]],
) -> CorridorFrame:
    """Fit a stable line using the corridor's dominant coordinate as predictor.

    This preserves narrow rail-head peaks when the vehicle path has small lateral
    oscillations that can rotate a short-segment PCA frame enough to smear them.
    """

    if len(camera_trajectory) < 2:
        raise ValueError("At least two camera positions are required to fit a corridor frame")
    xy = np.asarray(
        [[item["x"], item["y"]] for item in camera_trajectory],
        dtype=np.float64,
    )
    origin = np.mean(xy, axis=0)
    direction = xy[-1] - xy[0]
    if float(np.ptp(xy[:, 1])) >= float(np.ptp(xy[:, 0])):
        slope, _ = np.polyfit(xy[:, 1], xy[:, 0], 1)
        along = np.asarray([slope, 1.0], dtype=np.float64)
    else:
        slope, _ = np.polyfit(xy[:, 0], xy[:, 1], 1)
        along = np.asarray([1.0, slope], dtype=np.float64)
    if float(np.dot(along, direction)) < 0:
        along = -along
    along /= max(float(np.linalg.norm(along)), 1e-12)
    cross = np.asarray([-along[1], along[0]], dtype=np.float64)
    return CorridorFrame(origin, along, cross)


def fit_segment_corridor_frame(
    project: ProjectConfig,
    segment_id: str,
    context_before_m: float = 0.0,
    context_after_m: float = 0.0,
) -> tuple[CorridorFrame, list[dict[str, Any]]]:
    manifest_path = project.workspace_path("segment_manifest")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = load_json(manifest_path)
    segment = next(
        (item for item in manifest.get("segments", []) if item.get("id") == segment_id),
        None,
    )
    if segment is None:
        raise ValueError(f"Segment is not present in the manifest: {segment_id}")
    camera_path = effective_camera_csv_path(project)
    trajectory = camera_trajectory(load_camera_rows(camera_path))
    start = float(segment["chainage_start_m"])
    end = float(segment["chainage_end_m"])
    if context_before_m < 0 or context_after_m < 0:
        raise ValueError("Corridor frame context distances cannot be negative")
    context_start = max(0.0, start - float(context_before_m))
    context_end = min(
        float(trajectory[-1]["distance_m"]),
        end + float(context_after_m),
    )
    selected = [
        item
        for item in trajectory
        if context_start <= item["distance_m"] <= context_end
    ]
    if len(selected) < 2:
        index_from = int(segment["camera_index_from"])
        index_to = int(segment["camera_index_to"])
        selected = [item for item in trajectory if index_from <= item["index"] <= index_to]
    if len(selected) < 2:
        raise ValueError(f"Segment {segment_id} has fewer than two camera poses")
    return fit_corridor_frame(selected), selected
