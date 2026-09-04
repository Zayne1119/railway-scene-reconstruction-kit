"""Coordinate and geometry-consumer preflight for point-cloud/model comparisons.

The functions in this module are project agnostic. Callers provide the point
translation that places LAS/LAZ coordinates in the model frame and can tighten
or relax every threshold for their own scene scale and mesh consumer.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import laspy
import numpy as np


@dataclass(frozen=True)
class CoordinatePreflightThresholds:
    """Thresholds for bounds agreement and downstream OBJ compatibility."""

    minimum_axis_overlap_ratio: float = 0.05
    minimum_smaller_volume_overlap_ratio: float = 0.01
    maximum_normalized_center_distance: float = 1.0
    minimum_triangle_face_ratio: float = 0.0

    def __post_init__(self) -> None:
        unit_interval = {
            "minimum_axis_overlap_ratio": self.minimum_axis_overlap_ratio,
            "minimum_smaller_volume_overlap_ratio": (self.minimum_smaller_volume_overlap_ratio),
            "minimum_triangle_face_ratio": self.minimum_triangle_face_ratio,
        }
        for name, value in unit_interval.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1], got {value}")
        if self.maximum_normalized_center_distance < 0.0:
            raise ValueError("maximum_normalized_center_distance must be non-negative")


class CoordinatePreflightError(RuntimeError):
    """Raised when a point-cloud/model pair is unsafe for distance evaluation."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        codes = ", ".join(item["code"] for item in report["failures"])
        super().__init__(f"Point-cloud/model preflight failed: {codes}")


def _vector3(values: Sequence[float], label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (3,):
        raise ValueError(f"{label} must contain exactly three values")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain only finite values")
    return result


def _bounds(minimum: Sequence[float], maximum: Sequence[float], label: str) -> dict[str, Any]:
    lower = _vector3(minimum, f"{label} minimum")
    upper = _vector3(maximum, f"{label} maximum")
    extent = upper - lower
    if np.any(extent <= 0.0):
        raise ValueError(f"{label} bounds must have positive extent on every axis")
    return {
        "minimum": lower,
        "maximum": upper,
        "extent": extent,
        "center": (lower + upper) / 2.0,
        "volume": float(np.prod(extent)),
    }


def inspect_obj_geometry(path: Path) -> dict[str, Any]:
    """Return streaming OBJ bounds and face-arity statistics.

    The triangle ratio is useful when the downstream distance engine accepts
    triangles only and is configured not to triangulate polygons.
    """

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    minimum = np.full(3, np.inf)
    maximum = np.full(3, -np.inf)
    vertex_count = 0
    face_count = 0
    face_arity_counts: dict[int, int] = {}
    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for line_number, raw in enumerate(stream, 1):
            if raw.startswith("v "):
                fields = raw.split()
                if len(fields) < 4:
                    raise ValueError(f"{path}:{line_number}: malformed vertex")
                point = np.asarray([float(fields[1]), float(fields[2]), float(fields[3])])
                if not np.all(np.isfinite(point)):
                    raise ValueError(f"{path}:{line_number}: non-finite vertex")
                minimum = np.minimum(minimum, point)
                maximum = np.maximum(maximum, point)
                vertex_count += 1
            elif raw.startswith("f "):
                arity = len(raw.split()) - 1
                if arity < 3:
                    raise ValueError(f"{path}:{line_number}: face has fewer than 3 vertices")
                face_arity_counts[arity] = face_arity_counts.get(arity, 0) + 1
                face_count += 1
    if vertex_count == 0:
        raise ValueError(f"OBJ contains no vertices: {path}")
    if face_count == 0:
        raise ValueError(f"OBJ contains no faces: {path}")
    if np.any(maximum - minimum <= 0.0):
        raise ValueError(f"OBJ bounds must have positive extent on every axis: {path}")
    triangle_count = face_arity_counts.get(3, 0)
    return {
        "path": str(path),
        "vertex_count": vertex_count,
        "face_count": face_count,
        "triangle_face_count": triangle_count,
        "non_triangle_face_count": face_count - triangle_count,
        "triangle_face_ratio": triangle_count / face_count,
        "face_arity_counts": {
            str(key): face_arity_counts[key] for key in sorted(face_arity_counts)
        },
        "bounds": {
            "minimum": minimum.tolist(),
            "maximum": maximum.tolist(),
            "extent": (maximum - minimum).tolist(),
        },
    }


def inspect_las_bounds(
    paths: Sequence[Path], translation: Sequence[float] = (0.0, 0.0, 0.0)
) -> dict[str, Any]:
    """Combine LAS/LAZ header bounds and translate them into a model frame."""

    if not paths:
        raise ValueError("At least one LAS/LAZ path is required")
    offset = _vector3(translation, "point translation")
    minimum = np.full(3, np.inf)
    maximum = np.full(3, -np.inf)
    sources = []
    for item in paths:
        path = Path(item).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() not in {".las", ".laz"}:
            raise ValueError(f"Expected LAS/LAZ input: {path}")
        with laspy.open(path) as reader:
            lower = np.asarray(reader.header.mins, dtype=np.float64)
            upper = np.asarray(reader.header.maxs, dtype=np.float64)
            count = int(reader.header.point_count)
        minimum = np.minimum(minimum, lower)
        maximum = np.maximum(maximum, upper)
        sources.append(
            {
                "path": str(path),
                "point_count": count,
                "bounds": {"minimum": lower.tolist(), "maximum": upper.tolist()},
            }
        )
    translated_minimum = minimum + offset
    translated_maximum = maximum + offset
    _bounds(translated_minimum, translated_maximum, "translated point-cloud")
    return {
        "sources": sources,
        "source_point_count": sum(row["point_count"] for row in sources),
        "source_bounds": {"minimum": minimum.tolist(), "maximum": maximum.tolist()},
        "translation": offset.tolist(),
        "evaluation_bounds": {
            "minimum": translated_minimum.tolist(),
            "maximum": translated_maximum.tolist(),
            "extent": (translated_maximum - translated_minimum).tolist(),
        },
    }


def evaluate_point_model_consistency(
    point_minimum: Sequence[float],
    point_maximum: Sequence[float],
    model_minimum: Sequence[float],
    model_maximum: Sequence[float],
    *,
    thresholds: CoordinatePreflightThresholds | None = None,
    triangle_face_ratio: float | None = None,
) -> dict[str, Any]:
    """Evaluate whether point and model bounds plausibly share one coordinate frame."""

    configured = thresholds or CoordinatePreflightThresholds()
    point = _bounds(point_minimum, point_maximum, "point-cloud")
    model = _bounds(model_minimum, model_maximum, "model")
    intersection = np.maximum(
        0.0,
        np.minimum(point["maximum"], model["maximum"])
        - np.maximum(point["minimum"], model["minimum"]),
    )
    smaller_extent = np.minimum(point["extent"], model["extent"])
    axis_overlap = intersection / smaller_extent
    intersection_volume = float(np.prod(intersection))
    smaller_volume_overlap = intersection_volume / min(point["volume"], model["volume"])
    normalizer = max(float(np.linalg.norm(point["extent"])), float(np.linalg.norm(model["extent"])))
    normalized_center_distance = float(
        np.linalg.norm(point["center"] - model["center"]) / normalizer
    )

    checks = {
        "minimum_axis_overlap_ratio": bool(
            float(np.min(axis_overlap)) >= configured.minimum_axis_overlap_ratio
        ),
        "smaller_volume_overlap_ratio": bool(
            smaller_volume_overlap >= configured.minimum_smaller_volume_overlap_ratio
        ),
        "normalized_center_distance": bool(
            normalized_center_distance <= configured.maximum_normalized_center_distance
        ),
    }
    if triangle_face_ratio is not None:
        if not math.isfinite(triangle_face_ratio) or not 0.0 <= triangle_face_ratio <= 1.0:
            raise ValueError("triangle_face_ratio must be finite and within [0, 1]")
        checks["triangle_face_ratio"] = bool(
            triangle_face_ratio >= configured.minimum_triangle_face_ratio
        )

    failure_codes = {
        "minimum_axis_overlap_ratio": "point_model_axis_overlap_too_small",
        "smaller_volume_overlap_ratio": "point_model_volume_overlap_too_small",
        "normalized_center_distance": "point_model_centers_too_far_apart",
        "triangle_face_ratio": "model_face_arity_incompatible_with_triangle_consumer",
    }
    failures = [
        {"code": failure_codes[name], "check": name}
        for name, passed in checks.items()
        if not passed
    ]
    return {
        "schema_version": "railway.point-model-coordinate-preflight.v1",
        "passed": not failures,
        "thresholds": asdict(configured),
        "bounds": {
            "point_cloud": {
                "minimum": point["minimum"].tolist(),
                "maximum": point["maximum"].tolist(),
                "extent": point["extent"].tolist(),
            },
            "model": {
                "minimum": model["minimum"].tolist(),
                "maximum": model["maximum"].tolist(),
                "extent": model["extent"].tolist(),
            },
            "intersection_extent": intersection.tolist(),
        },
        "metrics": {
            "axis_overlap_ratio_of_smaller_extent": axis_overlap.tolist(),
            "minimum_axis_overlap_ratio": float(np.min(axis_overlap)),
            "intersection_volume": intersection_volume,
            "smaller_volume_overlap_ratio": smaller_volume_overlap,
            "normalized_center_distance": normalized_center_distance,
            "triangle_face_ratio": triangle_face_ratio,
        },
        "checks": checks,
        "failures": failures,
    }


def preflight_point_cloud_obj_consistency(
    point_cloud_paths: Sequence[Path],
    model_obj: Path,
    *,
    point_translation: Sequence[float] = (0.0, 0.0, 0.0),
    thresholds: CoordinatePreflightThresholds | None = None,
) -> dict[str, Any]:
    """Inspect source headers and OBJ text, then evaluate one comparison contract."""

    point_cloud = inspect_las_bounds(point_cloud_paths, point_translation)
    model = inspect_obj_geometry(model_obj)
    report = evaluate_point_model_consistency(
        point_cloud["evaluation_bounds"]["minimum"],
        point_cloud["evaluation_bounds"]["maximum"],
        model["bounds"]["minimum"],
        model["bounds"]["maximum"],
        thresholds=thresholds,
        triangle_face_ratio=model["triangle_face_ratio"],
    )
    report["point_cloud"] = point_cloud
    report["model_obj"] = model
    return report


def require_point_cloud_obj_consistency(
    point_cloud_paths: Sequence[Path],
    model_obj: Path,
    *,
    point_translation: Sequence[float] = (0.0, 0.0, 0.0),
    thresholds: CoordinatePreflightThresholds | None = None,
) -> dict[str, Any]:
    """Return a passing report or raise with the complete failing report."""

    report = preflight_point_cloud_obj_consistency(
        point_cloud_paths,
        model_obj,
        point_translation=point_translation,
        thresholds=thresholds,
    )
    if not report["passed"]:
        raise CoordinatePreflightError(report)
    return report
