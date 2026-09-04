from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, Delaunay

from .io import load_json, write_json
from .model_point_support import load_cloud_sample, object_vertex_indices, parse_obj_model


@dataclass(frozen=True)
class Plane:
    center_xy: np.ndarray
    coefficients: np.ndarray

    def predict(self, xy: np.ndarray) -> np.ndarray:
        centered = xy - self.center_xy
        return (
            self.coefficients[0] * centered[:, 0]
            + self.coefficients[1] * centered[:, 1]
            + self.coefficients[2]
        )

    def to_json(self) -> dict[str, list[float]]:
        return {
            "center_xy": [float(value) for value in self.center_xy],
            "z_equals_a_dx_plus_b_dy_plus_c": [
                float(value) for value in self.coefficients
            ],
        }


def fit_plane(xyz: np.ndarray, *, center_xy: np.ndarray | None = None) -> Plane:
    if len(xyz) < 3:
        raise ValueError("At least three points are required to fit a plane")
    center = (
        np.mean(xyz[:, :2], axis=0)
        if center_xy is None
        else np.asarray(center_xy, dtype=np.float64)
    )
    design = np.column_stack((xyz[:, :2] - center, np.ones(len(xyz))))
    coefficients, _, _, _ = np.linalg.lstsq(design, xyz[:, 2], rcond=None)
    return Plane(center, coefficients)


def top_envelope(vertices: np.ndarray, *, rounding_decimals: int = 5) -> np.ndarray:
    groups: dict[tuple[float, float], np.ndarray] = {}
    for vertex in vertices:
        key = tuple(np.round(vertex[:2], rounding_decimals))
        previous = groups.get(key)
        if previous is None or vertex[2] > previous[2]:
            groups[key] = vertex
    return np.asarray(list(groups.values()), dtype=np.float64)


def _inside_convex_hull(points_xy: np.ndarray, hull_xy: np.ndarray) -> np.ndarray:
    hull = ConvexHull(hull_xy)
    triangulation = Delaunay(hull_xy[hull.vertices])
    return triangulation.find_simplex(points_xy) >= 0


def robust_roof_plane_fit(
    cloud_xyz: np.ndarray,
    roof_vertices: np.ndarray,
    *,
    search_half_width_m: float = 0.35,
    seed_half_width_m: float = 0.08,
    maximum_residual_m: float = 0.08,
) -> dict[str, Any]:
    envelope = top_envelope(roof_vertices)
    current = fit_plane(envelope)
    minimum = np.min(envelope[:, :2], axis=0)
    maximum = np.max(envelope[:, :2], axis=0)
    bbox = (
        (cloud_xyz[:, 0] >= minimum[0])
        & (cloud_xyz[:, 0] <= maximum[0])
        & (cloud_xyz[:, 1] >= minimum[1])
        & (cloud_xyz[:, 1] <= maximum[1])
    )
    candidates = cloud_xyz[bbox]
    if len(candidates):
        candidates = candidates[_inside_convex_hull(candidates[:, :2], envelope[:, :2])]
    if not len(candidates):
        return {"passed": False, "reason": "no_points_inside_roof_footprint"}
    delta = candidates[:, 2] - current.predict(candidates[:, :2])
    near = np.abs(delta) <= search_half_width_m
    candidates = candidates[near]
    delta = delta[near]
    if len(candidates) < 100:
        return {
            "passed": False,
            "reason": "insufficient_near_plane_points",
            "candidate_point_count": len(candidates),
        }
    bin_edges = np.arange(-search_half_width_m, search_half_width_m + 0.01, 0.01)
    histogram, _ = np.histogram(delta, bins=bin_edges)
    mode_index = int(np.argmax(histogram))
    mode_delta = float((bin_edges[mode_index] + bin_edges[mode_index + 1]) / 2.0)
    inliers = np.abs(delta - mode_delta) <= seed_half_width_m
    fitted = current
    for _ in range(4):
        if np.count_nonzero(inliers) < 100:
            break
        fitted = fit_plane(candidates[inliers], center_xy=current.center_xy)
        residual = candidates[:, 2] - fitted.predict(candidates[:, :2])
        selected_residual = residual[inliers]
        median = float(np.median(selected_residual))
        mad = float(np.median(np.abs(selected_residual - median)))
        tolerance = min(maximum_residual_m, max(0.025, 3.0 * 1.4826 * mad))
        inliers = np.abs(residual - median) <= tolerance
    residual = candidates[:, 2] - fitted.predict(candidates[:, :2])
    selected = residual[inliers]
    residual_p90 = float(np.percentile(np.abs(selected), 90))
    center = current.center_xy.reshape(1, 2)
    vertical_correction = float(fitted.predict(center)[0] - current.predict(center)[0])
    current_normal = np.asarray([-current.coefficients[0], -current.coefficients[1], 1.0])
    fitted_normal = np.asarray([-fitted.coefficients[0], -fitted.coefficients[1], 1.0])
    cosine = float(
        np.dot(current_normal, fitted_normal)
        / (np.linalg.norm(current_normal) * np.linalg.norm(fitted_normal))
    )
    angle_change_deg = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    gates = {
        "minimum_inlier_count": int(np.count_nonzero(inliers)) >= 500,
        "minimum_inlier_fraction": float(np.mean(inliers)) >= 0.20,
        "residual_p90_within_80mm": residual_p90 <= maximum_residual_m,
        "vertical_correction_within_200mm": abs(vertical_correction) <= 0.20,
        "slope_change_within_1_5deg": angle_change_deg <= 1.5,
    }
    return {
        "passed": all(gates.values()),
        "candidate_point_count": len(candidates),
        "inlier_point_count": int(np.count_nonzero(inliers)),
        "inlier_fraction": float(np.mean(inliers)),
        "mode_delta_m": mode_delta,
        "residual_p90_m": residual_p90,
        "vertical_correction_at_center_m": vertical_correction,
        "plane_angle_change_deg": angle_change_deg,
        "current_top_plane": current.to_json(),
        "fitted_top_plane": fitted.to_json(),
        "gates": gates,
    }


def audit_canopy_plane_refits(
    obj_path: str | Path,
    origin_path: str | Path,
    registry_path: str | Path,
    cloud_path: str | Path,
    output_path: str | Path,
    *,
    target_cloud_points: int = 4_000_000,
) -> Path:
    model = parse_obj_model(obj_path)
    origin = np.asarray(load_json(Path(origin_path))["origin_xyz"], dtype=np.float64)
    registry = load_json(Path(registry_path))
    roof_names = sorted(
        str(asset["id"])
        for asset in registry.get("assets", [])
        if asset.get("type") == "canopy_roof_surface"
        and str(asset.get("id", "")) in model.faces_by_object
    )
    cloud, sampling = load_cloud_sample(cloud_path, target_point_count=target_cloud_points)
    records: list[dict[str, Any]] = []
    for name in roof_names:
        indices = object_vertex_indices(model, name)
        result = robust_roof_plane_fit(cloud, model.vertices[indices] + origin)
        records.append({"object_name": name, **result})
    report = {
        "schema_version": "railway.canopy-plane-refit-audit.v1",
        "inputs": {
            "obj": str(Path(obj_path).resolve()),
            "origin": str(Path(origin_path).resolve()),
            "registry": str(Path(registry_path).resolve()),
            "cloud": str(Path(cloud_path).resolve()),
        },
        "sampling": sampling,
        "roof_object_count": len(records),
        "passed_count": sum(bool(item["passed"]) for item in records),
        "records": records,
        "policy": (
            "Only passed records may enter a geometry candidate; failed records remain unchanged."
        ),
    }
    destination = Path(output_path).resolve()
    write_json(destination, report)
    return destination
