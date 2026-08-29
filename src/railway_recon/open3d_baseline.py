from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import laspy
import numpy as np

from .geometry import CorridorFrame
from .io import load_json, sha256_file, write_json


def _fit_line(longitudinal: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    if len(longitudinal) < 2 or float(np.ptp(longitudinal)) <= 1e-9:
        return 0.0, float(np.median(values)), math.nan
    slope, intercept = np.polyfit(longitudinal, values, 1)
    residual = np.abs(values - (slope * longitudinal + intercept))
    cutoff = max(0.02, float(np.percentile(residual, 90)))
    keep = residual <= cutoff
    if int(np.count_nonzero(keep)) >= 2:
        slope, intercept = np.polyfit(longitudinal[keep], values[keep], 1)
        residual = np.abs(values[keep] - (slope * longitudinal[keep] + intercept))
    return float(slope), float(intercept), float(np.percentile(residual, 90))


def _sample_source(
    source: Path,
    frame: CorridorFrame,
    z_range: tuple[float, float],
    maximum_input_points: int,
) -> np.ndarray:
    with laspy.open(source) as reader:
        stride = max(1, math.ceil(int(reader.header.point_count) / maximum_input_points))
        selected: list[np.ndarray] = []
        offset = 0
        for chunk in reader.chunk_iterator(2_000_000):
            local_index = np.arange(len(chunk), dtype=np.int64)
            keep_stride = ((local_index + offset) % stride) == 0
            offset += len(chunk)
            if not np.any(keep_stride):
                continue
            x = np.asarray(chunk.x, dtype=np.float64)[keep_stride]
            y = np.asarray(chunk.y, dtype=np.float64)[keep_stride]
            z = np.asarray(chunk.z, dtype=np.float64)[keep_stride]
            keep_z = (z >= z_range[0]) & (z <= z_range[1])
            if not np.any(keep_z):
                continue
            longitudinal, cross = frame.project(x[keep_z], y[keep_z])
            selected.append(np.column_stack((longitudinal, cross, z[keep_z])))
    return np.concatenate(selected) if selected else np.empty((0, 3), dtype=np.float64)


def detect_open3d_rail_baseline(
    source_path: str | Path,
    reference_report_path: str | Path,
    segment_id: str,
    output_path: str | Path,
    *,
    voxel_size_m: float = 0.08,
    plane_distance_m: float = 0.035,
    dbscan_eps_m: float = 0.12,
    dbscan_min_points: int = 8,
    longitudinal_scale: float = 0.02,
    minimum_cluster_length_m: float = 15.0,
    maximum_cluster_width_m: float = 0.35,
    maximum_input_points: int = 2_000_000,
    seed: int = 20260827,
) -> Path:
    """Run a generic Open3D plane-removal + DBSCAN rail baseline.

    The reference report contributes only the shared corridor frame and frozen
    height envelope.  No candidate positions, pairs or model geometry are read.
    """

    if min(voxel_size_m, plane_distance_m, dbscan_eps_m, longitudinal_scale) <= 0:
        raise ValueError("Open3D baseline distances and scale must be positive")
    if dbscan_min_points < 2 or maximum_input_points < 100:
        raise ValueError("Open3D baseline point limits are invalid")
    source = Path(source_path).resolve()
    reference_source = Path(reference_report_path).resolve()
    reference = load_json(reference_source)
    if reference.get("schema_version") != "railway.rail-candidates.v1":
        raise ValueError("Reference must be a rail-candidates report")
    if reference.get("segment_id") != segment_id:
        raise ValueError("Reference segment id differs from requested segment")
    frame = CorridorFrame.from_json(reference["frame"])
    z_range = tuple(float(value) for value in reference["search_z_range_m"])
    if len(z_range) != 2 or z_range[1] <= z_range[0]:
        raise ValueError("Reference report has an invalid search height range")

    try:
        import open3d as o3d
    except ImportError as exc:
        raise ValueError("Open3D baseline requires the optional 'baselines' dependency") from exc

    sampled = _sample_source(source, frame, z_range, maximum_input_points)
    if len(sampled) < 100:
        raise ValueError("Too few points remain for the Open3D baseline")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(sampled)
    cloud = cloud.voxel_down_sample(voxel_size_m)
    downsampled = np.asarray(cloud.points, dtype=np.float64)
    if len(downsampled) < 100:
        raise ValueError("Too few downsampled points remain for the Open3D baseline")

    o3d.utility.random.seed(seed)
    plane_model, inliers = cloud.segment_plane(
        distance_threshold=plane_distance_m,
        ransac_n=3,
        num_iterations=500,
    )
    non_plane = cloud.select_by_index(inliers, invert=True)
    residual = np.asarray(non_plane.points, dtype=np.float64)
    if len(residual) < dbscan_min_points:
        labels = np.empty(0, dtype=np.int64)
    else:
        feature_cloud = o3d.geometry.PointCloud()
        feature_cloud.points = o3d.utility.Vector3dVector(
            np.column_stack(
                (
                    residual[:, 1],
                    (residual[:, 2] - z_range[0]) * 2.0,
                    residual[:, 0] * longitudinal_scale,
                )
            )
        )
        labels = np.asarray(
            feature_cloud.cluster_dbscan(
                eps=dbscan_eps_m,
                min_points=dbscan_min_points,
                print_progress=False,
            ),
            dtype=np.int64,
        )

    clusters: list[dict[str, Any]] = []
    for label in sorted(set(labels.tolist()) - {-1}):
        points = residual[labels == label]
        longitudinal_extent = float(np.percentile(points[:, 0], 95) - np.percentile(points[:, 0], 5))
        cross_width = float(np.percentile(points[:, 1], 95) - np.percentile(points[:, 1], 5))
        if longitudinal_extent < minimum_cluster_length_m or cross_width > maximum_cluster_width_m:
            continue
        cross_slope, cross_intercept, cross_p90 = _fit_line(points[:, 0], points[:, 1])
        z_slope, z_intercept, z_p90 = _fit_line(points[:, 0], points[:, 2])
        clusters.append(
            {
                "label": int(label),
                "point_count": len(points),
                "longitudinal_extent_m": longitudinal_extent,
                "cross_width_p90_m": cross_width,
                "cross_fit_slope_m_per_m": cross_slope,
                "cross_fit_intercept_m": cross_intercept,
                "cross_fit_residual_p90_m": cross_p90,
                "z_fit_slope_m_per_m": z_slope,
                "z_fit_intercept_m": z_intercept,
                "z_fit_residual_p90_m": z_p90,
                "cross_position_m": float(np.median(points[:, 1])),
                "median_z_m": float(np.median(points[:, 2])),
            }
        )
    clusters.sort(key=lambda item: item["cross_position_m"])

    rail_head_width = 0.073
    pair_candidates: list[dict[str, Any]] = []
    for left in range(len(clusters)):
        for right in range(left + 1, len(clusters)):
            separation = float(
                clusters[right]["cross_position_m"] - clusters[left]["cross_position_m"]
            )
            if 1.35 <= separation <= 1.65:
                gauge = separation - rail_head_width
                pair_candidates.append(
                    {
                        "cluster_indexes": [left, right],
                        "cross_positions_m": [
                            clusters[left]["cross_position_m"],
                            clusters[right]["cross_position_m"],
                        ],
                        "observed_cross_positions_m": [
                            clusters[left]["cross_position_m"],
                            clusters[right]["cross_position_m"],
                        ],
                        "separation_m": separation,
                        "rail_center_spacing_m": separation,
                        "observed_rail_center_spacing_m": separation,
                        "gauge_m": gauge,
                        "observed_gauge_m": gauge,
                        "rail_position_correction_m": 0.0,
                        "gauge_constrained_to_nominal": False,
                        "score": float(
                            clusters[left]["point_count"]
                            + clusters[right]["point_count"]
                            - 1000.0 * abs(gauge - 1.435)
                        ),
                    }
                )
    pair_candidates.sort(key=lambda item: item["score"], reverse=True)
    pairs: list[dict[str, Any]] = []
    used: set[int] = set()
    for item in pair_candidates:
        indexes = set(item["cluster_indexes"])
        if indexes & used:
            continue
        pairs.append({**item, "track_id": f"TRACK-{len(pairs) + 1:04d}"})
        used.update(indexes)
        if len(pairs) >= 8:
            break
    selected_cluster_indexes = sorted({index for pair in pairs for index in pair["cluster_indexes"]})
    lines = []
    for line_index, cluster_index in enumerate(selected_cluster_indexes, start=1):
        value = clusters[cluster_index]
        lines.append(
            {
                "id": f"RAIL-{line_index:04d}",
                "cross_position_m": value["cross_position_m"],
                "median_z_m": value["median_z_m"],
                "point_count": value["point_count"],
                "cross_fit_slope_m_per_m": value["cross_fit_slope_m_per_m"],
                "cross_fit_intercept_m": value["cross_fit_intercept_m"],
                "cross_fit_sample_count": value["point_count"],
                "cross_fit_residual_p90_m": value["cross_fit_residual_p90_m"],
                "z_fit_slope_m_per_m": value["z_fit_slope_m_per_m"],
                "z_fit_intercept_m": value["z_fit_intercept_m"],
                "z_fit_sample_count": value["point_count"],
                "z_fit_residual_p90_m": value["z_fit_residual_p90_m"],
            }
        )

    report = {
        "schema_version": "railway.rail-candidates.v1",
        "project_id": reference.get("project_id"),
        "segment_id": segment_id,
        "source": str(source),
        "frame": frame.to_json(),
        "longitudinal_range_m": reference["longitudinal_range_m"],
        "search_z_range_m": list(z_range),
        "detection_method": "open3d_plane_ransac_plus_dbscan_v1",
        "execution_mode": "fully_automatic_generic_geometry_baseline",
        "shared_preprocessing": "reference_frame_and_frozen_z_envelope_only",
        "source_sha256": sha256_file(source),
        "reference_report": str(reference_source),
        "reference_report_sha256": sha256_file(reference_source),
        "generated_at": datetime.now(UTC).isoformat(),
        "parameters": {
            "voxel_size_m": voxel_size_m,
            "plane_distance_m": plane_distance_m,
            "dbscan_eps_m": dbscan_eps_m,
            "dbscan_min_points": dbscan_min_points,
            "longitudinal_scale": longitudinal_scale,
            "minimum_cluster_length_m": minimum_cluster_length_m,
            "maximum_cluster_width_m": maximum_cluster_width_m,
            "maximum_input_points": maximum_input_points,
            "seed": seed,
        },
        "sampled_point_count": len(sampled),
        "downsampled_point_count": len(downsampled),
        "plane_model": [float(value) for value in plane_model],
        "plane_inlier_count": len(inliers),
        "residual_point_count": len(residual),
        "dbscan_cluster_count": len(set(labels.tolist()) - {-1}),
        "eligible_cluster_count": len(clusters),
        "eligible_clusters": clusters,
        "rail_pair_count": len(pairs),
        "rail_pairs": pairs,
        "rail_lines": lines,
        "status": "geometry_candidates_only_no_manual_review",
        "limitations": [
            "Generic RANSAC/DBSCAN does not encode longitudinal rail-head prominence.",
            "The common corridor frame and height envelope are shared for a controlled comparison.",
            "No nominal-gauge correction is applied to the reported raw geometry.",
        ],
    }
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Open3D baseline report: {output}")
    write_json(output, report)
    return output
