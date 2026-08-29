from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .geometry import CorridorFrame
from .io import load_json, sha256_file, write_json
from .point_holdout import spatial_voxel_holdout_mask


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mae_m": None,
            "rmse_m": None,
            "p50_m": None,
            "p90_m": None,
            "p95_m": None,
            "maximum_m": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mae_m": float(np.mean(np.abs(array))),
        "rmse_m": float(np.sqrt(np.mean(np.square(array)))),
        "p50_m": float(np.percentile(array, 50)),
        "p90_m": float(np.percentile(array, 90)),
        "p95_m": float(np.percentile(array, 95)),
        "maximum_m": float(np.max(array)),
    }


def _line_value(line: dict[str, Any], coordinate: float, prefix: str) -> float:
    return float(line[f"{prefix}_fit_slope_m_per_m"]) * coordinate + float(
        line[f"{prefix}_fit_intercept_m"]
    )


def _match_records(
    train_values: list[float], holdout_values: list[float], maximum_distance_m: float
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    costs = np.abs(
        np.asarray(train_values, dtype=np.float64)[:, None]
        - np.asarray(holdout_values, dtype=np.float64)[None, :]
    )
    matches: list[tuple[int, int, float]] = []
    matched_train: set[int] = set()
    matched_holdout: set[int] = set()
    if costs.size:
        rows, columns = linear_sum_assignment(costs)
        for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
            distance = float(costs[row, column])
            if distance <= maximum_distance_m:
                matches.append((row, column, distance))
                matched_train.add(row)
                matched_holdout.add(column)
    return (
        matches,
        [index for index in range(len(train_values)) if index not in matched_train],
        [index for index in range(len(holdout_values)) if index not in matched_holdout],
    )


def _search_holdout_points(
    cloud_path: Path,
    frame: CorridorFrame,
    longitudinal_range: tuple[float, float],
    z_range: tuple[float, float],
    chunk_size: int = 2_000_000,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    with laspy.open(cloud_path) as reader:
        for points in reader.chunk_iterator(chunk_size):
            x = np.asarray(points.x, dtype=np.float64)
            y = np.asarray(points.y, dtype=np.float64)
            z = np.asarray(points.z, dtype=np.float64)
            longitudinal, cross = frame.project(x, y)
            selected = (
                (longitudinal >= longitudinal_range[0])
                & (longitudinal <= longitudinal_range[1])
                & (z >= z_range[0])
                & (z <= z_range[1])
            )
            if np.any(selected):
                chunks.append(
                    np.column_stack(
                        (longitudinal[selected], cross[selected], z[selected])
                    )
                )
    return np.concatenate(chunks) if chunks else np.empty((0, 3), dtype=np.float64)


def _segment_metrics(
    segment_id: str,
    train_path: Path,
    holdout_report_path: Path,
    holdout_cloud_path: Path,
    maximum_match_distance_m: float,
    core_length_m: float,
    sample_step_m: float,
    voxel_size_m: float,
    holdout_fraction: float,
    holdout_seed: int,
) -> dict[str, Any]:
    train = load_json(train_path)
    holdout = load_json(holdout_report_path)
    if train.get("schema_version") != "railway.rail-candidates.v1":
        raise ValueError(f"Unsupported train rail report: {train_path}")
    if holdout.get("schema_version") != "railway.rail-candidates.v1":
        raise ValueError(f"Unsupported holdout rail report: {holdout_report_path}")
    if train.get("segment_id") != segment_id or holdout.get("segment_id") != segment_id:
        raise ValueError(f"Segment id mismatch for {segment_id}")
    train_frame = CorridorFrame.from_json(train["frame"])
    holdout_frame = CorridorFrame.from_json(holdout["frame"])
    frame_difference = max(
        float(np.max(np.abs(train_frame.origin_xy - holdout_frame.origin_xy))),
        float(np.max(np.abs(train_frame.along_xy - holdout_frame.along_xy))),
        float(np.max(np.abs(train_frame.cross_xy - holdout_frame.cross_xy))),
    )
    if frame_difference > 1e-9:
        raise ValueError(
            f"Train and holdout reports use different corridor frames: {segment_id}"
        )
    train_range = tuple(float(value) for value in train["longitudinal_range_m"])
    holdout_range = tuple(float(value) for value in holdout["longitudinal_range_m"])
    common_start = max(train_range[0], holdout_range[0])
    common_end = min(train_range[1], holdout_range[1])
    if common_end <= common_start:
        raise ValueError(f"Train and holdout ranges do not overlap: {segment_id}")
    midpoint = (common_start + common_end) / 2.0
    half_core = min(core_length_m / 2.0, (common_end - common_start) / 2.0)
    core_range = midpoint - half_core, midpoint + half_core

    train_lines = list(train.get("rail_lines", []))
    holdout_lines = list(holdout.get("rail_lines", []))
    train_cross = [_line_value(item, midpoint, "cross") for item in train_lines]
    holdout_cross = [_line_value(item, midpoint, "cross") for item in holdout_lines]
    line_matches, unmatched_train, unmatched_holdout = _match_records(
        train_cross, holdout_cross, maximum_match_distance_m
    )
    lateral_differences = [item[2] for item in line_matches]
    vertical_differences = [
        abs(
            _line_value(train_lines[train_index], midpoint, "z")
            - _line_value(holdout_lines[holdout_index], midpoint, "z")
        )
        for train_index, holdout_index, _ in line_matches
    ]

    train_pairs = list(train.get("rail_pairs", []))
    holdout_pairs = list(holdout.get("rail_pairs", []))
    train_pair_centers = [
        float(np.mean(item["observed_cross_positions_m"])) for item in train_pairs
    ]
    holdout_pair_centers = [
        float(np.mean(item["observed_cross_positions_m"])) for item in holdout_pairs
    ]
    pair_matches, unmatched_train_pairs, unmatched_holdout_pairs = _match_records(
        train_pair_centers, holdout_pair_centers, maximum_match_distance_m * 2.0
    )
    gauge_differences = [
        abs(
            float(train_pairs[train_index]["observed_gauge_m"])
            - float(holdout_pairs[holdout_index]["observed_gauge_m"])
        )
        for train_index, holdout_index, _ in pair_matches
    ]

    z_range = tuple(float(value) for value in train["search_z_range_m"])
    holdout_points = _search_holdout_points(
        holdout_cloud_path, train_frame, core_range, z_range
    )
    support_distances: list[float] = []
    sample_count = 0
    if len(holdout_points) and train_lines:
        tree = cKDTree(holdout_points)
        sample_longitudinal = np.arange(
            core_range[0], core_range[1] + sample_step_m * 0.5, sample_step_m
        )
        samples: list[np.ndarray] = []
        for line in train_lines:
            cross = (
                float(line["cross_fit_slope_m_per_m"]) * sample_longitudinal
                + float(line["cross_fit_intercept_m"])
            )
            z = (
                float(line["z_fit_slope_m_per_m"]) * sample_longitudinal
                + float(line["z_fit_intercept_m"])
            )
            samples.append(np.column_stack((sample_longitudinal, cross, z)))
        model_samples = np.concatenate(samples)
        world_xy = train_frame.world_xy(model_samples[:, 0], model_samples[:, 1])
        assigned_to_holdout = spatial_voxel_holdout_mask(
            world_xy[:, 0],
            world_xy[:, 1],
            model_samples[:, 2],
            voxel_size_m,
            holdout_fraction,
            holdout_seed,
        )
        model_samples = model_samples[assigned_to_holdout]
        distances, _ = tree.query(model_samples, k=1, workers=-1)
        support_distances = [float(value) for value in distances]
        sample_count = len(model_samples)

    return {
        "segment_id": segment_id,
        "train_report": str(train_path),
        "train_report_sha256": sha256_file(train_path),
        "holdout_report": str(holdout_report_path),
        "holdout_report_sha256": sha256_file(holdout_report_path),
        "holdout_cloud": str(holdout_cloud_path),
        "holdout_cloud_sha256": sha256_file(holdout_cloud_path),
        "frame_max_abs_difference": frame_difference,
        "evaluation_longitudinal_range_m": list(core_range),
        "holdout_search_point_count": len(holdout_points),
        "line_repeatability": {
            "train_count": len(train_lines),
            "holdout_count": len(holdout_lines),
            "matched_count": len(line_matches),
            "train_match_recall": (
                len(line_matches) / len(train_lines) if train_lines else None
            ),
            "holdout_match_precision": (
                len(line_matches) / len(holdout_lines) if holdout_lines else None
            ),
            "unmatched_train_indexes": unmatched_train,
            "unmatched_holdout_indexes": unmatched_holdout,
            "lateral_difference": _stats(lateral_differences),
            "vertical_difference": _stats(vertical_differences),
        },
        "track_pair_repeatability": {
            "train_count": len(train_pairs),
            "holdout_count": len(holdout_pairs),
            "matched_count": len(pair_matches),
            "unmatched_train_indexes": unmatched_train_pairs,
            "unmatched_holdout_indexes": unmatched_holdout_pairs,
            "raw_gauge_difference": _stats(gauge_differences),
        },
        "model_to_holdout_support": {
            "sample_count": sample_count,
            "distance": _stats(support_distances),
            "coverage_at_0_02m": (
                sum(value <= 0.02 for value in support_distances)
                / len(support_distances)
                if support_distances
                else None
            ),
            "coverage_at_0_05m": (
                sum(value <= 0.05 for value in support_distances)
                / len(support_distances)
                if support_distances
                else None
            ),
            "coverage_at_0_10m": (
                sum(value <= 0.10 for value in support_distances)
                / len(support_distances)
                if support_distances
                else None
            ),
        },
        "_raw_metrics": {
            "lateral_difference_m": lateral_differences,
            "vertical_difference_m": vertical_differences,
            "raw_gauge_difference_m": gauge_differences,
            "support_distance_m": support_distances,
        },
    }


def evaluate_rail_holdout(
    train_reports: list[tuple[str, str | Path]],
    holdout_reports: list[tuple[str, str | Path]],
    holdout_clouds: list[tuple[str, str | Path]],
    holdout_manifest_path: str | Path,
    output_path: str | Path,
    maximum_match_distance_m: float = 0.30,
    core_length_m: float = 50.0,
    sample_step_m: float = 0.25,
) -> Path:
    if maximum_match_distance_m <= 0 or core_length_m <= 0 or sample_step_m <= 0:
        raise ValueError("Distances and sampling step must be positive")
    train = {segment: Path(path).resolve() for segment, path in train_reports}
    reports = {segment: Path(path).resolve() for segment, path in holdout_reports}
    clouds = {segment: Path(path).resolve() for segment, path in holdout_clouds}
    if not train or set(train) != set(reports) or set(train) != set(clouds):
        raise ValueError("Train report, holdout report and holdout cloud segments must match")
    holdout_manifest_source = Path(holdout_manifest_path).resolve()
    holdout_manifest = load_json(holdout_manifest_source)
    if holdout_manifest.get("schema_version") != "railway.point-holdout.v1":
        raise ValueError("Unsupported point holdout manifest")
    declared_clouds = {
        str(item["segment_id"]): Path(str(item["holdout"]["path"])).resolve()
        for item in holdout_manifest.get("segments", [])
    }
    if any(declared_clouds.get(segment) != path for segment, path in clouds.items()):
        raise ValueError("Holdout cloud does not match the frozen holdout manifest")
    voxel_size_m = float(holdout_manifest["voxel_size_m"])
    holdout_fraction = float(holdout_manifest["requested_holdout_fraction"])
    holdout_seed = int(holdout_manifest["seed"])
    segment_results = [
        _segment_metrics(
            segment,
            train[segment],
            reports[segment],
            clouds[segment],
            maximum_match_distance_m,
            core_length_m,
            sample_step_m,
            voxel_size_m,
            holdout_fraction,
            holdout_seed,
        )
        for segment in sorted(train)
    ]
    lateral = [
        value
        for result in segment_results
        for value in result["_raw_metrics"]["lateral_difference_m"]
    ]
    vertical = [
        value
        for result in segment_results
        for value in result["_raw_metrics"]["vertical_difference_m"]
    ]
    gauge = [
        value
        for result in segment_results
        for value in result["_raw_metrics"]["raw_gauge_difference_m"]
    ]
    support = [
        value
        for result in segment_results
        for value in result["_raw_metrics"]["support_distance_m"]
    ]
    total_train_lines = sum(
        item["line_repeatability"]["train_count"] for item in segment_results
    )
    total_holdout_lines = sum(
        item["line_repeatability"]["holdout_count"] for item in segment_results
    )
    total_matched_lines = sum(
        item["line_repeatability"]["matched_count"] for item in segment_results
    )
    support_samples = sum(
        item["model_to_holdout_support"]["sample_count"] for item in segment_results
    )
    report = {
        "schema_version": "railway.rail-holdout-evaluation.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "evaluation_type": "annotation_free_split_repeatability_and_support",
        "holdout_manifest": str(holdout_manifest_source),
        "holdout_manifest_sha256": sha256_file(holdout_manifest_source),
        "parameters": {
            "maximum_match_distance_m": maximum_match_distance_m,
            "core_length_m": core_length_m,
            "sample_step_m": sample_step_m,
            "voxel_size_m": voxel_size_m,
            "holdout_fraction": holdout_fraction,
            "holdout_seed": holdout_seed,
        },
        "summary": {
            "segment_count": len(segment_results),
            "train_line_count": total_train_lines,
            "holdout_line_count": total_holdout_lines,
            "matched_line_count": total_matched_lines,
            "train_line_match_recall": (
                total_matched_lines / total_train_lines if total_train_lines else None
            ),
            "holdout_line_match_precision": (
                total_matched_lines / total_holdout_lines
                if total_holdout_lines
                else None
            ),
            "model_to_holdout_sample_count": support_samples,
            "line_lateral_repeatability": _stats(lateral),
            "line_vertical_repeatability": _stats(vertical),
            "raw_gauge_repeatability": _stats(gauge),
            "model_to_holdout_support_distance": _stats(support),
            "model_to_holdout_coverage_at_0_02m": (
                sum(value <= 0.02 for value in support) / len(support)
                if support
                else None
            ),
            "model_to_holdout_coverage_at_0_05m": (
                sum(value <= 0.05 for value in support) / len(support)
                if support
                else None
            ),
            "model_to_holdout_coverage_at_0_10m": (
                sum(value <= 0.10 for value in support) / len(support)
                if support
                else None
            ),
        },
        "segments": segment_results,
        "limitations": [
            "This is split repeatability and raw-point support, not semantic ground truth.",
            "The same frozen detector is run independently on train and holdout voxels.",
            "Model-to-holdout distances measure support for predicted rail lines only.",
            "Support is scored only where the frozen voxel hash assigns the model sample to holdout.",
            "No asset precision/recall or absolute survey accuracy is claimed.",
        ],
    }
    for result in segment_results:
        result.pop("_raw_metrics")
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite rail holdout report: {output}")
    write_json(output, report)
    return output
