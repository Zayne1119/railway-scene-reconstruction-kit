"""Label-blind, straight-corridor rail candidates from an XYZ array.

This exploratory adapter deliberately has no trajectory, LAS, file, label or
instance API.  Its output is a geometric hypothesis, not a surveyed rail model.
The two modes share their entire local candidate pool; pairing only selects a
subset.  Unsupported longitudinal bins are never filled with output polylines.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks

from .algorithms.rail_candidates import (
    _paired_longitudinal_support,
    _robust_linear_fit,
)

PUBLIC_RAIL_CANDIDATE_MODES = ("height_prominence", "paired_geometry")

# Detection dimensions/thresholds below are copied from the generic
# resources/rail-detection.default.json, not selected using a public sample.
# Explicit adapter additions are the PCA/row-baseline and resource limits.
# Nominal gauge correction and customer/trajectory-specific settings are absent.
DEFAULT_PUBLIC_RAIL_POLICY: dict[str, float | int] = {
    "z_percentile_min": 1.0,
    "z_percentile_max": 35.0,
    "z_min_offset_m": 0.0,
    "z_max_offset_m": 0.5,
    "cross_bin_m": 0.05,
    "longitudinal_bin_m": 0.25,
    "gaussian_sigma_bins": 1.0,
    "height_baseline_filter_bins": 11,
    "minimum_height_prominence_m": 0.07,
    "minimum_line_coverage": 0.12,
    "minimum_peak_prominence": 0.02,
    "minimum_peak_spacing_m": 0.35,
    "rail_pair_minimum_m": 1.35,
    "rail_pair_maximum_m": 1.65,
    "candidate_half_width_m": 0.10,
    "candidate_vertical_below_rail_m": 0.12,
    "candidate_vertical_above_rail_m": 0.05,
    "maximum_rail_top_crosslevel_m": 0.20,
    "pair_support_longitudinal_bin_m": 0.25,
    "minimum_pair_joint_coverage": 0.10,
    "maximum_pair_internal_gap_m": 5.0,
    "maximum_pair_asymmetric_support_ratio": 0.25,
    "minimum_track_center_spacing_m": 3.0,
    "maximum_track_count": 8,
    "row_baseline_quantile": 0.20,
    "minimum_direction_eigenvalue_ratio": 2.0,
    "minimum_longitudinal_extent_m": 2.0,
    "minimum_search_point_count": 50,
    "minimum_line_sample_count": 4,
    "maximum_cross_fit_residual_p90_m": 0.10,
    "maximum_point_count": 2_000_000,
    "maximum_grid_cells": 4_000_000,
    "maximum_candidate_count": 64,
    "maximum_coordinate_extent_m": 1_000_000.0,
}


def _settings(policy: dict[str, Any] | None) -> dict[str, float | int]:
    if policy is not None and not isinstance(policy, dict):
        raise ValueError("policy must be a dictionary or None")
    if policy is not None and any(not isinstance(key, str) for key in policy):
        raise ValueError("Policy keys must be strings")
    unknown = set(policy or {}) - DEFAULT_PUBLIC_RAIL_POLICY.keys()
    if unknown:
        raise ValueError(f"Unknown public rail policy keys: {sorted(unknown)}")
    value = {**DEFAULT_PUBLIC_RAIL_POLICY, **(policy or {})}
    integer_keys = {
        "height_baseline_filter_bins",
        "maximum_track_count",
        "minimum_search_point_count",
        "minimum_line_sample_count",
        "maximum_point_count",
        "maximum_grid_cells",
        "maximum_candidate_count",
    }
    nonnegative = {
        "gaussian_sigma_bins",
        "minimum_peak_prominence",
        "candidate_vertical_below_rail_m",
        "candidate_vertical_above_rail_m",
        "maximum_rail_top_crosslevel_m",
        "maximum_pair_internal_gap_m",
        "minimum_track_center_spacing_m",
    }
    ratios = {
        "minimum_line_coverage",
        "minimum_pair_joint_coverage",
        "maximum_pair_asymmetric_support_ratio",
        "row_baseline_quantile",
    }
    for key, setting in value.items():
        if (
            isinstance(setting, bool)
            or not isinstance(setting, (int, float))
            or not math.isfinite(setting)
        ):
            raise ValueError(f"{key} must be a finite number")
        if key in integer_keys:
            if not isinstance(setting, int) or setting < 1:
                raise ValueError(f"{key} must be a positive integer")
        elif key in ratios:
            if not 0.0 <= setting <= 1.0:
                raise ValueError(f"{key} must be between zero and one")
        elif key in {"z_percentile_min", "z_percentile_max"}:
            if not 0.0 <= setting <= 100.0:
                raise ValueError(f"{key} must be between zero and 100")
        elif key not in {"z_min_offset_m", "z_max_offset_m"} and (
            setting < 0.0 or (key not in nonnegative and setting == 0.0)
        ):
            raise ValueError(f"{key} is outside its positive/nonnegative range")
    if value["z_percentile_min"] >= value["z_percentile_max"]:
        raise ValueError("Z percentiles must be strictly increasing")
    if value["rail_pair_minimum_m"] >= value["rail_pair_maximum_m"]:
        raise ValueError("Pair spacing bounds must be strictly increasing")
    if value["pair_support_longitudinal_bin_m"] < value["longitudinal_bin_m"]:
        raise ValueError("Pair support bin cannot be smaller than detection bin")
    if value["height_baseline_filter_bins"] < 3 or value["height_baseline_filter_bins"] % 2 == 0:
        raise ValueError("height_baseline_filter_bins must be odd and at least three")
    if value["minimum_line_sample_count"] < 2:
        raise ValueError("minimum_line_sample_count must be at least two")
    if value["minimum_direction_eigenvalue_ratio"] <= 1.0:
        raise ValueError("minimum_direction_eigenvalue_ratio must exceed one")
    return value


def _runs(rows: np.ndarray) -> list[np.ndarray]:
    if not len(rows):
        return []
    return list(np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1))


def extract_public_rail_candidates(
    xyz: np.ndarray,
    policy: dict[str, Any] | None = None,
    mode: str = "paired_geometry",
) -> dict[str, Any]:
    """Return an N-element boolean ``point_mask`` and JSON-safe ``report``.

    ``xyz`` must be a finite, real numeric N-by-3 ndarray in metres. No side
    channels or labels are accepted. Invalid inputs/policy raise ValueError;
    empty, ambiguous, oversized-grid and candidate-free scenes return an empty
    mask with a reason. Point-count limits are checked before any array copy.
    Height/PCA assumptions are suitable only for exploratory near-straight
    corridor candidates. Scores are ranking statistics, never probabilities.
    """
    if mode not in PUBLIC_RAIL_CANDIDATE_MODES:
        raise ValueError(f"mode must be one of {PUBLIC_RAIL_CANDIDATE_MODES}")
    config = _settings(policy)
    if (
        not isinstance(xyz, np.ndarray)
        or xyz.ndim != 2
        or xyz.shape[1] != 3
        or not np.issubdtype(xyz.dtype, np.number)
        or np.issubdtype(xyz.dtype, np.complexfloating)
    ):
        raise ValueError("xyz must be a real numeric N-by-3 ndarray")
    if len(xyz) > config["maximum_point_count"]:
        raise ValueError("Point count exceeds maximum_point_count")
    if not np.all(np.isfinite(xyz)):
        raise ValueError("xyz must contain only finite coordinates")
    points = np.asarray(xyz, dtype=np.float64)
    mask = np.zeros(len(points), dtype=bool)
    report: dict[str, Any] = {
        "schema_version": "railway.public-rail-candidates.v1",
        "mode": mode,
        "status": "no_candidates",
        "policy": config,
        "input_point_count": len(points),
        "candidate_point_count": 0,
        "candidate_pool": [],
        "selected_candidate_ids": [],
        "rail_pairs": [],
        "frame": None,
        "diagnostics": {},
        "disclosure": {
            "input_fields": ["x", "y", "z"],
            "uses_labels": False,
            "uses_instance_ids": False,
            "uses_camera_trajectory": False,
            "nominal_gauge_correction": False,
            "score_semantics": "uncalibrated_geometric_ranking_statistic",
            "geometry_role": "exploratory_candidates_not_surveyed_rail_geometry",
            "limitations": [
                "single_straight_corridor_direction_may_miss_curves_and_branches",
                "global_low_height_window_may_miss_sloping_or_elevated_rails",
                "height_maxima_and_density_can_create_nonrail_candidates",
                "spacing_is_fitted_center_spacing_not_measured_clear_gauge",
            ],
        },
    }
    result = {"point_mask": mask, "report": report}
    if len(points) < config["minimum_search_point_count"]:
        report["diagnostics"]["empty_reason"] = "insufficient_input_points"
        return result

    # A local origin avoids subtracting georeferenced heights in float32 grids.
    # Bound an extreme finite input before covariance/grid dimension arithmetic.
    with np.errstate(over="ignore", invalid="ignore"):
        origin = np.median(points, axis=0)
        local = points - origin
    if (
        not np.all(np.isfinite(local))
        or float(np.max(np.abs(local))) > config["maximum_coordinate_extent_m"]
    ):
        report["diagnostics"]["empty_reason"] = "coordinate_extent_resource_limit"
        return result
    z = local[:, 2]
    lower, upper = np.percentile(z, [config["z_percentile_min"], config["z_percentile_max"]])
    lower += config["z_min_offset_m"]
    upper += config["z_max_offset_m"]
    with np.errstate(over="ignore", invalid="ignore"):
        world_z_bounds = np.array([lower, upper]) + origin[2]
    if not np.all(np.isfinite(world_z_bounds)):
        raise ValueError("Policy height bounds overflow finite world coordinates")
    if upper <= lower:
        report["diagnostics"]["empty_reason"] = "degenerate_height_window"
        return result
    search = (z >= lower) & (z <= upper)
    search_count = int(np.count_nonzero(search))
    report["diagnostics"].update(
        {
            "search_point_count": search_count,
            "search_z_range_m": world_z_bounds.tolist(),
            "height_window_source": "all_input_xyz_percentiles",
        }
    )
    if search_count < config["minimum_search_point_count"]:
        report["diagnostics"]["empty_reason"] = "insufficient_height_window_points"
        return result

    direction_points = local[search, :2]
    xy_offset = np.mean(direction_points, axis=0)
    centered = direction_points - xy_offset
    covariance = centered.T @ centered / search_count
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    ratio = float(eigenvalues[1] / max(float(eigenvalues[0]), 1e-12))
    report["diagnostics"]["direction_eigenvalue_ratio"] = ratio
    if eigenvalues[1] <= 1e-12 or ratio < config["minimum_direction_eigenvalue_ratio"]:
        report["diagnostics"]["empty_reason"] = "ambiguous_or_degenerate_xy_direction"
        return result
    direction = eigenvectors[:, 1]
    if direction[int(np.argmax(np.abs(direction)))] < 0.0:
        direction = -direction
    lateral = np.array([-direction[1], direction[0]])
    xy_local = local[:, :2] - xy_offset
    longitudinal = xy_local @ direction
    cross = xy_local @ lateral
    origin[:2] += xy_offset
    report["frame"] = {
        "method": "xy_pca_of_label_blind_low_height_window",
        "origin_xyz": origin.tolist(),
        "longitudinal_xy": direction.tolist(),
        "cross_xy": lateral.tolist(),
        "direction_point_count": search_count,
    }
    if float(np.ptp(longitudinal[search])) < config["minimum_longitudinal_extent_m"]:
        report["diagnostics"]["empty_reason"] = "insufficient_longitudinal_extent"
        return result

    cross_bin = float(config["cross_bin_m"])
    long_bin = float(config["longitudinal_bin_m"])
    # Symmetric bin boundaries make PCA-axis sign reversal a grid reflection,
    # rather than changing the partition. Padding permits boundary line peaks.
    cross_extent = float(np.max(np.abs(cross[search])))
    long_extent = float(np.max(np.abs(longitudinal[search])))
    if (
        cross_extent > config["maximum_grid_cells"] * cross_bin
        or long_extent > config["maximum_grid_cells"] * long_bin
    ):
        report["diagnostics"]["empty_reason"] = "grid_cell_resource_limit"
        return result
    cross_radius = math.ceil(cross_extent / cross_bin) + 2
    long_radius = math.ceil(long_extent / long_bin)
    nx, ny = 2 * cross_radius + 1, 2 * long_radius + 1
    report["diagnostics"].update({"grid_shape": [ny, nx], "grid_cell_count": nx * ny})
    if nx * ny > config["maximum_grid_cells"]:
        report["diagnostics"]["empty_reason"] = "grid_cell_resource_limit"
        return result
    cross_min = -(cross_radius + 0.5) * cross_bin
    long_min = -(long_radius + 0.5) * long_bin
    all_rows = np.floor((longitudinal - long_min) / long_bin + 1e-9).astype(np.int64)
    rows = np.clip(all_rows[search], 0, ny - 1)
    cols = np.clip(
        np.floor((cross[search] - cross_min) / cross_bin + 1e-9).astype(np.int64), 0, nx - 1
    )
    search_z = z[search]
    row_baseline = np.full(ny, float(lower))
    order = np.argsort(rows, kind="stable")
    unique_rows, starts, counts = np.unique(rows[order], return_index=True, return_counts=True)
    for row, start, count in zip(unique_rows, starts, counts, strict=True):
        row_baseline[row] = np.quantile(
            search_z[order[start : start + count]], config["row_baseline_quantile"]
        )
    occupied = np.zeros((ny, nx), dtype=bool)
    occupied[rows, cols] = True
    height = np.broadcast_to(row_baseline[:, None], (ny, nx)).copy()
    np.maximum.at(height, (rows, cols), search_z)
    baseline = median_filter(
        height, size=(1, int(config["height_baseline_filter_bins"])), mode="nearest"
    )
    prominence = height - baseline
    prominent = occupied & (prominence >= config["minimum_height_prominence_m"])
    coverage = prominent.mean(axis=0)
    smooth = (
        gaussian_filter1d(coverage, sigma=float(config["gaussian_sigma_bins"]))
        if config["gaussian_sigma_bins"] > 0.0
        else coverage
    )
    peaks, _ = find_peaks(
        smooth,
        height=float(config["minimum_line_coverage"]),
        prominence=float(config["minimum_peak_prominence"]),
        distance=max(1, round(config["minimum_peak_spacing_m"] / cross_bin)),
    )
    report["diagnostics"]["local_peak_count_before_resource_limit"] = len(peaks)
    ranked = sorted(peaks.tolist(), key=lambda peak: (-float(smooth[peak]), peak))
    peaks = np.asarray(sorted(ranked[: int(config["maximum_candidate_count"])]), dtype=np.int64)
    report["diagnostics"]["candidate_pool_truncated"] = len(ranked) > len(peaks)
    peak_radius = max(1, math.ceil(config["candidate_half_width_m"] / cross_bin))
    records: list[dict[str, Any]] = []
    internal: list[dict[str, Any]] = []

    def line_point_mask(item: dict[str, Any]) -> np.ndarray:
        cross_fit, z_fit = item["cross_fit"], item["z_fit"]
        expected_cross = cross_fit["slope"] * longitudinal + cross_fit["intercept"]
        expected_z = z_fit["slope"] * longitudinal + z_fit["intercept"]
        in_grid = (all_rows >= 0) & (all_rows < ny)
        support = item["supported_rows"][np.clip(all_rows, 0, ny - 1)]
        return (
            search
            & in_grid
            & support
            & (np.abs(cross - expected_cross) <= config["candidate_half_width_m"])
            & (z >= expected_z - config["candidate_vertical_below_rail_m"])
            & (z <= expected_z + config["candidate_vertical_above_rail_m"])
        )

    def world_point(long_value: float, item: dict[str, Any]) -> list[float]:
        cross_value = item["cross_fit"]["slope"] * long_value + item["cross_fit"]["intercept"]
        xy = origin[:2] + direction * long_value + lateral * cross_value
        height_value = origin[2] + item["z_fit"]["slope"] * long_value + item["z_fit"]["intercept"]
        return [float(xy[0]), float(xy[1]), float(height_value)]

    for peak in peaks.tolist():
        left, right = max(0, peak - peak_radius), min(nx, peak + peak_radius + 1)
        support = np.any(prominent[:, left:right], axis=1)
        supported_rows = np.flatnonzero(support)
        if len(supported_rows) < config["minimum_line_sample_count"]:
            continue
        sample_cols = []
        for row in supported_rows:
            eligible = np.flatnonzero(prominent[row, left:right]) + left
            column = min(
                eligible.tolist(), key=lambda col: (abs(col - peak), -float(prominence[row, col]))
            )
            sample_cols.append(column)
        sample_long = long_min + (supported_rows + 0.5) * long_bin
        sample_cross = cross_min + (np.asarray(sample_cols) + 0.5) * cross_bin
        sample_z = height[supported_rows, sample_cols]
        cross_fit = _robust_linear_fit(sample_long, sample_cross)
        z_fit = _robust_linear_fit(sample_long, sample_z)
        if cross_fit["residual_p90_m"] > config["maximum_cross_fit_residual_p90_m"]:
            continue
        item = {"peak": peak, "supported_rows": support, "cross_fit": cross_fit, "z_fit": z_fit}
        point_selection = line_point_mask(item)
        point_count = int(np.count_nonzero(point_selection))
        if point_count < config["minimum_line_sample_count"]:
            continue
        selected_rows = all_rows[point_selection]
        selected_long = longitudinal[point_selection]
        polylines = []
        for run in _runs(np.unique(selected_rows)):
            run_points = (selected_rows >= run[0]) & (selected_rows <= run[-1])
            minimum, maximum = (
                float(np.min(selected_long[run_points])),
                float(np.max(selected_long[run_points])),
            )
            if maximum - minimum > 1e-9:
                polylines.append([world_point(minimum, item), world_point(maximum, item)])
        record = {
            "id": f"CANDIDATE-{len(records) + 1:04d}",
            "grid_peak_index": peak,
            "cross_position_m": float(cross_fit["intercept"]),
            "ranking_score": float(smooth[peak]),
            "longitudinal_support_ratio": float(np.mean(support)),
            "supported_bin_count": len(supported_rows),
            "point_count": point_count,
            "cross_fit": cross_fit,
            "z_fit_local_origin": z_fit,
            "polylines_xyz": polylines,
            "unsupported_bins_bridged": 0,
        }
        records.append(record)
        internal.append(item)

    report["candidate_pool"] = records
    pair_candidates: list[dict[str, Any]] = []
    for left_index, left in enumerate(internal):
        for right_index in range(left_index + 1, len(internal)):
            right = internal[right_index]
            jointly_supported = np.flatnonzero(left["supported_rows"] & right["supported_rows"])
            if not len(jointly_supported):
                continue
            long_values = long_min + (jointly_supported + 0.5) * long_bin
            separation = (
                (right["cross_fit"]["slope"] - left["cross_fit"]["slope"]) * long_values
                + right["cross_fit"]["intercept"]
                - left["cross_fit"]["intercept"]
            )
            observed_spacing = float(np.median(separation))
            crosslevel = float(
                np.max(
                    np.abs(
                        (right["z_fit"]["slope"] - left["z_fit"]["slope"]) * long_values
                        + right["z_fit"]["intercept"]
                        - left["z_fit"]["intercept"]
                    )
                )
            )
            if (
                not config["rail_pair_minimum_m"]
                <= observed_spacing
                <= config["rail_pair_maximum_m"]
            ):
                continue
            if crosslevel > config["maximum_rail_top_crosslevel_m"]:
                continue
            continuity = _paired_longitudinal_support(
                prominent,
                left["peak"],
                right["peak"],
                peak_radius,
                long_bin,
                float(config["pair_support_longitudinal_bin_m"]),
            )
            passed = (
                continuity["joint_support_ratio"] >= config["minimum_pair_joint_coverage"]
                and continuity["asymmetric_support_ratio"]
                <= config["maximum_pair_asymmetric_support_ratio"]
                and continuity["maximum_internal_joint_gap_m"]
                <= config["maximum_pair_internal_gap_m"]
            )
            pair_candidates.append(
                {
                    "candidate_ids": [records[left_index]["id"], records[right_index]["id"]],
                    "candidate_indexes": [left_index, right_index],
                    "observed_center_spacing_m": observed_spacing,
                    "observed_center_spacing_range_m": [
                        float(np.min(separation)),
                        float(np.max(separation)),
                    ],
                    "maximum_fitted_crosslevel_m": crosslevel,
                    "cross_center_m": float(
                        (left["cross_fit"]["intercept"] + right["cross_fit"]["intercept"]) / 2.0
                    ),
                    "ranking_score": float(
                        continuity["joint_support_ratio"]
                        - continuity["asymmetric_support_ratio"]
                        + min(
                            records[left_index]["ranking_score"],
                            records[right_index]["ranking_score"],
                        )
                    ),
                    "shared_support_pass": passed,
                    "nominal_gauge_correction": False,
                    **continuity,
                }
            )
    report["diagnostics"]["gauge_compatible_pair_candidates"] = pair_candidates
    selected: set[int] = set()
    selected_centers: list[float] = []
    pairs = []
    for pair in sorted(
        pair_candidates, key=lambda item: (-item["ranking_score"], item["candidate_indexes"])
    ):
        if not pair["shared_support_pass"] or selected.intersection(pair["candidate_indexes"]):
            continue
        if any(
            abs(pair["cross_center_m"] - center) < config["minimum_track_center_spacing_m"]
            for center in selected_centers
        ):
            continue
        selected.update(pair["candidate_indexes"])
        selected_centers.append(pair["cross_center_m"])
        pairs.append(pair)
        if len(pairs) >= config["maximum_track_count"]:
            break
    if mode == "height_prominence":
        selected = set(range(len(records)))
    else:
        report["rail_pairs"] = pairs
    for index in sorted(selected):
        mask |= line_point_mask(internal[index])
    report["selected_candidate_ids"] = [records[index]["id"] for index in sorted(selected)]
    report["candidate_point_count"] = int(np.count_nonzero(mask))
    report["diagnostics"].update(
        {
            "candidate_pool_count": len(records),
            "selected_line_count": len(selected),
            "selected_pair_count": len(report["rail_pairs"]),
            "grid_longitudinal_origin_m": long_min,
            "grid_cross_origin_m": cross_min,
        }
    )
    if np.any(mask):
        report["status"] = "exploratory_geometry_candidates"
    else:
        report["diagnostics"]["empty_reason"] = (
            "no_local_lines" if not records else "no_pair_with_shared_support"
        )
    return result
