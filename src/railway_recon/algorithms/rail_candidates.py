from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks

from ..camera import camera_trajectory, load_camera_rows
from ..config import ProjectConfig
from ..geometry import (
    fit_corridor_frame_dominant_axis_regression,
    fit_segment_corridor_frame,
)
from ..io import load_json, write_json


def _robust_linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    if len(x) != len(y) or len(x) < 2 or float(np.ptp(x)) <= 1e-9:
        return {
            "slope": 0.0,
            "intercept": float(np.median(y)) if len(y) else math.nan,
            "sample_count": len(y),
            "residual_p90_m": math.nan,
        }
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    limit = max(0.02, 3.0 * 1.4826 * mad)
    keep = np.abs(residual - median) <= limit
    if int(np.count_nonzero(keep)) >= 2 and float(np.ptp(x[keep])) > 1e-9:
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        residual = y[keep] - (slope * x[keep] + intercept)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "sample_count": len(residual),
        "residual_p90_m": float(np.percentile(np.abs(residual), 90)),
    }


def _longest_false_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values.tolist():
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _paired_longitudinal_support(
    prominent: np.ndarray,
    left_peak: int,
    right_peak: int,
    peak_radius: int,
    longitudinal_bin_m: float,
    support_bin_m: float | None = None,
) -> dict[str, float | int]:
    """Measure whether two candidate rails are supported in the same route bins.

    This adapts GISLab RailTrack's paired-growth idea into a vectorized,
    non-generative diagnostic. It never invents geometry across an unsupported
    interval; it records joint support, asymmetric support and gaps for ranking
    and downstream quality gates.
    """

    if prominent.ndim != 2 or prominent.shape[0] == 0 or prominent.shape[1] == 0:
        raise ValueError("prominent grid must be a non-empty two-dimensional array")
    if longitudinal_bin_m <= 0:
        raise ValueError("longitudinal_bin_m must be positive")
    if peak_radius < 0:
        raise ValueError("peak_radius must be non-negative")

    width = prominent.shape[1]

    def row_support(peak: int) -> np.ndarray:
        if peak < 0 or peak >= width:
            raise ValueError("rail peak lies outside the prominence grid")
        left = max(0, peak - peak_radius)
        right = min(width, peak + peak_radius + 1)
        return np.any(prominent[:, left:right], axis=1)

    left = row_support(left_peak)
    right = row_support(right_peak)
    aggregation = 1
    if support_bin_m is not None:
        if support_bin_m < longitudinal_bin_m:
            raise ValueError(
                "pair_support_longitudinal_bin_m cannot be smaller than the detection bin"
            )
        aggregation = max(1, round(support_bin_m / longitudinal_bin_m))
    if aggregation > 1:
        pad = (-len(left)) % aggregation
        left = np.pad(left, (0, pad), constant_values=False).reshape(-1, aggregation).any(axis=1)
        right = np.pad(right, (0, pad), constant_values=False).reshape(-1, aggregation).any(axis=1)
    effective_bin_m = longitudinal_bin_m * aggregation
    joint = left & right
    asymmetric = left ^ right
    supported = np.flatnonzero(joint)
    maximum_gap_bins = _longest_false_run(joint)
    if len(supported) >= 2:
        internal = joint[int(supported[0]) : int(supported[-1]) + 1]
        maximum_internal_gap_bins = _longest_false_run(internal)
    else:
        maximum_internal_gap_bins = len(joint)

    return {
        "longitudinal_bin_count": len(joint),
        "support_longitudinal_bin_m": float(effective_bin_m),
        "left_support_ratio": float(np.mean(left)),
        "right_support_ratio": float(np.mean(right)),
        "joint_support_ratio": float(np.mean(joint)),
        "asymmetric_support_ratio": float(np.mean(asymmetric)),
        "joint_supported_bin_count": int(np.count_nonzero(joint)),
        "maximum_joint_gap_m": float(maximum_gap_bins * effective_bin_m),
        "maximum_internal_joint_gap_m": float(
            maximum_internal_gap_bins * effective_bin_m
        ),
    }


def _pair_continuity_settings(
    config: dict[str, Any], longitudinal_bin_m: float = 0.25
) -> dict[str, float]:
    settings = {
        "joint_support_score_weight": float(
            config.get("pair_joint_support_score_weight", 0.0)
        ),
        "asymmetric_support_penalty_weight": float(
            config.get("pair_asymmetric_support_penalty_weight", 0.0)
        ),
        "minimum_joint_coverage": float(
            config.get("minimum_pair_joint_coverage", 0.10)
        ),
        "maximum_internal_gap_m": float(
            config.get("maximum_pair_internal_gap_m", 5.0)
        ),
        "maximum_asymmetric_support_ratio": float(
            config.get("maximum_pair_asymmetric_support_ratio", 0.25)
        ),
        "support_longitudinal_bin_m": float(
            config.get("pair_support_longitudinal_bin_m", longitudinal_bin_m)
        ),
    }
    for name in ("joint_support_score_weight", "asymmetric_support_penalty_weight"):
        if settings[name] < 0.0:
            raise ValueError(f"{name} must be non-negative")
    for name in ("minimum_joint_coverage", "maximum_asymmetric_support_ratio"):
        if not 0.0 <= settings[name] <= 1.0:
            raise ValueError(f"{name} must lie between zero and one")
    if settings["maximum_internal_gap_m"] < 0.0:
        raise ValueError("maximum_internal_gap_m must be non-negative")
    if settings["support_longitudinal_bin_m"] <= 0.0:
        raise ValueError("support_longitudinal_bin_m must be positive")
    return settings


def _ordered_pair_candidates(
    candidates: list[dict[str, Any]], policy: str
) -> list[dict[str, Any]]:
    """Order or reject continuity candidates without mutating the inputs."""

    if policy not in {"diagnostic_only", "prefer_pass", "require_pass"}:
        raise ValueError(
            "pair_continuity_selection_policy must be 'diagnostic_only', "
            "'prefer_pass' or 'require_pass'"
        )
    eligible = (
        [item for item in candidates if item.get("pair_continuity_status") == "pass"]
        if policy == "require_pass"
        else list(candidates)
    )
    if policy == "prefer_pass":
        return sorted(
            eligible,
            key=lambda item: (
                item.get("pair_continuity_status") == "pass",
                float(item["score"]),
            ),
            reverse=True,
        )
    return sorted(eligible, key=lambda item: float(item["score"]), reverse=True)


def _anchored_rail_top_samples(
    longitudinal: np.ndarray,
    cross: np.ndarray,
    z: np.ndarray,
    search: np.ndarray,
    position_m: float,
    anchor_z_m: float,
    long_min_m: float,
    longitudinal_bin_m: float,
    half_width_m: float,
    below_m: float,
    above_m: float,
    quantile: float,
    minimum_bin_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate one rail-top height per longitudinal bin without using maxima."""

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("rail_top_quantile must lie between zero and one")
    if minimum_bin_points < 1:
        raise ValueError("rail_top_minimum_bin_points must be positive")
    selected = (
        search
        & (np.abs(cross - position_m) <= half_width_m)
        & (z >= anchor_z_m - below_m)
        & (z <= anchor_z_m + above_m)
    )
    if not np.any(selected):
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    selected_longitudinal = longitudinal[selected]
    selected_z = z[selected]
    rows = np.floor(
        (selected_longitudinal - long_min_m) / longitudinal_bin_m
    ).astype(np.int64)
    order = np.argsort(rows, kind="stable")
    rows = rows[order]
    selected_z = selected_z[order]
    unique_rows, starts, counts = np.unique(
        rows, return_index=True, return_counts=True
    )
    sample_longitudinal: list[float] = []
    sample_z: list[float] = []
    for row, start, count in zip(
        unique_rows.tolist(), starts.tolist(), counts.tolist(), strict=True
    ):
        if count < minimum_bin_points:
            continue
        sample_longitudinal.append(
            long_min_m + (float(row) + 0.5) * longitudinal_bin_m
        )
        sample_z.append(
            float(np.quantile(selected_z[start : start + count], quantile))
        )
    return (
        np.asarray(sample_longitudinal, dtype=np.float64),
        np.asarray(sample_z, dtype=np.float64),
    )


def _local_rail_top_samples(
    longitudinal: np.ndarray,
    cross: np.ndarray,
    z: np.ndarray,
    search: np.ndarray,
    position_m: float,
    long_min_m: float,
    longitudinal_bin_m: float,
    half_width_m: float,
    quantile: float,
    minimum_bin_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate rail height from the full configured Z search window."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("rail_top_quantile must lie between zero and one")
    if minimum_bin_points < 1:
        raise ValueError("rail_top_minimum_bin_points must be positive")
    selected = search & (np.abs(cross - position_m) <= half_width_m)
    if not np.any(selected):
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    selected_longitudinal = longitudinal[selected]
    selected_z = z[selected]
    rows = np.floor(
        (selected_longitudinal - long_min_m) / longitudinal_bin_m
    ).astype(np.int64)
    order = np.argsort(rows, kind="stable")
    rows = rows[order]
    selected_z = selected_z[order]
    unique_rows, starts, counts = np.unique(
        rows, return_index=True, return_counts=True
    )
    sample_longitudinal: list[float] = []
    sample_z: list[float] = []
    for row, start, count in zip(
        unique_rows.tolist(), starts.tolist(), counts.tolist(), strict=True
    ):
        if count < minimum_bin_points:
            continue
        sample_longitudinal.append(
            long_min_m + (float(row) + 0.5) * longitudinal_bin_m
        )
        sample_z.append(
            float(np.quantile(selected_z[start : start + count], quantile))
        )
    return (
        np.asarray(sample_longitudinal, dtype=np.float64),
        np.asarray(sample_z, dtype=np.float64),
    )


def _pairing_rail_top_z(
    height_grid_top_z: float | None,
    selected_fit_top_z: float | None,
    method: str,
) -> float | None:
    if method == "height_grid_max":
        return height_grid_top_z
    if method == "selected_fit":
        return selected_fit_top_z
    raise ValueError(
        "rail_pair_top_method must be 'height_grid_max' or 'selected_fit'"
    )


def _height_mask(
    z: np.ndarray,
    config: dict[str, Any],
    *,
    estimation_mask: np.ndarray | None = None,
    reference_z: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    if estimation_mask is None:
        estimation_mask = np.ones(len(z), dtype=bool)
    if len(estimation_mask) != len(z):
        raise ValueError("rail height estimation mask must match the point count")
    if not np.any(estimation_mask):
        raise ValueError("rail height estimation mask is empty")
    mode = config.get("z_mode", "percentile")
    if mode == "absolute":
        minimum = float(config["minimum_z"])
        maximum = float(config["maximum_z"])
        mask = (z >= minimum) & (z <= maximum)
    elif mode == "percentile":
        minimum, maximum = np.percentile(
            z[estimation_mask],
            [float(config.get("z_percentile_min", 1.0)), float(config.get("z_percentile_max", 35.0))],
        )
        minimum += float(config.get("z_min_offset_m", 0.0))
        maximum += float(config.get("z_max_offset_m", 0.0))
        mask = (z >= minimum) & (z <= maximum)
    elif mode == "trajectory_relative":
        if reference_z is None or len(reference_z) != len(z):
            raise ValueError(
                "trajectory_relative rail detection requires one trajectory "
                "reference height per point"
            )
        minimum_offset = float(config["trajectory_z_min_offset_m"])
        maximum_offset = float(config["trajectory_z_max_offset_m"])
        if maximum_offset <= minimum_offset:
            raise ValueError("Invalid trajectory-relative rail search height offsets")
        lower = reference_z + minimum_offset
        upper = reference_z + maximum_offset
        mask = (z >= lower) & (z <= upper)
        minimum = float(np.min(lower[estimation_mask]))
        maximum = float(np.max(upper[estimation_mask]))
    else:
        raise ValueError(
            "rail detection z_mode must be 'absolute', 'percentile' or "
            "'trajectory_relative'"
        )
    if maximum <= minimum:
        raise ValueError("Invalid rail search height range")
    return mask, (float(minimum), float(maximum))


def _core_longitudinal_context(
    project: ProjectConfig,
    segment_id: str,
    frame: Any,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[tuple[float, float], tuple[float, float]],
]:
    manifest_path = project.workspace_path("segment_manifest")
    manifest = load_json(manifest_path)
    segment = next(
        (item for item in manifest.get("segments", []) if item.get("id") == segment_id),
        None,
    )
    if segment is None:
        raise ValueError(f"Segment is not present in the manifest: {segment_id}")
    chainage_start = float(segment["chainage_start_m"])
    chainage_end = float(segment["chainage_end_m"])
    camera_path = project.input_path("camera_csv")
    if camera_path is None or not camera_path.is_file():
        raise FileNotFoundError(camera_path)
    trajectory = camera_trajectory(load_camera_rows(camera_path))
    distances = np.asarray([item["distance_m"] for item in trajectory], dtype=np.float64)
    x = np.asarray([item["x"] for item in trajectory], dtype=np.float64)
    y = np.asarray([item["y"] for item in trajectory], dtype=np.float64)
    z = np.asarray([item["z"] for item in trajectory], dtype=np.float64)
    endpoint_x = np.interp([chainage_start, chainage_end], distances, x)
    endpoint_y = np.interp([chainage_start, chainage_end], distances, y)
    endpoint_z = np.interp([chainage_start, chainage_end], distances, z)
    endpoint_longitudinal, _ = frame.project(endpoint_x, endpoint_y)
    core_start = float(min(endpoint_longitudinal))
    core_end = float(max(endpoint_longitudinal))
    if core_end - core_start <= 1e-6:
        raise ValueError(f"Segment core has no longitudinal extent: {segment_id}")
    longitudinal_z = sorted(
        zip(endpoint_longitudinal.tolist(), endpoint_z.tolist(), strict=True)
    )
    return (
        (core_start, core_end),
        (chainage_start, chainage_end),
        (longitudinal_z[0], longitudinal_z[1]),
    )


def _core_longitudinal_mask(
    longitudinal: np.ndarray, core_range_m: tuple[float, float]
) -> np.ndarray:
    return (longitudinal >= core_range_m[0]) & (longitudinal <= core_range_m[1])


def _save_diagnostic(
    occupancy: np.ndarray,
    peaks: np.ndarray,
    pairs: list[dict[str, Any]],
    output: Path,
) -> None:
    density = np.log1p(occupancy.astype(np.float32))
    if density.max() > 0:
        density /= density.max()
    image = np.zeros((occupancy.shape[0], occupancy.shape[1], 3), dtype=np.uint8)
    image[..., 0] = (density * 24).astype(np.uint8)
    image[..., 1] = (density * 150).astype(np.uint8)
    image[..., 2] = (35 + density * 210).astype(np.uint8)
    image = np.flipud(image)
    scale = max(1, 1400 // max(1, image.shape[1]))
    image = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
    canvas = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(canvas)
    for peak in peaks:
        x = int(peak * scale)
        draw.line((x, 0, x, canvas.height - 1), fill=(255, 100, 45), width=max(1, scale))
    paired = {int(index) for pair in pairs for index in pair["peak_indexes"]}
    for peak in paired:
        x = int(peak * scale)
        draw.line((x, 0, x, canvas.height - 1), fill=(190, 255, 40), width=max(2, scale))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def detect_rail_candidates(
    project: ProjectConfig,
    segment_id: str,
    overwrite: bool = False,
    source_value: str | Path | None = None,
    settings_value: str | Path | None = None,
    report_value: str | Path | None = None,
) -> dict[str, Any]:
    settings_path = (
        Path(settings_value).resolve()
        if settings_value is not None
        else project.resolve(project.value["algorithms"]["rail_detection"])
    )
    config = load_json(settings_path)
    source = (
        Path(source_value).resolve()
        if source_value is not None
        else project.workspace_path("segments") / f"{segment_id}.laz"
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    report_path = (
        Path(report_value).resolve()
        if report_value is not None
        else project.workspace_path("reports") / f"{segment_id}_rail_candidates.json"
    )
    output = (
        report_path.with_suffix(".laz")
        if report_value is not None
        else project.workspace_path("derived") / f"{segment_id}_rail_candidates.laz"
    )
    diagnostic = (
        report_path.with_suffix(".png")
        if report_value is not None
        else project.workspace_path("reports") / f"{segment_id}_rail_candidates.png"
    )
    for path in (output, report_path, diagnostic):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite: {path}")

    cloud = laspy.read(source)
    x = np.asarray(cloud.x, dtype=np.float64)
    y = np.asarray(cloud.y, dtype=np.float64)
    z = np.asarray(cloud.z, dtype=np.float64)
    if not len(x):
        raise ValueError(f"Point cloud is empty: {source}")

    frame_context_before = float(config.get("frame_context_before_m", 100.0))
    frame_context_after = float(config.get("frame_context_after_m", 100.0))
    frame, segment_cameras = fit_segment_corridor_frame(
        project,
        segment_id,
        context_before_m=frame_context_before,
        context_after_m=frame_context_after,
    )
    frame_method = str(config.get("corridor_frame_method", "dominant_axis_regression"))
    if frame_method == "dominant_axis_regression":
        frame = fit_corridor_frame_dominant_axis_regression(segment_cameras)
    elif frame_method != "pca":
        raise ValueError(
            "corridor_frame_method must be 'dominant_axis_regression' or 'pca'"
        )
    longitudinal, cross = frame.project(x, y)
    (
        core_longitudinal_range,
        core_chainage_range,
        trajectory_longitudinal_z,
    ) = _core_longitudinal_context(project, segment_id, frame)
    core_longitudinal = _core_longitudinal_mask(
        longitudinal, core_longitudinal_range
    )
    trajectory_reference_z = np.interp(
        longitudinal,
        [item[0] for item in trajectory_longitudinal_z],
        [item[1] for item in trajectory_longitudinal_z],
    )
    search, z_range = _height_mask(
        z,
        config,
        estimation_mask=core_longitudinal,
        reference_z=trajectory_reference_z,
    )
    core_search = search & core_longitudinal
    if not np.any(core_search):
        raise ValueError("No points remain in the rail search height range")

    cross_bin = float(config["cross_bin_m"])
    longitudinal_bin = float(config["longitudinal_bin_m"])
    # Pad the histogram so a real rail at the search envelope edge can still be a peak.
    cross_min = (
        math.floor(float(cross[core_search].min()) / cross_bin) * cross_bin
        - 2 * cross_bin
    )
    cross_max = (
        math.ceil(float(cross[core_search].max()) / cross_bin) * cross_bin
        + 2 * cross_bin
    )
    long_min = math.floor(core_longitudinal_range[0] / longitudinal_bin) * longitudinal_bin
    long_max = math.ceil(core_longitudinal_range[1] / longitudinal_bin) * longitudinal_bin
    nx = max(1, math.ceil((cross_max - cross_min) / cross_bin) + 1)
    ny = max(1, math.ceil((long_max - long_min) / longitudinal_bin) + 1)
    xi = np.clip(
        ((cross[core_search] - cross_min) / cross_bin).astype(np.int64), 0, nx - 1
    )
    yi = np.clip(
        ((longitudinal[core_search] - long_min) / longitudinal_bin).astype(np.int64),
        0,
        ny - 1,
    )
    occupancy = np.zeros((ny, nx), dtype=np.uint32)
    np.add.at(occupancy, (yi, xi), 1)
    height_grid = np.full((ny, nx), float(z_range[0]), dtype=np.float32)
    np.maximum.at(height_grid, (yi, xi), z[core_search].astype(np.float32))
    baseline_bins = int(config.get("height_baseline_filter_bins", 11))
    if baseline_bins < 3:
        raise ValueError("height_baseline_filter_bins must be at least 3")
    if baseline_bins % 2 == 0:
        baseline_bins += 1
    height_baseline = median_filter(
        height_grid,
        size=(1, baseline_bins),
        mode="nearest",
    )
    height_prominence = height_grid - height_baseline
    prominent = height_prominence >= float(
        config.get("minimum_height_prominence_m", 0.07)
    )
    coverage = prominent.mean(axis=0)
    smooth = gaussian_filter1d(
        median_filter(coverage, size=int(config.get("median_filter_bins", 3))),
        sigma=float(config.get("gaussian_sigma_bins", 1.0)),
    )
    minimum_peak_distance = max(
        1, round(float(config["minimum_peak_spacing_m"]) / cross_bin)
    )
    peaks, _ = find_peaks(
        smooth,
        height=float(config["minimum_line_coverage"]),
        prominence=float(config["minimum_peak_prominence"]),
        distance=minimum_peak_distance,
    )
    refined_offsets = np.zeros(len(peaks), dtype=np.float64)
    for index, peak in enumerate(peaks):
        if 0 < peak < len(smooth) - 1:
            left_value = float(smooth[peak - 1])
            center_value = float(smooth[peak])
            right_value = float(smooth[peak + 1])
            denominator = left_value - 2.0 * center_value + right_value
            if abs(denominator) > 1e-12:
                refined_offsets[index] = float(
                    np.clip(0.5 * (left_value - right_value) / denominator, -0.5, 0.5)
                )
    positions = cross_min + (peaks + 0.5 + refined_offsets) * cross_bin
    scores = smooth[peaks]

    half_width = float(config["candidate_half_width_m"])
    peak_radius = max(1, math.ceil(half_width / cross_bin))
    height_grid_peak_top_z: dict[int, float | None] = {}
    peak_top_z: dict[int, float | None] = {}
    peak_line_fits: dict[int, dict[str, Any]] = {}
    rail_top_fit_method = str(config.get("rail_top_fit_method", "height_grid_max"))
    if rail_top_fit_method not in {
        "height_grid_max",
        "anchored_quantile",
        "local_quantile",
    }:
        raise ValueError(
            "rail_top_fit_method must be 'height_grid_max', 'anchored_quantile' "
            "or 'local_quantile'"
        )
    rail_top_quantile = float(config.get("rail_top_quantile", 0.50))
    rail_top_minimum_bin_points = int(
        config.get("rail_top_minimum_bin_points", 3)
    )
    vertical_below = float(config.get("candidate_vertical_below_rail_m", 0.12))
    vertical_above = float(config.get("candidate_vertical_above_rail_m", 0.05))
    minimum_height_prominence = float(
        config.get("minimum_height_prominence_m", 0.07)
    )
    for peak in peaks:
        left = max(0, int(peak) - peak_radius)
        right = min(nx, int(peak) + peak_radius + 1)
        window_mask = prominent[:, left:right]
        window_values = height_grid[:, left:right][window_mask]
        initial_peak_top_z = (
            float(np.median(window_values)) if window_values.size else None
        )
        height_grid_peak_top_z[int(peak)] = initial_peak_top_z
        peak_top_z[int(peak)] = initial_peak_top_z
        sample_longitudinal: list[float] = []
        sample_cross: list[float] = []
        sample_z: list[float] = []
        for row in range(ny):
            row_prominence = height_prominence[row, left:right]
            eligible = np.flatnonzero(
                row_prominence >= minimum_height_prominence
            )
            if not len(eligible):
                continue
            relative = int(
                min(
                    eligible,
                    key=lambda value: (
                        abs((left + int(value)) - int(peak)),
                        -float(row_prominence[int(value)]),
                    ),
                )
            )
            column = left + relative
            sample_longitudinal.append(long_min + (row + 0.5) * longitudinal_bin)
            sample_cross.append(cross_min + (column + 0.5) * cross_bin)
            sample_z.append(float(height_grid[row, column]))
        longitudinal_values = np.asarray(sample_longitudinal, dtype=np.float64)
        peak_line_fits[int(peak)] = {
            "cross": _robust_linear_fit(
                longitudinal_values,
                np.asarray(sample_cross, dtype=np.float64),
            ),
            "z": _robust_linear_fit(
                longitudinal_values,
                np.asarray(sample_z, dtype=np.float64),
            ),
        }
        if rail_top_fit_method == "anchored_quantile" and initial_peak_top_z is not None:
            position = cross_min + (float(peak) + 0.5) * cross_bin
            quantile_longitudinal, quantile_z = _anchored_rail_top_samples(
                longitudinal,
                cross,
                z,
                core_search,
                position,
                initial_peak_top_z,
                long_min,
                longitudinal_bin,
                half_width,
                vertical_below,
                vertical_above,
                rail_top_quantile,
                rail_top_minimum_bin_points,
            )
            if len(quantile_z) >= 2:
                peak_top_z[int(peak)] = float(np.median(quantile_z))
                peak_line_fits[int(peak)]["z"] = _robust_linear_fit(
                    quantile_longitudinal, quantile_z
                )
        elif rail_top_fit_method == "local_quantile":
            position = cross_min + (float(peak) + 0.5) * cross_bin
            quantile_longitudinal, quantile_z = _local_rail_top_samples(
                longitudinal,
                cross,
                z,
                core_search,
                position,
                long_min,
                longitudinal_bin,
                half_width,
                rail_top_quantile,
                rail_top_minimum_bin_points,
            )
            if len(quantile_z) >= 2:
                peak_top_z[int(peak)] = float(np.median(quantile_z))
                peak_line_fits[int(peak)]["z"] = _robust_linear_fit(
                    quantile_longitudinal, quantile_z
                )

    candidates: list[dict[str, Any]] = []
    nominal_gauge = float(config.get("nominal_gauge_m", 1.435))
    rail_head_width = float(config.get("assumed_rail_head_width_m", 0.073))
    maximum_crosslevel = float(config.get("maximum_rail_top_crosslevel_m", 0.20))
    rail_pair_top_method = str(config.get("rail_pair_top_method", "selected_fit"))
    if rail_pair_top_method not in {"height_grid_max", "selected_fit"}:
        raise ValueError(
            "rail_pair_top_method must be 'height_grid_max' or 'selected_fit'"
        )
    gauge_score_weight = float(config.get("pair_gauge_score_weight", 2.0))
    continuity_settings = _pair_continuity_settings(config, longitudinal_bin)
    joint_support_score_weight = continuity_settings["joint_support_score_weight"]
    asymmetric_support_penalty_weight = continuity_settings[
        "asymmetric_support_penalty_weight"
    ]
    minimum_pair_joint_coverage = continuity_settings["minimum_joint_coverage"]
    maximum_pair_internal_gap = continuity_settings["maximum_internal_gap_m"]
    maximum_pair_asymmetric_support = continuity_settings[
        "maximum_asymmetric_support_ratio"
    ]
    nominal_center_spacing = nominal_gauge + rail_head_width
    constrain_gauge = bool(config.get("constrain_candidate_gauge_to_nominal", True))
    maximum_position_correction = float(
        config.get("maximum_candidate_rail_position_correction_m", 0.06)
    )
    for left in range(len(peaks)):
        for right in range(left + 1, len(peaks)):
            separation = float(positions[right] - positions[left])
            left_peak = int(peaks[left])
            right_peak = int(peaks[right])
            left_z = _pairing_rail_top_z(
                height_grid_peak_top_z[left_peak],
                peak_top_z[left_peak],
                rail_pair_top_method,
            )
            right_z = _pairing_rail_top_z(
                height_grid_peak_top_z[right_peak],
                peak_top_z[right_peak],
                rail_pair_top_method,
            )
            crosslevel = (
                abs(float(right_z) - float(left_z))
                if left_z is not None and right_z is not None
                else math.inf
            )
            if float(config["rail_pair_minimum_m"]) <= separation <= float(
                config["rail_pair_maximum_m"]
            ) and crosslevel <= maximum_crosslevel:
                continuity = _paired_longitudinal_support(
                    prominent,
                    left_peak,
                    right_peak,
                    peak_radius,
                    longitudinal_bin,
                    continuity_settings["support_longitudinal_bin_m"],
                )
                observed_positions = [float(positions[left]), float(positions[right])]
                observed_gauge = separation - rail_head_width
                center = float(np.mean(observed_positions))
                position_correction = abs(nominal_center_spacing - separation) / 2.0
                use_nominal = constrain_gauge and position_correction <= maximum_position_correction
                corrected_positions = (
                    [
                        center - nominal_center_spacing / 2.0,
                        center + nominal_center_spacing / 2.0,
                    ]
                    if use_nominal
                    else observed_positions
                )
                rail_center_spacing = (
                    nominal_center_spacing if use_nominal else separation
                )
                gauge = nominal_gauge if use_nominal else observed_gauge
                base_score = float(
                    scores[left]
                    + scores[right]
                    - gauge_score_weight * abs(observed_gauge - nominal_gauge)
                )
                pair_score = float(
                    base_score
                    + joint_support_score_weight
                    * float(continuity["joint_support_ratio"])
                    - asymmetric_support_penalty_weight
                    * float(continuity["asymmetric_support_ratio"])
                )
                continuity_status = (
                    "pass"
                    if float(continuity["joint_support_ratio"])
                    >= minimum_pair_joint_coverage
                    and float(continuity["asymmetric_support_ratio"])
                    <= maximum_pair_asymmetric_support
                    and float(continuity["maximum_internal_joint_gap_m"])
                    <= maximum_pair_internal_gap
                    else "review_required"
                )
                candidates.append(
                    {
                        "peak_indexes": [int(peaks[left]), int(peaks[right])],
                        "cross_positions_m": corrected_positions,
                        "observed_cross_positions_m": observed_positions,
                        "separation_m": rail_center_spacing,
                        "rail_center_spacing_m": rail_center_spacing,
                        "observed_rail_center_spacing_m": separation,
                        "gauge_m": gauge,
                        "observed_gauge_m": observed_gauge,
                        "rail_position_correction_m": (
                            position_correction if use_nominal else 0.0
                        ),
                        "gauge_constrained_to_nominal": use_nominal,
                        "rail_top_crosslevel_m": crosslevel,
                        "base_score": base_score,
                        "score": pair_score,
                        "pair_continuity_status": continuity_status,
                        **continuity,
                    }
                )
    continuity_selection_policy = str(
        config.get("pair_continuity_selection_policy", "diagnostic_only")
    )
    ordered_candidates = _ordered_pair_candidates(
        candidates, continuity_selection_policy
    )
    pairs: list[dict[str, Any]] = []
    used: set[int] = set()
    selected_centers: list[float] = []
    minimum_track_spacing = float(config.get("minimum_track_center_spacing_m", 3.0))
    maximum_track_count = int(config.get("maximum_track_count", 8))
    for pair in ordered_candidates:
        indexes = set(pair["peak_indexes"])
        if indexes & used:
            continue
        center = float(np.mean(pair["cross_positions_m"]))
        if any(abs(center - existing) < minimum_track_spacing for existing in selected_centers):
            continue
        pairs.append({**pair, "track_id": f"TRACK-{len(pairs) + 1:04d}"})
        used.update(indexes)
        selected_centers.append(center)
        if len(pairs) >= maximum_track_count:
            break

    selected_peak_indexes = sorted(
        {int(index) for pair in pairs for index in pair["peak_indexes"]}
    )
    position_by_peak = {
        int(peak): float(position) for peak, position in zip(peaks, positions)
    }
    candidate_mask = np.zeros(len(x), dtype=bool)
    line_records: list[dict[str, Any]] = []
    for index, peak in enumerate(selected_peak_indexes, start=1):
        position = position_by_peak[peak]
        line_mask = core_search & (np.abs(cross - position) <= half_width)
        rail_top_z = peak_top_z[peak]
        if rail_top_z is not None:
            line_mask &= z >= rail_top_z - vertical_below
            line_mask &= z <= rail_top_z + vertical_above
        candidate_mask |= line_mask
        line_records.append(
            {
                "id": f"RAIL-{index:04d}",
                "cross_position_m": float(position),
                "median_z_m": rail_top_z,
                "point_count": int(np.count_nonzero(line_mask)),
                "cross_fit_slope_m_per_m": peak_line_fits[peak]["cross"]["slope"],
                "cross_fit_intercept_m": peak_line_fits[peak]["cross"]["intercept"],
                "cross_fit_sample_count": peak_line_fits[peak]["cross"]["sample_count"],
                "cross_fit_residual_p90_m": peak_line_fits[peak]["cross"][
                    "residual_p90_m"
                ],
                "z_fit_slope_m_per_m": peak_line_fits[peak]["z"]["slope"],
                "z_fit_intercept_m": peak_line_fits[peak]["z"]["intercept"],
                "z_fit_sample_count": peak_line_fits[peak]["z"]["sample_count"],
                "z_fit_residual_p90_m": peak_line_fits[peak]["z"][
                    "residual_p90_m"
                ],
            }
        )

    result_cloud = laspy.LasData(copy.deepcopy(cloud.header))
    result_cloud.points = cloud.points[candidate_mask]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
    result_cloud.write(temporary)
    os.replace(temporary, output)
    _save_diagnostic(occupancy, peaks, pairs, diagnostic)

    report = {
        "schema_version": "railway.rail-candidates.v1",
        "project_id": project.project_id,
        "segment_id": segment_id,
        "source": str(source),
        "settings_source": str(settings_path),
        "frame": frame.to_json(),
        "frame_method": frame_method,
        "frame_context_before_m": frame_context_before,
        "frame_context_after_m": frame_context_after,
        "frame_camera_count": len(segment_cameras),
        "frame_camera_index_range": [
            segment_cameras[0]["index"], segment_cameras[-1]["index"]
        ],
        "longitudinal_range_m": [float(long_min), float(long_max)],
        "core_longitudinal_range_m": list(core_longitudinal_range),
        "core_chainage_range_m": list(core_chainage_range),
        "search_z_range_m": list(z_range),
        "search_z_mode": str(config.get("z_mode", "percentile")),
        "trajectory_reference_z_range_m": [
            float(trajectory_longitudinal_z[0][1]),
            float(trajectory_longitudinal_z[1][1]),
        ],
        "source_search_point_count": int(np.count_nonzero(search)),
        "search_point_count": int(np.count_nonzero(core_search)),
        "peak_count": len(peaks),
        "peaks": [
            {
                "grid_index": int(i),
                "cross_position_m": float(p),
                "coverage": float(s),
                "median_z_m": peak_top_z[int(i)],
                "cross_fit_slope_m_per_m": peak_line_fits[int(i)]["cross"]["slope"],
                "cross_fit_intercept_m": peak_line_fits[int(i)]["cross"]["intercept"],
                "cross_fit_sample_count": peak_line_fits[int(i)]["cross"][
                    "sample_count"
                ],
                "cross_fit_residual_p90_m": peak_line_fits[int(i)]["cross"][
                    "residual_p90_m"
                ],
                "z_fit_slope_m_per_m": peak_line_fits[int(i)]["z"]["slope"],
                "z_fit_intercept_m": peak_line_fits[int(i)]["z"]["intercept"],
                "z_fit_sample_count": peak_line_fits[int(i)]["z"]["sample_count"],
                "z_fit_residual_p90_m": peak_line_fits[int(i)]["z"]["residual_p90_m"],
            }
            for i, p, s in zip(peaks, positions, scores)
        ],
        "detection_method": "longitudinal_height_prominence_v2",
        "rail_top_fit_method": rail_top_fit_method,
        "rail_pair_top_method": rail_pair_top_method,
        "rail_top_quantile": rail_top_quantile,
        "rail_top_minimum_bin_points": rail_top_minimum_bin_points,
        "height_baseline_filter_bins": baseline_bins,
        "minimum_height_prominence_m": float(
            config.get("minimum_height_prominence_m", 0.07)
        ),
        "nominal_gauge_m": nominal_gauge,
        "assumed_rail_head_width_m": rail_head_width,
        "constrain_candidate_gauge_to_nominal": constrain_gauge,
        "maximum_candidate_rail_position_correction_m": maximum_position_correction,
        "pair_joint_support_score_weight": joint_support_score_weight,
        "pair_asymmetric_support_penalty_weight": asymmetric_support_penalty_weight,
        "minimum_pair_joint_coverage": minimum_pair_joint_coverage,
        "maximum_pair_internal_gap_m": maximum_pair_internal_gap,
        "maximum_pair_asymmetric_support_ratio": maximum_pair_asymmetric_support,
        "pair_support_longitudinal_bin_m": continuity_settings[
            "support_longitudinal_bin_m"
        ],
        "pair_continuity_selection_policy": continuity_selection_policy,
        "rail_pair_candidate_count_before_continuity_policy": len(candidates),
        "rail_pair_candidate_count_after_continuity_policy": len(
            ordered_candidates
        ),
        "rail_pair_count": len(pairs),
        "rail_pairs": pairs,
        "rail_lines": line_records,
        "candidate_point_count": int(np.count_nonzero(candidate_mask)),
        "candidate_output": str(output),
        "diagnostic_image": str(diagnostic),
        "status": "geometry_candidates_only_manual_review_required",
    }
    write_json(report_path, report)
    return report
