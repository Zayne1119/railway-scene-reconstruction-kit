from __future__ import annotations

import copy
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .geometry import CorridorFrame
from .io import load_json, sha256_file, write_json


def prepare_gislab_railtrack_input(
    source_path: str | Path,
    reference_report_path: str | Path,
    segment_id: str,
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    coordinate_space: str = "world_xy",
) -> tuple[Path, Path]:
    """Apply only the frame/range envelope shared by all compared methods."""

    if coordinate_space not in {"world_xy", "corridor_local_xy"}:
        raise ValueError(f"Unsupported GISLab input coordinate space: {coordinate_space}")
    source = Path(source_path).resolve()
    reference_source = Path(reference_report_path).resolve()
    reference = load_json(reference_source)
    if reference.get("schema_version") != "railway.rail-candidates.v1":
        raise ValueError("Reference must be a rail-candidates report")
    if reference.get("segment_id") != segment_id:
        raise ValueError("Reference segment id differs from requested segment")
    frame = CorridorFrame.from_json(reference["frame"])
    longitudinal_range = tuple(float(value) for value in reference["longitudinal_range_m"])
    z_range = tuple(float(value) for value in reference["search_z_range_m"])
    output = Path(output_path).resolve()
    manifest_output = Path(manifest_path).resolve()
    if output.exists() or manifest_output.exists():
        raise FileExistsError(f"Refusing to overwrite GISLab prepared input: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
    selected_count = 0
    source_count = 0
    try:
        with laspy.open(source) as reader:
            source_count = int(reader.header.point_count)
            header = copy.deepcopy(reader.header)
            if coordinate_space == "corridor_local_xy":
                header.offsets = np.asarray(
                    [0.0, 0.0, float(reader.header.offsets[2])], dtype=np.float64
                )
            with laspy.open(
                temporary,
                mode="w",
                header=header,
                do_compress=output.suffix.lower() == ".laz",
            ) as writer:
                for points in reader.chunk_iterator(2_000_000):
                    x = np.asarray(points.x, dtype=np.float64)
                    y = np.asarray(points.y, dtype=np.float64)
                    z = np.asarray(points.z, dtype=np.float64)
                    longitudinal, _ = frame.project(x, y)
                    keep = (
                        (longitudinal >= longitudinal_range[0])
                        & (longitudinal <= longitudinal_range[1])
                        & (z >= z_range[0])
                        & (z <= z_range[1])
                    )
                    if np.any(keep):
                        if coordinate_space == "world_xy":
                            selected_points = points[keep]
                        else:
                            selected_points = laspy.ScaleAwarePointRecord.zeros(
                                int(np.count_nonzero(keep)), header=header
                            )
                            for dimension in header.point_format.dimension_names:
                                if dimension not in {"X", "Y"}:
                                    selected_points[dimension] = np.asarray(points[dimension])[keep]
                            selected_points.x = longitudinal[keep]
                            _, cross = frame.project(x, y)
                            selected_points.y = cross[keep]
                        writer.write_points(selected_points)
                        selected_count += int(np.count_nonzero(keep))
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    manifest = {
        "schema_version": "railway.external-baseline-preprocess.v1",
        "method": "shared_frozen_frame_longitudinal_and_z_envelope",
        "segment_id": segment_id,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "source_point_count": source_count,
        "reference_report": str(reference_source),
        "reference_report_sha256": sha256_file(reference_source),
        "frame": frame.to_json(),
        "longitudinal_range_m": list(longitudinal_range),
        "z_range_m": list(z_range),
        "coordinate_space": coordinate_space,
        "coordinate_transform": (
            "x=frame.longitudinal,y=frame.cross,z=world_z"
            if coordinate_space == "corridor_local_xy"
            else "identity"
        ),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "selected_point_count": selected_count,
        "rejected_point_count": source_count - selected_count,
        "manual_seed_used": False,
        "limitations": [
            "This shared envelope is already used by the in-house and Open3D evaluations.",
            "No rail position, gauge pair, semantic label or model geometry is supplied.",
        ],
    }
    write_json(manifest_output, manifest)
    return output, manifest_output


def record_gislab_no_detection(
    run_manifest_path: str | Path,
    reference_report_path: str | Path,
    segment_id: str,
    output_path: str | Path,
    *,
    source_coordinate_space: str = "corridor_local_xy",
) -> Path:
    """Convert the upstream empty-result crash into an explicit zero detection."""

    run_source = Path(run_manifest_path).resolve()
    run = load_json(run_source)
    if run.get("schema_version") != "railway.external-baseline-run.v1":
        raise ValueError("Unsupported external baseline run manifest")
    if run.get("method") != "GISLab-ELTE/railroad:RailTrack":
        raise ValueError("Run manifest is not a GISLab RailTrack run")
    exit_code = int(run.get("exit_code", 0))
    if exit_code not in {134, 139}:
        raise ValueError("Only a proven upstream empty-result abort can be normalized")
    log_path = run_source.with_name("combined.log")
    log_bytes = log_path.read_bytes()
    if log_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
        log_text = log_bytes.decode("utf-16", errors="replace")
    else:
        log_text = log_bytes.decode("utf-8-sig", errors="replace")
    zero_pair_proof = "Pairs: 0" in log_text and "-> 0" in log_text
    zero_candidate_proof = "railH95 size: 0" in log_text and "cvHough size: 0" in log_text
    if not (zero_pair_proof or zero_candidate_proof):
        raise ValueError("Run log does not prove a zero-detection upstream result")
    detection_stage = "pair_filter" if zero_pair_proof else "rail_candidate_extraction"
    reference_source = Path(reference_report_path).resolve()
    reference = load_json(reference_source)
    if reference.get("schema_version") != "railway.rail-candidates.v1":
        raise ValueError("Reference must be a rail-candidates report")
    if reference.get("segment_id") != segment_id:
        raise ValueError("Reference segment id differs from requested segment")
    report = {
        "schema_version": "railway.rail-candidates.v1",
        "project_id": reference.get("project_id"),
        "segment_id": segment_id,
        "source": str(run["input"]),
        "source_sha256": str(run["input_sha256"]),
        "reference_report": str(reference_source),
        "reference_report_sha256": sha256_file(reference_source),
        "frame": reference["frame"],
        "longitudinal_range_m": reference["longitudinal_range_m"],
        "search_z_range_m": reference["search_z_range_m"],
        "detection_method": "gislab_railtrack_3557ecf_zero_detection_v1",
        "execution_mode": "unmodified_upstream_detection_explicit_empty_result",
        "shared_preprocessing": "frozen_frame_longitudinal_z_envelope_and_rigid_axis_alignment",
        "source_coordinate_space": source_coordinate_space,
        "generated_at": datetime.now(UTC).isoformat(),
        "upstream": {
            "repository": "https://github.com/GISLab-ELTE/railroad",
            "commit": str(run["upstream_commit"]),
            "lastools_commit": str(run["lastools_commit"]),
            "algorithm": "RailTrack",
        },
        "upstream_run_manifest": str(run_source),
        "upstream_run_manifest_sha256": sha256_file(run_source),
        "upstream_log": str(log_path),
        "upstream_log_sha256": sha256_file(log_path),
        "upstream_exit_code": exit_code,
        "zero_detection_proof": detection_stage,
        "eligible_cluster_count": 0,
        "eligible_clusters": [],
        "rail_pair_count": 0,
        "rail_pairs": [],
        "rail_lines": [],
        "status": "upstream_zero_detection_then_empty_output_crash",
        "limitations": [
            (
                "The unmodified detector found zero gauge-compatible rail pairs."
                if zero_pair_proof
                else "The unmodified detector produced zero rail-height and Hough-line candidates."
            ),
            f"Upstream then exits {exit_code} while handling the empty result.",
            "This report adds no geometry; it preserves the zero detection for evaluation.",
            "No rail position, gauge pair, semantic label or manual seed was supplied.",
        ],
    }
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite GISLab empty result report: {output}")
    write_json(output, report)
    return output


def record_gislab_timeout(
    run_manifest_path: str | Path,
    reference_report_path: str | Path,
    segment_id: str,
    output_path: str | Path,
    *,
    source_coordinate_space: str = "corridor_local_xy",
) -> Path:
    """Record a resource-capped run as an explicit non-result, without geometry."""

    run_source = Path(run_manifest_path).resolve()
    run = load_json(run_source)
    if run.get("schema_version") != "railway.external-baseline-run.v1":
        raise ValueError("Unsupported external baseline run manifest")
    if run.get("method") != "GISLab-ELTE/railroad:RailTrack":
        raise ValueError("Run manifest is not a GISLab RailTrack run")
    exit_code = int(run.get("exit_code", 0))
    elapsed_seconds = float(run.get("elapsed_seconds", 0.0))
    timeout_seconds = int(run.get("timeout_seconds", 600))
    proven_timeout = exit_code == 124 or (exit_code == 137 and elapsed_seconds >= 600.0)
    if not proven_timeout:
        raise ValueError("Run manifest does not prove a resource-capped timeout")
    reference_source = Path(reference_report_path).resolve()
    reference = load_json(reference_source)
    if reference.get("schema_version") != "railway.rail-candidates.v1":
        raise ValueError("Reference must be a rail-candidates report")
    if reference.get("segment_id") != segment_id:
        raise ValueError("Reference segment id differs from requested segment")
    log_path = run_source.with_name("combined.log")
    report = {
        "schema_version": "railway.rail-candidates.v1",
        "project_id": reference.get("project_id"),
        "segment_id": segment_id,
        "source": str(run["input"]),
        "source_sha256": str(run["input_sha256"]),
        "reference_report": str(reference_source),
        "reference_report_sha256": sha256_file(reference_source),
        "frame": reference["frame"],
        "longitudinal_range_m": reference["longitudinal_range_m"],
        "search_z_range_m": reference["search_z_range_m"],
        "detection_method": "gislab_railtrack_3557ecf_timeout_v1",
        "execution_mode": "unmodified_upstream_detection_resource_capped_non_result",
        "shared_preprocessing": "frozen_frame_longitudinal_z_envelope_and_rigid_axis_alignment",
        "source_coordinate_space": source_coordinate_space,
        "generated_at": datetime.now(UTC).isoformat(),
        "upstream": {
            "repository": "https://github.com/GISLab-ELTE/railroad",
            "commit": str(run["upstream_commit"]),
            "lastools_commit": str(run["lastools_commit"]),
            "algorithm": "RailTrack",
        },
        "upstream_run_manifest": str(run_source),
        "upstream_run_manifest_sha256": sha256_file(run_source),
        "upstream_log": str(log_path),
        "upstream_log_sha256": sha256_file(log_path),
        "upstream_exit_code": exit_code,
        "elapsed_seconds": elapsed_seconds,
        "timeout_seconds": timeout_seconds,
        "eligible_cluster_count": 0,
        "eligible_clusters": [],
        "rail_pair_count": 0,
        "rail_pairs": [],
        "rail_lines": [],
        "status": "upstream_timeout_no_output",
        "limitations": [
            "The unmodified detector did not finish inside the declared per-block resource cap.",
            "This report adds no geometry and keeps timeout distinct from zero detection.",
            "No rail position, gauge pair, semantic label or manual seed was supplied.",
        ],
    }
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite GISLab timeout report: {output}")
    write_json(output, report)
    return output


def _robust_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, int]:
    if len(x) < 2 or float(np.ptp(x)) <= 1e-9:
        value = float(np.median(y)) if len(y) else math.nan
        return 0.0, value, math.nan, len(y)
    if len(x) > 250_000:
        indexes = np.linspace(0, len(x) - 1, 250_000, dtype=np.int64)
        x = x[indexes]
        y = y[indexes]
    keep = np.ones(len(x), dtype=bool)
    slope = 0.0
    intercept = float(np.median(y))
    for _ in range(3):
        if int(np.count_nonzero(keep)) < 2:
            break
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        residual = np.abs(y - (slope * x + intercept))
        cutoff = max(0.005, float(np.percentile(residual[keep], 90)))
        keep = residual <= cutoff
    residual = np.abs(y[keep] - (slope * x[keep] + intercept))
    p90 = float(np.percentile(residual, 90)) if len(residual) else math.nan
    return float(slope), float(intercept), p90, int(np.count_nonzero(keep))


def _read_projected_points(
    source: Path,
    frame: CorridorFrame,
    longitudinal_range: tuple[float, float],
    z_range: tuple[float, float],
    maximum_points: int,
    source_coordinate_space: str,
) -> tuple[np.ndarray, int]:
    selected: list[np.ndarray] = []
    with laspy.open(source) as reader:
        total = int(reader.header.point_count)
        stride = max(1, math.ceil(total / maximum_points))
        offset = 0
        for chunk in reader.chunk_iterator(2_000_000):
            indexes = np.arange(len(chunk), dtype=np.int64)
            keep_stride = ((indexes + offset) % stride) == 0
            offset += len(chunk)
            if not np.any(keep_stride):
                continue
            x = np.asarray(chunk.x, dtype=np.float64)[keep_stride]
            y = np.asarray(chunk.y, dtype=np.float64)[keep_stride]
            z = np.asarray(chunk.z, dtype=np.float64)[keep_stride]
            if source_coordinate_space == "world_xy":
                longitudinal, cross = frame.project(x, y)
            else:
                longitudinal, cross = x, y
            keep = (
                np.isfinite(longitudinal)
                & np.isfinite(cross)
                & np.isfinite(z)
                & (longitudinal >= longitudinal_range[0])
                & (longitudinal <= longitudinal_range[1])
                & (z >= z_range[0])
                & (z <= z_range[1])
            )
            if np.any(keep):
                selected.append(np.column_stack((longitudinal[keep], cross[keep], z[keep])))
    points = np.concatenate(selected) if selected else np.empty((0, 3), dtype=np.float64)
    return points, total


def _global_cross_trend(points: np.ndarray, bin_size_m: float = 1.0) -> tuple[float, float]:
    longitudinal = points[:, 0]
    cross = points[:, 1]
    bins = np.floor((longitudinal - float(np.min(longitudinal))) / bin_size_m).astype(int)
    centers: list[float] = []
    medians: list[float] = []
    for value in np.unique(bins):
        mask = bins == value
        if int(np.count_nonzero(mask)) >= 3:
            centers.append(float(np.median(longitudinal[mask])))
            medians.append(float(np.median(cross[mask])))
    if len(centers) < 2:
        return 0.0, float(np.median(cross))
    slope, intercept, _, _ = _robust_line(
        np.asarray(centers, dtype=np.float64), np.asarray(medians, dtype=np.float64)
    )
    return slope, intercept


def _closest_peak(values: np.ndarray, peaks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(peaks)
    sorted_peaks = peaks[order]
    right = np.searchsorted(sorted_peaks, values)
    right = np.clip(right, 0, len(sorted_peaks) - 1)
    left = np.clip(right - 1, 0, len(sorted_peaks) - 1)
    choose_left = np.abs(values - sorted_peaks[left]) <= np.abs(values - sorted_peaks[right])
    chosen_sorted = np.where(choose_left, left, right)
    return order[chosen_sorted], np.abs(values - sorted_peaks[chosen_sorted])


def normalize_gislab_railtrack_points(
    source_path: str | Path,
    reference_report_path: str | Path,
    segment_id: str,
    output_path: str | Path,
    *,
    cross_bin_m: float = 0.025,
    smoothing_sigma_m: float = 0.05,
    minimum_peak_spacing_m: float = 0.40,
    assignment_radius_m: float = 0.14,
    minimum_cluster_length_m: float = 15.0,
    minimum_cluster_points: int = 25,
    maximum_input_points: int = 2_000_000,
    source_coordinate_space: str = "world_xy",
) -> Path:
    """Normalize unmodified GISLab RailTrack point output for common evaluation.

    The upstream detector remains untouched. This step only fits line summaries
    to its output points using the frozen frame and height envelope shared by all
    compared methods.
    """

    if min(cross_bin_m, smoothing_sigma_m, minimum_peak_spacing_m, assignment_radius_m) <= 0:
        raise ValueError("GISLab normalization distances must be positive")
    if minimum_cluster_points < 2 or maximum_input_points < 100:
        raise ValueError("GISLab normalization point limits are invalid")
    if source_coordinate_space not in {"world_xy", "corridor_local_xy"}:
        raise ValueError(
            f"Unsupported GISLab output coordinate space: {source_coordinate_space}"
        )
    source = Path(source_path).resolve()
    reference_source = Path(reference_report_path).resolve()
    reference = load_json(reference_source)
    if reference.get("schema_version") != "railway.rail-candidates.v1":
        raise ValueError("Reference must be a rail-candidates report")
    if reference.get("segment_id") != segment_id:
        raise ValueError("Reference segment id differs from requested segment")
    frame = CorridorFrame.from_json(reference["frame"])
    longitudinal_range = tuple(float(value) for value in reference["longitudinal_range_m"])
    z_range = tuple(float(value) for value in reference["search_z_range_m"])
    points, source_point_count = _read_projected_points(
        source,
        frame,
        longitudinal_range,
        z_range,
        maximum_input_points,
        source_coordinate_space,
    )

    candidates: list[dict[str, Any]] = []
    peak_positions = np.empty(0, dtype=np.float64)
    trend_slope = 0.0
    trend_intercept = 0.0
    if len(points) >= minimum_cluster_points:
        trend_slope, trend_intercept = _global_cross_trend(points)
        midpoint = float(np.mean(longitudinal_range))
        detrended = points[:, 1] - trend_slope * (points[:, 0] - midpoint)
        low = math.floor(float(np.percentile(detrended, 0.1)) / cross_bin_m) * cross_bin_m
        high = math.ceil(float(np.percentile(detrended, 99.9)) / cross_bin_m) * cross_bin_m
        if high <= low:
            high = low + cross_bin_m
        edges = np.arange(low, high + cross_bin_m * 1.5, cross_bin_m)
        histogram, _ = np.histogram(detrended, bins=edges)
        smooth = gaussian_filter1d(
            histogram.astype(np.float64), smoothing_sigma_m / cross_bin_m, mode="nearest"
        )
        maximum = float(np.max(smooth)) if len(smooth) else 0.0
        padded_smooth = np.pad(smooth, (1, 1), mode="constant")
        indexes, _ = find_peaks(
            padded_smooth,
            height=max(2.0, maximum * 0.01),
            prominence=max(1.0, maximum * 0.02),
            distance=max(1, round(minimum_peak_spacing_m / cross_bin_m)),
        )
        indexes = indexes - 1
        indexes = indexes[(indexes >= 0) & (indexes < len(smooth))]
        peak_positions = low + (indexes.astype(np.float64) + 0.5) * cross_bin_m
        if len(peak_positions):
            assignments, distances = _closest_peak(detrended, peak_positions)
            for peak_index, initial_position in enumerate(peak_positions):
                mask = (assignments == peak_index) & (distances <= assignment_radius_m)
                if int(np.count_nonzero(mask)) < minimum_cluster_points:
                    continue
                cluster = points[mask]
                longitudinal_extent = float(
                    np.percentile(cluster[:, 0], 95) - np.percentile(cluster[:, 0], 5)
                )
                if longitudinal_extent < minimum_cluster_length_m:
                    continue
                cross_slope, cross_intercept, cross_p90, cross_count = _robust_line(
                    cluster[:, 0], cluster[:, 1]
                )
                z_slope, z_intercept, z_p90, z_count = _robust_line(cluster[:, 0], cluster[:, 2])
                occupied = len(np.unique(np.floor(cluster[:, 0] / 0.5).astype(int)))
                possible = max(1, math.ceil(longitudinal_extent / 0.5))
                candidates.append(
                    {
                        "initial_detrended_peak_m": float(initial_position),
                        "cross_position_m": float(cross_slope * midpoint + cross_intercept),
                        "median_z_m": float(np.median(cluster[:, 2])),
                        "point_count": len(cluster),
                        "longitudinal_extent_m": longitudinal_extent,
                        "longitudinal_coverage": min(1.0, occupied / possible),
                        "cross_fit_slope_m_per_m": cross_slope,
                        "cross_fit_intercept_m": cross_intercept,
                        "cross_fit_sample_count": cross_count,
                        "cross_fit_residual_p90_m": cross_p90,
                        "z_fit_slope_m_per_m": z_slope,
                        "z_fit_intercept_m": z_intercept,
                        "z_fit_sample_count": z_count,
                        "z_fit_residual_p90_m": z_p90,
                    }
                )

    candidates.sort(key=lambda item: item["cross_position_m"])
    rail_head_width_m = 0.073
    pair_candidates: list[dict[str, Any]] = []
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            separation = float(
                candidates[right]["cross_position_m"] - candidates[left]["cross_position_m"]
            )
            if 1.35 <= separation <= 1.65:
                gauge = separation - rail_head_width_m
                score = (
                    candidates[left]["longitudinal_coverage"]
                    + candidates[right]["longitudinal_coverage"]
                    - 4.0 * abs(gauge - 1.435)
                )
                pair_candidates.append(
                    {
                        "candidate_indexes": [left, right],
                        "observed_cross_positions_m": [
                            candidates[left]["cross_position_m"],
                            candidates[right]["cross_position_m"],
                        ],
                        "observed_rail_center_spacing_m": separation,
                        "observed_gauge_m": gauge,
                        "score": float(score),
                    }
                )
    pair_candidates.sort(key=lambda item: item["score"], reverse=True)
    used: set[int] = set()
    pairs: list[dict[str, Any]] = []
    for item in pair_candidates:
        indexes = set(item["candidate_indexes"])
        if indexes & used:
            continue
        pairs.append({**item, "track_id": f"TRACK-{len(pairs) + 1:04d}"})
        used.update(indexes)
        if len(pairs) >= 8:
            break

    selected_indexes = sorted(used, key=lambda index: candidates[index]["cross_position_m"])
    index_to_line = {value: index for index, value in enumerate(selected_indexes)}
    lines: list[dict[str, Any]] = []
    for number, candidate_index in enumerate(selected_indexes, start=1):
        value = dict(candidates[candidate_index])
        value.pop("initial_detrended_peak_m", None)
        value["id"] = f"RAIL-{number:04d}"
        lines.append(value)
    for pair in pairs:
        pair["cluster_indexes"] = [index_to_line[value] for value in pair.pop("candidate_indexes")]

    report = {
        "schema_version": "railway.rail-candidates.v1",
        "project_id": reference.get("project_id"),
        "segment_id": segment_id,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "reference_report": str(reference_source),
        "reference_report_sha256": sha256_file(reference_source),
        "frame": frame.to_json(),
        "longitudinal_range_m": list(longitudinal_range),
        "search_z_range_m": list(z_range),
        "detection_method": "gislab_railtrack_3557ecf_plus_point_output_normalizer_v1",
        "execution_mode": "unmodified_upstream_detection_plus_common_schema_normalization",
        "shared_preprocessing": "reference_frame_and_frozen_z_envelope_only",
        "source_coordinate_space": source_coordinate_space,
        "generated_at": datetime.now(UTC).isoformat(),
        "upstream": {
            "repository": "https://github.com/GISLab-ELTE/railroad",
            "commit": "3557ecff1bd284108c7a833cce2b50ec9fcd5189",
            "lastools_commit": "9bdc92c73047b46be25e5c2ed4abda2521e30fba",
            "algorithm": "RailTrack",
        },
        "parameters": {
            "cross_bin_m": cross_bin_m,
            "smoothing_sigma_m": smoothing_sigma_m,
            "minimum_peak_spacing_m": minimum_peak_spacing_m,
            "assignment_radius_m": assignment_radius_m,
            "minimum_cluster_length_m": minimum_cluster_length_m,
            "minimum_cluster_points": minimum_cluster_points,
            "maximum_input_points": maximum_input_points,
            "source_coordinate_space": source_coordinate_space,
            "gauge_pairing_range_m": [1.35, 1.65],
            "assumed_rail_head_width_m": rail_head_width_m,
        },
        "source_point_count": source_point_count,
        "normalized_point_count": len(points),
        "global_cross_trend": {"slope_m_per_m": trend_slope, "intercept_m": trend_intercept},
        "histogram_peak_count": len(peak_positions),
        "eligible_cluster_count": len(candidates),
        "eligible_clusters": candidates,
        "rail_pair_count": len(pairs),
        "rail_pairs": pairs,
        "rail_lines": lines,
        "status": "external_detector_points_normalized_no_manual_review",
        "limitations": [
            "The upstream detector emits points, so common line and gauge metrics require this explicit post-process.",
            "The common corridor frame and height envelope are shared for controlled evaluation.",
            "No nominal-gauge correction is applied to reported geometry.",
            "This is an annotation-free repeatability comparison, not ground-truth accuracy.",
        ],
    }
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite GISLab normalization report: {output}")
    write_json(output, report)
    return output
