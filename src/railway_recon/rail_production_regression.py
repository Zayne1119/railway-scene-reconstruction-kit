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


def _distance_stats(values: np.ndarray) -> dict[str, float | int | None]:
    if not len(values):
        return {"count": 0, "p50_m": None, "p90_m": None, "p95_m": None, "maximum_m": None}
    return {
        "count": len(values),
        "p50_m": float(np.percentile(values, 50)),
        "p90_m": float(np.percentile(values, 90)),
        "p95_m": float(np.percentile(values, 95)),
        "maximum_m": float(np.max(values)),
    }


def _line_support_distance_groups(
    report: dict[str, Any],
    tree: cKDTree,
    *,
    core_length_m: float,
    sample_step_m: float,
) -> list[np.ndarray]:
    start, end = (float(value) for value in report["longitudinal_range_m"])
    midpoint = (start + end) / 2.0
    half_core = min(core_length_m / 2.0, (end - start) / 2.0)
    longitudinal = np.arange(
        midpoint - half_core,
        midpoint + half_core + sample_step_m * 0.5,
        sample_step_m,
    )
    groups: list[np.ndarray] = []
    for line in report["rail_lines"]:
        cross = (
            float(line["cross_fit_slope_m_per_m"]) * longitudinal
            + float(line["cross_fit_intercept_m"])
        )
        z = (
            float(line["z_fit_slope_m_per_m"]) * longitudinal
            + float(line["z_fit_intercept_m"])
        )
        samples = np.column_stack((longitudinal, cross, z))
        distances, _ = tree.query(samples, workers=-1)
        groups.append(np.asarray(distances, dtype=np.float64))
    return groups


def _concatenate_distance_groups(groups: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(groups) if groups else np.empty(0, dtype=np.float64)


def audit_rail_production_regression(
    baseline_reports: list[tuple[str, str | Path]],
    candidate_reports: list[tuple[str, str | Path]],
    output_path: str | Path,
    *,
    support_clouds: list[tuple[str, str | Path]] | None = None,
    allow_additional_lines: bool = False,
    core_length_m: float = 50.0,
    sample_step_m: float = 0.25,
    maximum_cross_change_m: float = 0.001,
    per_segment_support_p90_tolerance_m: float = 0.005,
    maximum_matched_line_support_p90_m: float = 0.060,
    maximum_additional_line_support_p90_m: float = 0.075,
) -> Path:
    if core_length_m <= 0 or sample_step_m <= 0:
        raise ValueError("Support sampling dimensions must be positive")
    baseline_paths = {segment_id: Path(path).resolve() for segment_id, path in baseline_reports}
    candidate_paths = {segment_id: Path(path).resolve() for segment_id, path in candidate_reports}
    if len(baseline_paths) != len(baseline_reports) or len(candidate_paths) != len(
        candidate_reports
    ):
        raise ValueError("Duplicate segment id in rail regression inputs")
    if not baseline_paths or set(baseline_paths) != set(candidate_paths):
        raise ValueError("Baseline and candidate segment sets differ")
    support_paths = (
        {segment_id: Path(path).resolve() for segment_id, path in support_clouds}
        if support_clouds is not None
        else {}
    )
    if support_clouds is not None and len(support_paths) != len(support_clouds):
        raise ValueError("Duplicate segment id in support clouds")
    if support_paths and set(support_paths) != set(baseline_paths):
        raise ValueError("Support cloud segment set differs from reports")

    segments: list[dict[str, Any]] = []
    baseline_all: list[np.ndarray] = []
    candidate_all: list[np.ndarray] = []
    for segment_id in sorted(baseline_paths):
        baseline_path = baseline_paths[segment_id]
        candidate_path = candidate_paths[segment_id]
        baseline = load_json(baseline_path)
        candidate = load_json(candidate_path)
        for name, value in (("baseline", baseline), ("candidate", candidate)):
            if value.get("schema_version") != "railway.rail-candidates.v1":
                raise ValueError(f"Unsupported {name} rail report for {segment_id}")
            if value.get("segment_id") != segment_id:
                raise ValueError(f"Segment id mismatch in {name} report: {segment_id}")
        baseline_source = Path(baseline["source"]).resolve()
        candidate_source = Path(candidate["source"]).resolve()
        if baseline_source != candidate_source and not support_paths:
            raise ValueError(f"Source cloud changed for {segment_id}")
        support_source = support_paths.get(segment_id, baseline_source)
        if not support_source.is_file():
            raise FileNotFoundError(support_source)
        baseline_frame = CorridorFrame.from_json(baseline["frame"])
        candidate_frame = CorridorFrame.from_json(candidate["frame"])
        frame_delta = max(
            float(np.max(np.abs(baseline_frame.origin_xy - candidate_frame.origin_xy))),
            float(np.max(np.abs(baseline_frame.along_xy - candidate_frame.along_xy))),
            float(np.max(np.abs(baseline_frame.cross_xy - candidate_frame.cross_xy))),
        )
        if frame_delta > 1e-9:
            raise ValueError(f"Corridor frame changed for {segment_id}")

        cloud = laspy.read(support_source)
        longitudinal, cross = baseline_frame.project(
            np.asarray(cloud.x, dtype=np.float64),
            np.asarray(cloud.y, dtype=np.float64),
        )
        points = np.column_stack(
            (longitudinal, cross, np.asarray(cloud.z, dtype=np.float64))
        )
        tree = cKDTree(points)
        baseline_distance_groups = _line_support_distance_groups(
            baseline,
            tree,
            core_length_m=core_length_m,
            sample_step_m=sample_step_m,
        )
        candidate_distance_groups = _line_support_distance_groups(
            candidate,
            tree,
            core_length_m=core_length_m,
            sample_step_m=sample_step_m,
        )
        baseline_distances = _concatenate_distance_groups(baseline_distance_groups)
        candidate_distances = _concatenate_distance_groups(candidate_distance_groups)
        baseline_all.append(baseline_distances)
        candidate_all.append(candidate_distances)

        baseline_lines = sorted(
            baseline["rail_lines"], key=lambda item: float(item["cross_position_m"])
        )
        candidate_lines = sorted(
            candidate["rail_lines"], key=lambda item: float(item["cross_position_m"])
        )
        costs = np.abs(
            np.asarray(
                [float(item["cross_position_m"]) for item in baseline_lines],
                dtype=np.float64,
            )[:, None]
            - np.asarray(
                [float(item["cross_position_m"]) for item in candidate_lines],
                dtype=np.float64,
            )[None, :]
        )
        rows, columns = linear_sum_assignment(costs)
        matches = list(zip(rows.tolist(), columns.tolist(), strict=True))
        cross_deltas = [float(costs[row, column]) for row, column in matches]
        z_deltas = [
            float(candidate_lines[column]["z_fit_intercept_m"])
            - float(baseline_lines[row]["z_fit_intercept_m"])
            for row, column in matches
        ]
        matched_candidate = {column for _, column in matches}
        additional_candidate_positions = [
            float(item["cross_position_m"])
            for index, item in enumerate(candidate_lines)
            if index not in matched_candidate
        ]
        matched_candidate_distances = _concatenate_distance_groups(
            [candidate_distance_groups[index] for index in sorted(matched_candidate)]
        )
        additional_candidate_distances = _concatenate_distance_groups(
            [
                group
                for index, group in enumerate(candidate_distance_groups)
                if index not in matched_candidate
            ]
        )
        baseline_stats = _distance_stats(baseline_distances)
        candidate_stats = _distance_stats(candidate_distances)
        matched_candidate_stats = _distance_stats(matched_candidate_distances)
        additional_candidate_stats = _distance_stats(additional_candidate_distances)
        pair_delta = int(candidate["rail_pair_count"]) - int(baseline["rail_pair_count"])
        line_delta = len(candidate_lines) - len(baseline_lines)
        topology_guardrails = {
            "rail_pair_count_valid": (
                pair_delta >= 0 if allow_additional_lines else pair_delta == 0
            ),
            "rail_line_count_valid": (
                line_delta >= 0 if allow_additional_lines else line_delta == 0
            ),
            "every_baseline_line_matched": len(matches) == len(baseline_lines),
            "cross_position_within_tolerance": (
                bool(cross_deltas) and max(cross_deltas) <= maximum_cross_change_m
            ),
            "additional_lines_form_complete_pairs": (
                line_delta == pair_delta * 2 if allow_additional_lines else line_delta == 0
            ),
        }
        matched_relative_support_passes = (
                matched_candidate_stats["p90_m"] is not None
                and baseline_stats["p90_m"] is not None
                and float(matched_candidate_stats["p90_m"])
                <= float(baseline_stats["p90_m"]) + per_segment_support_p90_tolerance_m
        )
        matched_absolute_support_passes = (
            matched_candidate_stats["p90_m"] is not None
            and float(matched_candidate_stats["p90_m"])
            <= maximum_matched_line_support_p90_m
        )
        additional_support_passes = (
            bool(len(additional_candidate_distances))
            and additional_candidate_stats["p90_m"] is not None
            and float(additional_candidate_stats["p90_m"])
            <= maximum_additional_line_support_p90_m
        )
        support_guardrails = {
            "matched_support_p90_within_relative_or_absolute_limit": (
                matched_relative_support_passes or matched_absolute_support_passes
            ),
            "matched_100mm_coverage_not_lower": (
                float(np.mean(matched_candidate_distances <= 0.10))
                >= float(np.mean(baseline_distances <= 0.10))
            ),
            "additional_line_support_p90_within_limit": (
                additional_support_passes if allow_additional_lines else True
            ),
        }
        segments.append(
            {
                "segment_id": segment_id,
                "baseline_source": str(baseline_source),
                "candidate_source": str(candidate_source),
                "support_cloud": str(support_source),
                "support_cloud_sha256": sha256_file(support_source),
                "baseline_report": str(baseline_path),
                "baseline_report_sha256": sha256_file(baseline_path),
                "candidate_report": str(candidate_path),
                "candidate_report_sha256": sha256_file(candidate_path),
                "rail_pair_count": {
                    "baseline": int(baseline["rail_pair_count"]),
                    "candidate": int(candidate["rail_pair_count"]),
                },
                "rail_line_count": {
                    "baseline": len(baseline_lines),
                    "candidate": len(candidate_lines),
                },
                "maximum_cross_position_change_m": (
                    max(cross_deltas) if cross_deltas else None
                ),
                "additional_candidate_cross_positions_m": additional_candidate_positions,
                "z_fit_intercept_delta_m": z_deltas,
                "baseline_support_distance": baseline_stats,
                "candidate_support_distance": candidate_stats,
                "matched_candidate_support_distance": matched_candidate_stats,
                "additional_candidate_support_distance": additional_candidate_stats,
                "baseline_coverage_at_0_05m": float(
                    np.mean(baseline_distances <= 0.05)
                ),
                "candidate_coverage_at_0_05m": float(
                    np.mean(candidate_distances <= 0.05)
                ),
                "baseline_coverage_at_0_10m": float(
                    np.mean(baseline_distances <= 0.10)
                ),
                "candidate_coverage_at_0_10m": float(
                    np.mean(candidate_distances <= 0.10)
                ),
                "topology_guardrails": topology_guardrails,
                "support_guardrails": support_guardrails,
                "passes": all(topology_guardrails.values())
                and all(support_guardrails.values()),
            }
        )

    baseline_distances = np.concatenate(baseline_all)
    candidate_distances = np.concatenate(candidate_all)
    baseline_stats = _distance_stats(baseline_distances)
    candidate_stats = _distance_stats(candidate_distances)
    aggregate_guardrails = {
        "all_segments_pass": all(segment["passes"] for segment in segments),
        "aggregate_support_p90_improved": (
            float(candidate_stats["p90_m"]) < float(baseline_stats["p90_m"])
        ),
        "aggregate_support_p95_improved": (
            float(candidate_stats["p95_m"]) < float(baseline_stats["p95_m"])
        ),
        "aggregate_100mm_coverage_not_lower": (
            float(np.mean(candidate_distances <= 0.10))
            >= float(np.mean(baseline_distances <= 0.10))
        ),
    }
    report = {
        "schema_version": "railway.rail-production-regression.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "evaluation_role": "same-input-engineering-regression_not_independent_accuracy",
        "parameters": {
            "core_length_m": core_length_m,
            "sample_step_m": sample_step_m,
            "maximum_cross_change_m": maximum_cross_change_m,
            "per_segment_support_p90_tolerance_m": per_segment_support_p90_tolerance_m,
            "maximum_matched_line_support_p90_m": (
                maximum_matched_line_support_p90_m
            ),
            "maximum_additional_line_support_p90_m": (
                maximum_additional_line_support_p90_m
            ),
            "allow_additional_lines": allow_additional_lines,
        },
        "segments": segments,
        "aggregate": {
            "baseline_support_distance": baseline_stats,
            "candidate_support_distance": candidate_stats,
            "baseline_coverage_at_0_05m": float(np.mean(baseline_distances <= 0.05)),
            "candidate_coverage_at_0_05m": float(np.mean(candidate_distances <= 0.05)),
            "baseline_coverage_at_0_10m": float(np.mean(baseline_distances <= 0.10)),
            "candidate_coverage_at_0_10m": float(np.mean(candidate_distances <= 0.10)),
        },
        "guardrails": aggregate_guardrails,
        "status": (
            "pass" if all(aggregate_guardrails.values()) else "fail"
        ),
        "limitations": [
            "The same source cloud is used for fitting and support checks; this is engineering QA only.",
            "Independent point-holdout results remain the source for research repeatability claims.",
            "A passing report does not replace TrackGraph validation or human/owner review.",
        ],
    }
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite rail regression audit: {output}")
    write_json(output, report)
    return output
