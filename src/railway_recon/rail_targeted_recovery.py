from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import laspy
import numpy as np

from .io import load_json, sha256_file, write_json


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


def _closest_line(report: dict[str, Any], position_m: float) -> dict[str, Any]:
    lines = list(report.get("rail_lines", []))
    if not lines:
        raise ValueError(f"Rail report has no line records: {report.get('segment_id')}")
    line = min(
        lines,
        key=lambda item: abs(float(item["cross_position_m"]) - position_m),
    )
    if abs(float(line["cross_position_m"]) - position_m) > 0.15:
        raise ValueError("TrackGraph rail cannot be bound to its source line")
    return line


def _fit_value(
    line: dict[str, Any], prefix: str, longitudinal: np.ndarray, fallback: float
) -> np.ndarray:
    slope = line.get(f"{prefix}_fit_slope_m_per_m")
    intercept = line.get(f"{prefix}_fit_intercept_m")
    if slope is None or intercept is None:
        return np.full(len(longitudinal), fallback, dtype=np.float64)
    return float(slope) * longitudinal + float(intercept)


def _fixed_center_support(
    cloud_path: Path,
    report: dict[str, Any],
    positions: list[float],
    *,
    longitudinal_bin_m: float,
    minimum_bin_points: int,
    cross_half_width_m: float,
    vertical_below_m: float,
    vertical_above_m: float,
) -> dict[str, Any]:
    frame = report["frame"]
    origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross_axis = np.asarray(frame["cross_xy"], dtype=np.float64)
    if origin.shape != (2,) or along.shape != (2,) or cross_axis.shape != (2,):
        raise ValueError("Invalid rail report frame")
    along /= np.linalg.norm(along)
    cross_axis /= np.linalg.norm(cross_axis)
    longitudinal_range = report.get("core_longitudinal_range_m")
    if not isinstance(longitudinal_range, list) or len(longitudinal_range) != 2:
        longitudinal_range = report.get("longitudinal_range_m")
    if not isinstance(longitudinal_range, list) or len(longitudinal_range) != 2:
        raise ValueError("Rail report has no longitudinal recovery range")
    start, end = (float(value) for value in longitudinal_range)
    bin_count = max(1, int(np.ceil((end - start) / longitudinal_bin_m)))
    lines = [_closest_line(report, position) for position in positions]
    counts = [np.zeros(bin_count, dtype=np.int64) for _ in positions]
    selected_point_counts = [0 for _ in positions]

    with laspy.open(cloud_path) as reader:
        for points in reader.chunk_iterator(2_000_000):
            x = np.asarray(points.x, dtype=np.float64)
            y = np.asarray(points.y, dtype=np.float64)
            z = np.asarray(points.z, dtype=np.float64)
            xy = np.column_stack((x, y)) - origin
            longitudinal = xy @ along
            cross = xy @ cross_axis
            core = (longitudinal >= start) & (longitudinal <= end)
            if not np.any(core):
                continue
            for index, (position, line) in enumerate(zip(positions, lines, strict=True)):
                expected_cross = _fit_value(line, "cross", longitudinal, position)
                fallback_z = float(line["median_z_m"])
                expected_z = _fit_value(line, "z", longitudinal, fallback_z)
                selected = (
                    core
                    & (np.abs(cross - expected_cross) <= cross_half_width_m)
                    & (z >= expected_z - vertical_below_m)
                    & (z <= expected_z + vertical_above_m)
                )
                if not np.any(selected):
                    continue
                rows = np.floor((longitudinal[selected] - start) / longitudinal_bin_m).astype(
                    np.int64
                )
                rows = np.clip(rows, 0, bin_count - 1)
                counts[index] += np.bincount(rows, minlength=bin_count)
                selected_point_counts[index] += int(np.count_nonzero(selected))

    support = [value >= minimum_bin_points for value in counts]
    joint = support[0] & support[1]
    asymmetric = support[0] ^ support[1]
    supported_rows = np.flatnonzero(joint)
    if len(supported_rows) >= 2:
        internal = joint[int(supported_rows[0]) : int(supported_rows[-1]) + 1]
        maximum_internal_gap_bins = _longest_false_run(internal)
    else:
        maximum_internal_gap_bins = len(joint)
    return {
        "longitudinal_bin_count": bin_count,
        "left_point_count": selected_point_counts[0],
        "right_point_count": selected_point_counts[1],
        "left_support_ratio": float(np.mean(support[0])),
        "right_support_ratio": float(np.mean(support[1])),
        "joint_support_ratio": float(np.mean(joint)),
        "asymmetric_support_ratio": float(np.mean(asymmetric)),
        "maximum_internal_joint_gap_m": float(
            maximum_internal_gap_bins * longitudinal_bin_m
        ),
    }


def evaluate_targeted_rail_recovery(
    graph_value: str | Path,
    output_value: str | Path,
    *,
    observation_ids: list[str] | None = None,
    longitudinal_bin_m: float = 0.50,
    minimum_bin_points: int = 3,
    cross_half_width_m: float = 0.10,
    vertical_below_m: float = 0.12,
    vertical_above_m: float = 0.05,
    minimum_joint_support_ratio: float = 0.60,
    maximum_asymmetric_support_ratio: float = 0.25,
    maximum_internal_joint_gap_m: float = 5.0,
) -> Path:
    if longitudinal_bin_m <= 0.0 or minimum_bin_points < 1:
        raise ValueError("Targeted recovery bin settings must be positive")
    graph_path = Path(graph_value).resolve()
    output_path = Path(output_value).resolve()
    graph = load_json(graph_path)
    if graph.get("schema_version") != "railway.track-graph.v1":
        raise ValueError("Targeted recovery requires a TrackGraph v1 input")
    observations = {str(item["id"]): item for item in graph.get("observations", [])}
    sources = {str(item["segment_id"]): item for item in graph.get("sources", [])}
    requested = list(dict.fromkeys(observation_ids or []))
    missing = sorted(set(requested) - set(observations))
    if missing:
        raise ValueError(f"Unknown targeted recovery observations: {missing}")
    selected_ids = [
        str(task["observation_id"])
        for task in graph.get("pair_continuity_recovery", [])
        if task.get("status") == "unresolved"
    ]
    selected_ids = list(dict.fromkeys([*selected_ids, *requested]))

    results: list[dict[str, Any]] = []
    for observation_id in selected_ids:
        observation = observations[observation_id]
        source = sources[str(observation["segment_id"])]
        report_path = Path(str(source["path"])).resolve()
        if sha256_file(report_path) != str(source["sha256"]):
            raise ValueError(f"TrackGraph source report changed: {report_path}")
        report = load_json(report_path)
        cloud_path = Path(str(report["source"])).resolve()
        if not cloud_path.is_file():
            raise FileNotFoundError(cloud_path)
        positions = [float(value) for value in observation["rail_cross_positions_local_m"]]
        support = _fixed_center_support(
            cloud_path,
            report,
            positions,
            longitudinal_bin_m=longitudinal_bin_m,
            minimum_bin_points=minimum_bin_points,
            cross_half_width_m=cross_half_width_m,
            vertical_below_m=vertical_below_m,
            vertical_above_m=vertical_above_m,
        )
        passed = (
            float(support["joint_support_ratio"]) >= minimum_joint_support_ratio
            and float(support["asymmetric_support_ratio"])
            <= maximum_asymmetric_support_ratio
            and float(support["maximum_internal_joint_gap_m"])
            <= maximum_internal_joint_gap_m
        )
        results.append(
            {
                "observation_id": observation["id"],
                "global_track_id": observation["global_track_id"],
                "segment_id": observation["segment_id"],
                "status": "fixed_center_evidence_recovered" if passed else "unresolved",
                "rail_cross_positions_local_m": positions,
                "source_report": str(report_path),
                "source_cloud": str(cloud_path),
                "support": support,
            }
        )
    recovered_count = sum(
        item["status"] == "fixed_center_evidence_recovered" for item in results
    )
    report = {
        "schema_version": "railway.targeted-rail-recovery.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "graph_path": str(graph_path),
        "graph_sha256": sha256_file(graph_path),
        "parameters": {
            "requested_observation_ids": requested,
            "longitudinal_bin_m": longitudinal_bin_m,
            "minimum_bin_points": minimum_bin_points,
            "cross_half_width_m": cross_half_width_m,
            "vertical_below_m": vertical_below_m,
            "vertical_above_m": vertical_above_m,
            "minimum_joint_support_ratio": minimum_joint_support_ratio,
            "maximum_asymmetric_support_ratio": maximum_asymmetric_support_ratio,
            "maximum_internal_joint_gap_m": maximum_internal_joint_gap_m,
        },
        "summary": {
            "candidate_count": len(results),
            "recovered_count": recovered_count,
            "unresolved_count": len(results) - recovered_count,
        },
        "recoveries": results,
        "policy": "Fixed-center evidence evaluation only; no geometry is created or moved.",
    }
    write_json(output_path, report)
    return output_path


def apply_targeted_rail_recovery(
    graph_value: str | Path,
    recovery_value: str | Path,
    output_value: str | Path,
) -> Path:
    """Bind fixed-center evidence to a new graph without editing geometry."""

    graph_path = Path(graph_value).resolve()
    recovery_path = Path(recovery_value).resolve()
    output_path = Path(output_value).resolve()
    graph = load_json(graph_path)
    recovery = load_json(recovery_path)
    if recovery.get("schema_version") != "railway.targeted-rail-recovery.v1":
        raise ValueError("Unsupported targeted rail recovery report")
    if str(recovery.get("graph_sha256")) != sha256_file(graph_path):
        raise ValueError("Targeted recovery report does not match the TrackGraph")
    recovered = {
        str(item["observation_id"]): item
        for item in recovery.get("recoveries", [])
        if item.get("status") == "fixed_center_evidence_recovered"
    }
    observation_lookup = {
        str(item["id"]): item for item in graph.get("observations", [])
    }
    for observation_id, evidence in recovered.items():
        if observation_id not in observation_lookup:
            raise ValueError(f"Recovery references an unknown observation: {observation_id}")
        observation = observation_lookup[observation_id]
        original_positions = [
            float(value) for value in observation["rail_cross_positions_local_m"]
        ]
        recovered_positions = [
            float(value) for value in evidence["rail_cross_positions_local_m"]
        ]
        if not np.allclose(original_positions, recovered_positions, atol=1e-9, rtol=0.0):
            raise ValueError("Targeted recovery attempted to move rail geometry")
        observation["pair_recovery_status"] = "fixed_center_evidence_recovered"
        observation["pair_recovery_support_observation_id"] = None
        observation["targeted_recovery_support"] = evidence["support"]
        # A rule-inferred center is allowed to become point-supported only after
        # this command has verified both rails at the exact frozen geometry.
        # The original evidence class is retained explicitly for provenance.
        if observation.get("evidence_level") == "rule_inferred":
            observation["source_evidence_level"] = "rule_inferred"
            observation["evidence_level"] = "observed"
            observation["evidence_interpretation"] = "fixed_center_point_supported"
            for edge in graph.get("edges", []):
                if str(edge.get("source_observation_id")) != observation_id:
                    continue
                edge["source_evidence_level"] = "rule_inferred"
                edge["evidence_level"] = "observed"
                edge["evidence_interpretation"] = "fixed_center_point_supported"
    for task in graph.get("pair_continuity_recovery", []):
        observation_id = str(task["observation_id"])
        if observation_id in recovered:
            task["status"] = "fixed_center_evidence_recovered"
            task["fixed_center_support"] = recovered[observation_id]["support"]
    graph["targeted_pair_recovery"] = {
        "applied_at": datetime.now(UTC).isoformat(),
        "source_graph_path": str(graph_path),
        "source_graph_sha256": sha256_file(graph_path),
        "recovery_report_path": str(recovery_path),
        "recovery_report_sha256": sha256_file(recovery_path),
        "recovered_observation_count": len(recovered),
        "geometry_changed": False,
        "rule_inferred_observations_promoted_after_fixed_center_support": sum(
            item.get("evidence_interpretation") == "fixed_center_point_supported"
            for item in observation_lookup.values()
        ),
    }
    write_json(output_path, graph)
    return output_path
