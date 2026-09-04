from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import ProjectConfig
from .io import load_json, sha256_file, write_json
from .track_graph import load_track_graph_settings


def _pair_summary(report: dict[str, Any], pair: dict[str, Any]) -> dict[str, Any]:
    positions = [float(value) for value in pair["cross_positions_m"]]
    lines = list(report.get("rail_lines", []))
    top_values: list[float] = []
    for position in positions:
        if not lines:
            raise ValueError(f"Rail report has no rail lines: {report.get('segment_id')}")
        line = min(
            lines, key=lambda item: abs(float(item["cross_position_m"]) - position)
        )
        if line.get("median_z_m") is None:
            raise ValueError(f"Rail line has no median height: {line}")
        top_values.append(float(line["median_z_m"]))
    return {
        "track_id": pair.get("track_id"),
        "center_cross_m": float(np.mean(positions)),
        "cross_positions_m": positions,
        "rail_top_z_m": float(np.mean(top_values)),
        "rail_top_crosslevel_m": float(pair["rail_top_crosslevel_m"]),
        "gauge_m": float(pair["gauge_m"]),
        "observed_gauge_m": float(pair.get("observed_gauge_m", pair["gauge_m"])),
        "joint_support_ratio": float(pair.get("joint_support_ratio", 0.0)),
        "pair_continuity_status": pair.get("pair_continuity_status"),
    }


def compare_seam_rail_reports(
    project: ProjectConfig,
    left_report_path: Path,
    right_report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    left_path = left_report_path.resolve()
    right_path = right_report_path.resolve()
    destination = output_path.resolve()
    for path in (left_path, right_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    left_report = load_json(left_path)
    right_report = load_json(right_path)
    left_segment = str(left_report.get("segment_id"))
    right_segment = str(right_report.get("segment_id"))
    if left_segment != right_segment:
        raise ValueError(
            "Seam context reports must use the same segment frame and chainage window"
        )
    left_pairs = [_pair_summary(left_report, pair) for pair in left_report.get("rail_pairs", [])]
    right_pairs = [
        _pair_summary(right_report, pair) for pair in right_report.get("rail_pairs", [])
    ]
    settings = load_track_graph_settings(project)
    matching = settings["matching"]
    quality = settings["quality"]
    maximum_match_center = float(matching["maximum_lateral_difference_m"])
    thresholds = {
        "maximum_center_difference_m": float(quality["maximum_seam_lateral_error_m"]),
        "maximum_rail_top_difference_m": float(quality["maximum_seam_vertical_error_m"]),
        "maximum_observed_gauge_difference_m": float(quality["maximum_gauge_error_m"]),
    }

    assignments: list[tuple[int, int]] = []
    if left_pairs and right_pairs:
        impossible = 1e9
        costs = np.full((len(left_pairs), len(right_pairs)), impossible, dtype=np.float64)
        for left_index, left in enumerate(left_pairs):
            for right_index, right in enumerate(right_pairs):
                center_difference = abs(
                    float(left["center_cross_m"]) - float(right["center_cross_m"])
                )
                if center_difference <= maximum_match_center:
                    costs[left_index, right_index] = center_difference
        rows, columns = linear_sum_assignment(costs)
        assignments = [
            (int(row), int(column))
            for row, column in zip(rows, columns)
            if costs[row, column] < impossible
        ]

    matches: list[dict[str, Any]] = []
    for left_index, right_index in assignments:
        left = left_pairs[left_index]
        right = right_pairs[right_index]
        differences = {
            "center_m": abs(
                float(left["center_cross_m"]) - float(right["center_cross_m"])
            ),
            "rail_top_m": abs(
                float(left["rail_top_z_m"]) - float(right["rail_top_z_m"])
            ),
            "observed_gauge_m": abs(
                float(left["observed_gauge_m"]) - float(right["observed_gauge_m"])
            ),
        }
        passed = (
            differences["center_m"] <= thresholds["maximum_center_difference_m"]
            and differences["rail_top_m"]
            <= thresholds["maximum_rail_top_difference_m"]
            and differences["observed_gauge_m"]
            <= thresholds["maximum_observed_gauge_difference_m"]
        )
        matches.append(
            {
                "left": left,
                "right": right,
                "differences": differences,
                "status": "pass" if passed else "fail",
            }
        )
    assigned_left = {item[0] for item in assignments}
    assigned_right = {item[1] for item in assignments}
    unmatched_left = [
        pair for index, pair in enumerate(left_pairs) if index not in assigned_left
    ]
    unmatched_right = [
        pair for index, pair in enumerate(right_pairs) if index not in assigned_right
    ]
    if not left_pairs or not right_pairs:
        status = "insufficient_shared_rail_support"
    elif (
        len(matches) == len(left_pairs) == len(right_pairs)
        and all(item["status"] == "pass" for item in matches)
    ):
        status = "pass"
    else:
        status = "fail"

    report = {
        "schema_version": "railway.seam-rail-comparison.v1",
        "project_id": project.project_id,
        "segment_id": left_segment,
        "left": {
            "report": str(left_path),
            "sha256": sha256_file(left_path),
            "source": left_report.get("source"),
            "rail_pair_count": len(left_pairs),
        },
        "right": {
            "report": str(right_path),
            "sha256": sha256_file(right_path),
            "source": right_report.get("source"),
            "rail_pair_count": len(right_pairs),
        },
        "thresholds": thresholds,
        "matches": matches,
        "unmatched_left": unmatched_left,
        "unmatched_right": unmatched_right,
        "matched_pair_count": len(matches),
        "passed_pair_count": sum(item["status"] == "pass" for item in matches),
        "report_path": str(destination),
        "status": status,
        "mesh_authority": "none_evidence_only",
        "interpretation": (
            "The comparison is performed in one shared corridor frame. A pass supports "
            "cross-source rail continuity; insufficient support forbids automatic claims "
            "but does not prove a physical track termination."
        ),
    }
    write_json(destination, report)
    return report
