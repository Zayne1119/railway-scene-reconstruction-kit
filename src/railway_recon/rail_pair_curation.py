from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .io import load_json, sha256_file, write_json

BOUNDARY_REASONS = {"confirmed_turnout", "confirmed_buffer_stop"}


def _safe_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ValueError(
            "Curation output name may contain only letters, digits, dot, dash and underscore"
        )
    return value


def _resolve_source(selection_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (selection_path.parent / path).resolve()


def _selected_lines(
    report: dict[str, Any], pairs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    lines = list(report.get("rail_lines", []))
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for pair in pairs:
        for position in pair.get("cross_positions_m", []):
            candidates = [
                (index, line)
                for index, line in enumerate(lines)
                if index not in used
            ]
            if not candidates:
                raise ValueError("Accepted rail pair has no available rail line")
            index, line = min(
                candidates,
                key=lambda item: abs(
                    float(item[1]["cross_position_m"]) - float(position)
                ),
            )
            if abs(float(line["cross_position_m"]) - float(position)) > 0.15:
                raise ValueError("Accepted rail pair cannot be bound to its rail line")
            used.add(index)
            selected.append(dict(line))
    return selected


def _synthetic_pair(
    value: dict[str, Any], longitudinal_midpoint_m: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    track_id = str(value["track_id"])
    center = float(value["center_cross_m"])
    gauge = float(value.get("gauge_m", 1.435))
    rail_head_width = float(value.get("rail_head_width_m", 0.073))
    spacing = gauge + rail_head_width
    z = float(value["rail_top_z_m"])
    cross_slope = float(value.get("cross_slope_m_per_m", 0.0))
    z_slope = float(value.get("z_slope_m_per_m", 0.0))
    positions = [center - spacing / 2.0, center + spacing / 2.0]
    pair = {
        "peak_indexes": [],
        "cross_positions_m": positions,
        "observed_cross_positions_m": positions,
        "separation_m": spacing,
        "rail_center_spacing_m": spacing,
        "observed_rail_center_spacing_m": spacing,
        "gauge_m": gauge,
        "observed_gauge_m": gauge,
        "rail_position_correction_m": 0.0,
        "gauge_constrained_to_nominal": True,
        "rail_top_crosslevel_m": 0.0,
        "score": 0.0,
        "track_id": track_id,
        "evidence_level": "rule_inferred",
        "inference_reason": str(value["reason"]),
    }
    lines = [
        {
            "id": f"{track_id}:RAIL-{index:02d}",
            "cross_position_m": position,
            "median_z_m": z,
            "point_count": 0,
            "cross_fit_slope_m_per_m": cross_slope,
            "cross_fit_intercept_m": position
            - cross_slope * longitudinal_midpoint_m,
            "cross_fit_sample_count": 0,
            "cross_fit_residual_p90_m": None,
            "z_fit_slope_m_per_m": z_slope,
            "z_fit_intercept_m": z - z_slope * longitudinal_midpoint_m,
            "z_fit_sample_count": 0,
            "z_fit_residual_p90_m": None,
            "evidence_level": "rule_inferred",
        }
        for index, position in enumerate(positions, start=1)
    ]
    return pair, lines


def curate_rail_pairs(
    project: ProjectConfig,
    selection_value: str | Path,
    output_name: str,
) -> dict[str, Any]:
    selection_path = Path(selection_value).resolve()
    selection = load_json(selection_path)
    if selection.get("schema_version") != "railway.rail-pair-curation.v1":
        raise ValueError("Unsupported rail pair curation schema")
    if selection.get("project_id") != project.project_id:
        raise ValueError("Rail pair curation project_id does not match project")
    review = selection.get("review", {})
    approved_by = str(review.get("approved_by", "")).strip()
    reason = str(review.get("reason", "")).strip()
    if not approved_by or not reason:
        raise ValueError("Rail pair curation requires review.approved_by and review.reason")
    if review.get("independent_review_completed") is not False:
        raise ValueError(
            "This command records project-owner curation only; "
            "independent_review_completed must be false"
        )

    root = project.workspace_path("reports") / _safe_name(output_name)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite curation output: {root}")
    root.mkdir(parents=True)
    selection_hash = sha256_file(selection_path)
    created_at = datetime.now(UTC).isoformat()
    outputs: list[dict[str, Any]] = []
    seen_segments: set[str] = set()
    for item in selection.get("segments", []):
        segment_id = str(item["segment_id"])
        if segment_id in seen_segments:
            raise ValueError(f"Duplicate curation segment: {segment_id}")
        seen_segments.add(segment_id)
        source = _resolve_source(selection_path, str(item["source_report"]))
        report = load_json(source)
        if report.get("segment_id") != segment_id:
            raise ValueError(f"Curation source segment mismatch: {source}")
        source_hash = sha256_file(source)
        expected_hash = item.get("source_sha256")
        if expected_hash is not None and str(expected_hash) != source_hash:
            raise ValueError(f"Curation source hash mismatch: {source}")

        original_pairs = list(report.get("rail_pairs", []))
        original_ids = {str(pair["track_id"]) for pair in original_pairs}
        accepted_ids = {str(value) for value in item.get("accepted_track_ids", [])}
        rejected_ids = {str(value) for value in item.get("rejected_track_ids", [])}
        if accepted_ids & rejected_ids or accepted_ids | rejected_ids != original_ids:
            raise ValueError(
                f"Curation must explicitly accept or reject every pair in {segment_id}"
            )
        accepted_pairs = [
            dict(pair)
            for pair in original_pairs
            if str(pair["track_id"]) in accepted_ids
        ]
        selected_lines = _selected_lines(report, accepted_pairs)
        boundary_overrides = item.get("boundary_overrides", {})
        for pair in accepted_pairs:
            override = boundary_overrides.get(str(pair["track_id"]), {})
            for key in ("start_boundary_reason", "end_boundary_reason"):
                if key in override:
                    boundary_reason = str(override[key])
                    if boundary_reason not in BOUNDARY_REASONS:
                        raise ValueError(f"Unsupported topology boundary: {boundary_reason}")
                    pair[key] = boundary_reason

        longitudinal_range = report.get("core_longitudinal_range_m")
        if not isinstance(longitudinal_range, list) or len(longitudinal_range) != 2:
            longitudinal_range = report.get("longitudinal_range_m")
        if not isinstance(longitudinal_range, list) or len(longitudinal_range) != 2:
            raise ValueError(f"Curation source has no longitudinal range: {source}")
        longitudinal_midpoint = 0.5 * (
            float(longitudinal_range[0]) + float(longitudinal_range[1])
        )
        for synthetic in item.get("synthetic_pairs", []):
            pair, lines = _synthetic_pair(synthetic, longitudinal_midpoint)
            if str(pair["track_id"]) in {str(value["track_id"]) for value in accepted_pairs}:
                raise ValueError(f"Duplicate curated pair id in {segment_id}")
            accepted_pairs.append(pair)
            selected_lines.extend(lines)

        selected_peak_indexes = {
            int(index)
            for pair in accepted_pairs
            for index in pair.get("peak_indexes", [])
        }
        curated = dict(report)
        curated["rail_pairs"] = accepted_pairs
        curated["rail_pair_count"] = len(accepted_pairs)
        curated["rail_lines"] = selected_lines
        curated["candidate_point_count"] = sum(
            int(line.get("point_count", 0)) for line in selected_lines
        )
        curated["peaks"] = [
            peak
            for peak in report.get("peaks", [])
            if int(peak.get("grid_index", -1)) in selected_peak_indexes
        ]
        curated["peak_count"] = len(curated["peaks"])
        curated["review_status"] = "owner_override_accepted"
        curated["review"] = {
            "decision": "accepted_with_pair_curation",
            "mode": "project_owner_override",
            "approved_by": approved_by,
            "approved_at": created_at,
            "reason": reason,
            "independent_review_completed": False,
            "source_candidate_path": str(source),
            "source_candidate_sha256": source_hash,
            "selection_path": str(selection_path),
            "selection_sha256": selection_hash,
        }
        curated["pair_curation"] = {
            "accepted_original_track_ids": sorted(accepted_ids),
            "rejected_original_track_ids": sorted(rejected_ids),
            "synthetic_pair_ids": [
                str(value["track_id"]) for value in item.get("synthetic_pairs", [])
            ],
            "boundary_overrides": boundary_overrides,
        }
        output = root / f"{segment_id}_rail_candidates.curated.json"
        write_json(output, curated)
        outputs.append(
            {
                "segment_id": segment_id,
                "path": str(output),
                "sha256": sha256_file(output),
                "accepted_pair_count": len(accepted_pairs),
            }
        )

    result = {
        "schema_version": "railway.rail-pair-curation-result.v1",
        "project_id": project.project_id,
        "created_at": created_at,
        "status": "accepted",
        "passed": True,
        "review_mode": "project_owner_override",
        "independent_review_completed": False,
        "approved_by": approved_by,
        "reason": reason,
        "selection_path": str(selection_path),
        "selection_sha256": selection_hash,
        "segment_count": len(outputs),
        "curated_sources": outputs,
        "warning": "Pair curation is project-owner evidence interpretation, not independent review.",
    }
    output_path = root / "curation-result.json"
    write_json(output_path, result)
    return {**result, "output_report_path": str(output_path)}
