from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import ProjectConfig
from .io import load_json, sha256_file, write_json


def _owned_station(
    local_s: float,
    local_c: float,
    platform_frame: dict[str, Any],
    ownership_frame: dict[str, Any],
) -> float:
    local_origin = np.asarray(platform_frame["origin_xy"], dtype=np.float64)
    world_xy = (
        local_origin
        + local_s * np.asarray(platform_frame["along_xy"], dtype=np.float64)
        + local_c * np.asarray(platform_frame["cross_xy"], dtype=np.float64)
    )
    ownership_origin = np.asarray(ownership_frame["origin_xy"], dtype=np.float64)
    ownership_along = np.asarray(ownership_frame["along_xy"], dtype=np.float64)
    ownership_along /= max(float(np.linalg.norm(ownership_along)), 1e-12)
    return float((world_xy - ownership_origin) @ ownership_along)


def evaluate_platform_candidate_gate(
    project: ProjectConfig,
    segment_id: str,
    platform_value: str | Path,
    output_value: str | Path,
    *,
    component_id: str | None = None,
    ownership_plan_value: str | Path | None = None,
    gap_point_evidence_value: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a candidate-only mesh gate from deterministic platform QA fields."""

    platform_path = project.resolve(platform_value)
    output_path = project.resolve(output_value)
    if not platform_path.is_file():
        raise FileNotFoundError(platform_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    report = load_json(platform_path)
    if report.get("schema_version") != "railway.platform-surface.v1":
        raise ValueError("Automatic platform gate requires a platform-surface report")
    if str(report.get("segment_id")) != segment_id:
        raise ValueError("Platform report belongs to a different segment")
    components = list(report.get("platform_components", []))
    if component_id is None:
        if len(components) != 1:
            raise ValueError("Select a component explicitly when the report is not singular")
        component = components[0]
    else:
        component = next(
            (item for item in components if str(item.get("id")) == component_id),
            None,
        )
        if component is None:
            raise ValueError(f"Unknown platform component: {component_id}")

    ownership_path = project.resolve(ownership_plan_value) if ownership_plan_value else None
    ownership_interval: list[float] | None = None
    ownership_frame: dict[str, Any] | None = None
    if ownership_path:
        ownership = load_json(ownership_path)
        entry = next(
            (
                item
                for item in ownership.get("segments", [])
                if str(item.get("segment_id")) == segment_id
            ),
            None,
        )
        if entry is None:
            raise ValueError(f"Segment missing from ownership plan: {segment_id}")
        ownership_interval = [float(value) for value in entry["owned_interval_m"]]
        ownership_frame = ownership["frame"]

    fits = list(component.get("fit_segments", []))
    gaps = list(component.get("interior_gap_candidates", []))
    if ownership_interval is not None and ownership_frame is not None:
        owned_fits = []
        for fit in fits:
            fit_cross = sum(float(value) for value in fit["observed_cross_range_m"]) / 2.0
            stations = [
                _owned_station(float(local_s), fit_cross, report["frame"], ownership_frame)
                for local_s in fit["longitudinal_range_m"]
            ]
            if max(stations) >= ownership_interval[0] and min(stations) <= ownership_interval[1]:
                owned_fits.append(fit)
        fits = owned_fits
        owned_gaps = []
        for gap in gaps:
            center_s = sum(float(value) for value in gap["longitudinal_range_m"]) / 2.0
            center_c = sum(float(value) for value in gap["cross_range_m"]) / 2.0
            station = _owned_station(center_s, center_c, report["frame"], ownership_frame)
            if ownership_interval[0] <= station <= ownership_interval[1]:
                owned_gaps.append(gap)
        gaps = owned_gaps
    gap_evidence_path = (
        project.resolve(gap_point_evidence_value) if gap_point_evidence_value else None
    )
    occluder_gap_ids: set[str] = set()
    if gap_evidence_path:
        gap_evidence = load_json(gap_evidence_path)
        if str(gap_evidence.get("segment_id")) != segment_id:
            raise ValueError("Gap point evidence belongs to a different segment")
        if gap_evidence.get("platform_report_sha256") != sha256_file(platform_path):
            raise ValueError("Gap point evidence is stale for the selected platform report")
        occluder_gap_ids = {
            str(item["gap_id"])
            for item in gap_evidence.get("gaps", [])
            if item.get("classification")
            == "vertical_occluder_point_support_not_platform_opening"
        }

    quality = project.value["quality"]
    residual_limit = float(quality["platform_fit_p90_max_m"])
    peripheral_max_area = float(
        quality.get("platform_candidate_peripheral_gap_max_area_m2", 1.5)
    )
    peripheral_fraction = float(
        quality.get("platform_candidate_peripheral_gap_fraction", 0.75)
    )
    residuals = [
        float(item["absolute_residual_p90_m"])
        for item in fits
    ]
    maximum_residual = max(residuals, default=float("inf"))
    minimum_inlier_fraction = min(
        (float(item.get("robust_inlier_fraction", 1.0)) for item in fits),
        default=0.0,
    )
    cross_min, cross_max = (
        float(value) for value in component.get("cross_range_m", [0.0, 1.0])
    )
    cross_span = max(cross_max - cross_min, 1e-9)
    suppressed_gaps = []
    unresolved_gaps = []
    for gap in gaps:
        if gap.get("gap_classification") != "unique_enclosed_gap_review_required":
            continue
        gap_id = str(gap["id"])
        center_cross = sum(float(value) for value in gap["cross_range_m"]) / 2.0
        outer_fraction = (
            (center_cross - cross_min) / cross_span
            if component.get("side", "right") == "right"
            else (cross_max - center_cross) / cross_span
        )
        if gap_id in occluder_gap_ids:
            suppressed_gaps.append(
                {"gap_id": gap_id, "reason": "vertical_occluder_point_support"}
            )
        elif (
            float(gap["area_m2"]) <= peripheral_max_area
            and outer_fraction >= peripheral_fraction
        ):
            suppressed_gaps.append(
                {"gap_id": gap_id, "reason": "small_peripheral_no_return_candidate"}
            )
        else:
            unresolved_gaps.append(gap)
    unresolved = len(unresolved_gaps)
    confirmed_openings = sum(
        str(item.get("mesh_action")) == "create_confirmed_opening"
        for item in gaps
    )
    checks = {
        "fit_segments_present": bool(residuals),
        "fit_residual_within_project_limit": maximum_residual <= residual_limit,
        "fit_inlier_fraction_acceptable": minimum_inlier_fraction
        >= float(quality.get("platform_fit_minimum_inlier_fraction", 0.65)),
        "no_unresolved_interior_gaps": unresolved == 0,
        "positive_longitudinal_extent": (
            float(component["longitudinal_range_m"][1])
            > float(component["longitudinal_range_m"][0])
        ),
        "automatic_surface_status": str(component.get("status", "")).startswith(
            "automatic_platform_surface_hypothesis"
        ),
    }
    passed = all(checks.values())
    result = {
        "schema_version": "railway.platform-mesh-gate.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "project_id": project.project_id,
        "segment_id": segment_id,
        "platform_component_id": str(component["id"]),
        "platform_report": str(platform_path),
        "platform_report_sha256": sha256_file(platform_path),
        "ownership_plan": str(ownership_path) if ownership_path else None,
        "owned_longitudinal_interval_m": ownership_interval,
        "gap_point_evidence": str(gap_evidence_path) if gap_evidence_path else None,
        "gate_mode": "automatic_candidate_only",
        "checks": checks,
        "maximum_fit_residual_p90_m": maximum_residual,
        "fit_residual_limit_m": residual_limit,
        "minimum_fit_inlier_fraction": minimum_inlier_fraction,
        "confirmed_opening_count": confirmed_openings,
        "unresolved_gap_count": unresolved,
        "evaluated_fit_segment_count": len(fits),
        "evaluated_gap_count": len(gaps),
        "suppressed_candidate_gap_count": len(suppressed_gaps),
        "suppressed_candidate_gaps": suppressed_gaps,
        "mesh_gate": "passed_automatic_candidate" if passed else "blocked_automatic_candidate",
        "passed": passed,
        "status": "pass_candidate" if passed else "blocked_candidate",
        "limitations": [
            "This gate authorizes candidate geometry only, not final survey acceptance.",
            "Cross-segment ownership clipping and seam QA are still required.",
        ],
    }
    write_json(output_path, result)
    return result
