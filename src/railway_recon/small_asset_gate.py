from __future__ import annotations

from typing import Any


def evaluate_freestanding_sign_gate(
    candidate: dict[str, Any],
    interpretation: dict[str, Any],
    *,
    minimum_point_count: int = 250,
    minimum_dark_point_fraction: float = 0.20,
) -> dict[str, Any]:
    """Gate a point-cloud component before it becomes a freestanding sign mesh."""
    spans = [float(value) for value in candidate.get("span_s_c_z_m", (0.0, 0.0, 0.0))]
    if len(spans) != 3:
        raise ValueError("span_s_c_z_m must contain longitudinal, cross and height spans")
    trusted_views = int(interpretation.get("trusted_reviewable_view_count", 0))
    visible_sign = bool(interpretation.get("visible_freestanding_sign", False))
    base_supported = bool(interpretation.get("base_support_confirmed", False))
    point_count = int(candidate.get("point_count", 0))
    dark_fraction = float(candidate.get("dark_point_fraction", 0.0))
    checks = {
        "point_support": point_count >= minimum_point_count,
        "plausible_width_0_7_to_4_0m": 0.7 <= spans[0] <= 4.0,
        "plausible_thickness_at_most_0_9m": 0.0 < spans[1] <= 0.9,
        "plausible_height_1_2_to_3_2m": 1.2 <= spans[2] <= 3.2,
        "dark_material_support": dark_fraction >= minimum_dark_point_fraction,
        "at_least_two_trusted_reviewable_views": trusted_views >= 2,
        "visible_freestanding_sign": visible_sign,
        "base_support_confirmed": base_supported,
    }
    passed = all(checks.values())
    return {
        "schema_version": "railway.freestanding-sign-gate.v1",
        "candidate_id": candidate.get("id"),
        "checks": checks,
        "metrics": {
            "point_count": point_count,
            "span_s_c_z_m": spans,
            "dark_point_fraction": dark_fraction,
            "trusted_reviewable_view_count": trusted_views,
        },
        "passed": passed,
        "failed_checks": [name for name, value in checks.items() if not value],
        "status": "pass" if passed else "reject_or_hold_no_geometry",
        "limitations": [
            "Dark material support is a configurable cue for this dataset, not a universal sign rule.",
            "Photo projection localizes review; visible sign semantics and platform contact remain mandatory.",
        ],
    }


def evaluate_platform_fence_gate(
    candidate: dict[str, Any],
    interpretation: dict[str, Any],
    *,
    minimum_point_count: int = 1000,
) -> dict[str, Any]:
    """Gate a long point-cloud component before it becomes a platform fence mesh."""
    spans = [float(value) for value in candidate.get("span_s_c_z_m", (0.0, 0.0, 0.0))]
    if len(spans) != 3:
        raise ValueError("span_s_c_z_m must contain longitudinal, cross and height spans")
    trusted_views = int(interpretation.get("trusted_reviewable_view_count", 0))
    point_count = int(candidate.get("point_count", 0))
    continuity = float(interpretation.get("longitudinal_support_fraction", 0.0))
    checks = {
        "point_support": point_count >= minimum_point_count,
        "plausible_longitudinal_span_at_least_3m": spans[0] >= 3.0,
        "plausible_thickness_at_most_0_8m": 0.0 < spans[1] <= 0.8,
        "plausible_height_0_6_to_2_0m": 0.6 <= spans[2] <= 2.0,
        "longitudinal_support_at_least_90_percent": continuity >= 0.90,
        "at_least_two_trusted_reviewable_views": trusted_views >= 2,
        "visible_platform_fence": bool(interpretation.get("visible_platform_fence", False)),
        "base_support_confirmed": bool(interpretation.get("base_support_confirmed", False)),
        "outside_platform_edge_confirmed": bool(
            interpretation.get("outside_platform_edge_confirmed", False)
        ),
    }
    passed = all(checks.values())
    return {
        "schema_version": "railway.platform-fence-gate.v1",
        "candidate_id": candidate.get("id"),
        "checks": checks,
        "metrics": {
            "point_count": point_count,
            "span_s_c_z_m": spans,
            "trusted_reviewable_view_count": trusted_views,
            "longitudinal_support_fraction": continuity,
        },
        "passed": passed,
        "failed_checks": [name for name, value in checks.items() if not value],
        "status": "pass" if passed else "reject_or_hold_no_geometry",
        "limitations": [
            "A core-boundary-clipped fence may pass as a clipped asset, but must not be labelled complete.",
            "Fine member spacing remains an explicit measured or photo-interpreted parameter.",
        ],
    }
