from __future__ import annotations

import pytest

from railway_recon.canopy_group_refinement import (
    RIGHT_MAIN_ROOF_OBJECTS,
    compare_roof_cross_section_unions,
    evaluate_canopy_seam_candidate_gate,
    select_uniform_group_correction,
)


def _report(corrections: list[float]) -> dict:
    return {
        "records": [
            {
                "object_name": name,
                "passed": index != 1,
                "inlier_point_count": 1000 + index,
                "vertical_correction_at_center_m": correction,
                "residual_p90_m": 0.01,
                "plane_angle_change_deg": 0.4 if index != 1 else 1.6,
                "gates": {
                    "minimum_inlier_count": True,
                    "minimum_inlier_fraction": True,
                    "residual_p90_within_80mm": True,
                    "vertical_correction_within_200mm": True,
                    "slope_change_within_1_5deg": index != 1,
                },
            }
            for index, (name, correction) in enumerate(
                zip(RIGHT_MAIN_ROOF_OBJECTS, corrections, strict=True)
            )
        ]
    }


def test_uniform_group_gate_accepts_consistent_vertical_offsets() -> None:
    result = select_uniform_group_correction(_report([-0.11, -0.10, -0.112]))
    assert result["status"] == "passed_uniform_translation_gate"
    assert result["correction_spread_m"] == pytest.approx(0.012)
    assert result["maximum_post_group_center_residual_m"] < 0.01


def test_uniform_group_gate_rejects_inconsistent_offsets() -> None:
    with pytest.raises(ValueError, match="spread is unsafe"):
        select_uniform_group_correction(_report([-0.11, -0.04, -0.112]))


def _section(cross_min: float, cross_max: float, bottom: float, top: float) -> dict:
    return {
        "cross_min_m": cross_min,
        "cross_max_m": cross_max,
        "bottom_z_at_min_cross_m": bottom,
        "top_z_at_min_cross_m": top,
        "bottom_z_at_max_cross_m": bottom,
        "top_z_at_max_cross_m": top,
    }


def test_physical_seam_comparison_ignores_internal_partition() -> None:
    left = [_section(0.0, 1.0, 3.0, 3.1), _section(1.0, 2.0, 3.0, 3.1)]
    right = [_section(0.0, 2.0, 3.0, 3.1)]
    result = compare_roof_cross_section_unions(left, right)
    assert result["passed"] is True
    assert result["endpoint_nearest_distance_p90_m"] == pytest.approx(0.0)


def test_physical_seam_comparison_detects_vertical_offset() -> None:
    left = [_section(0.0, 2.0, 3.0, 3.1)]
    right = [_section(0.0, 2.0, 3.08, 3.18)]
    result = compare_roof_cross_section_unions(left, right)
    assert result["passed"] is False
    assert result["endpoint_nearest_distance_p90_m"] > 0.03


def test_seam_gate_rejects_new_plane_failure_and_support_loss() -> None:
    baseline_support = {
        "objects": [{"object_name": "roof", "coverage_at_0_10m": 0.65}]
    }
    candidate_support = {
        "objects": [{"object_name": "roof", "coverage_at_0_10m": 0.61}]
    }
    baseline_plane = {
        "passed_count": 1,
        "records": [{"object_name": "roof", "passed": True}],
    }
    candidate_plane = {
        "passed_count": 0,
        "records": [{"object_name": "roof", "passed": False}],
    }
    result = evaluate_canopy_seam_candidate_gate(
        baseline_support=baseline_support,
        candidate_support=candidate_support,
        baseline_plane_audit=baseline_plane,
        candidate_plane_audit=candidate_plane,
        weld_report={
            "changed_objects": ["roof"],
            "maximum_vertex_displacement_m": 0.42,
        },
        physical_seam_audit={"passed": True},
    )
    assert result["accepted"] is False
    assert result["new_plane_failures"] == ["roof"]
    assert result["decision"] == "stop_before_fixed_view_glb_and_web_promotion"
