import numpy as np

from railway_recon.platform_gap_point_evidence import (
    _resource_settings,
    analyze_platform_gap_point_evidence_data,
)


def _platform() -> dict:
    return {
        "project_id": "test",
        "segment_id": "s0000_0050m",
        "frame": {
            "origin_xy": [0.0, 0.0],
            "along_xy": [1.0, 0.0],
            "cross_xy": [0.0, 1.0],
        },
        "platform_components": [
            {
                "id": "PLATFORM-1",
                "fit_segments": [
                    {
                        "longitudinal_range_m": [0.0, 10.0],
                        "plane_z_equals_a_s_plus_b_c_plus_d": [0.0, 0.0, 1.0],
                    }
                ],
                "interior_gap_candidates": [
                    {
                        "id": "GAP-1",
                        "longitudinal_range_m": [4.0, 5.0],
                        "cross_range_m": [2.0, 3.0],
                        "gap_classification": "unique_enclosed_gap_review_required",
                    }
                ],
            }
        ],
    }


def test_above_surface_vertical_returns_support_occluder() -> None:
    x = np.full(100, 4.5)
    y = np.full(100, 2.5)
    z = np.linspace(1.2, 3.0, 100)
    result = analyze_platform_gap_point_evidence_data(
        _platform(), x, y, z, _resource_settings()
    )
    assert result["vertical_occluder_supported_count"] == 1
    assert result["gaps"][0]["mesh_action"] == "preserve_platform_surface_do_not_create_opening"


def test_below_surface_returns_do_not_auto_fill_opening() -> None:
    x = np.full(100, 4.5)
    y = np.full(100, 2.5)
    z = np.linspace(-1.0, 0.8, 100)
    result = analyze_platform_gap_point_evidence_data(
        _platform(), x, y, z, _resource_settings()
    )
    assert result["vertical_occluder_supported_count"] == 0
    assert result["gaps"][0]["classification"].startswith("below_surface_void_or_stair")
