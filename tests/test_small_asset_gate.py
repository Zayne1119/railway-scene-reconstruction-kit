from __future__ import annotations

from railway_recon.small_asset_gate import (
    evaluate_freestanding_sign_gate,
    evaluate_platform_fence_gate,
)


def test_observed_multiview_sign_passes() -> None:
    candidate = {
        "id": "SIGN-1",
        "point_count": 1200,
        "span_s_c_z_m": [2.4, 0.25, 2.1],
        "dark_point_fraction": 0.65,
    }
    report = evaluate_freestanding_sign_gate(
        candidate,
        {
            "trusted_reviewable_view_count": 4,
            "visible_freestanding_sign": True,
            "base_support_confirmed": True,
        },
    )
    assert report["passed"] is True


def test_geometry_without_visible_sign_is_rejected() -> None:
    candidate = {
        "id": "EDGE-1",
        "point_count": 5000,
        "span_s_c_z_m": [2.0, 0.2, 2.0],
        "dark_point_fraction": 0.8,
    }
    report = evaluate_freestanding_sign_gate(
        candidate,
        {
            "trusted_reviewable_view_count": 4,
            "visible_freestanding_sign": False,
            "base_support_confirmed": False,
        },
    )
    assert report["passed"] is False
    assert "visible_freestanding_sign" in report["failed_checks"]


def test_out_of_range_sign_dimensions_are_rejected() -> None:
    candidate = {
        "id": "WALL-1",
        "point_count": 1000,
        "span_s_c_z_m": [8.0, 0.2, 2.0],
        "dark_point_fraction": 0.6,
    }
    report = evaluate_freestanding_sign_gate(
        candidate,
        {
            "trusted_reviewable_view_count": 3,
            "visible_freestanding_sign": True,
            "base_support_confirmed": True,
        },
    )
    assert report["passed"] is False
    assert "plausible_width_0_7_to_4_0m" in report["failed_checks"]


def test_continuous_multiview_platform_fence_passes() -> None:
    report = evaluate_platform_fence_gate(
        {
            "id": "FENCE-1",
            "point_count": 45000,
            "span_s_c_z_m": [50.0, 0.25, 1.4],
        },
        {
            "trusted_reviewable_view_count": 4,
            "longitudinal_support_fraction": 0.99,
            "visible_platform_fence": True,
            "base_support_confirmed": True,
            "outside_platform_edge_confirmed": True,
        },
    )
    assert report["passed"] is True


def test_short_disconnected_component_is_not_a_fence() -> None:
    report = evaluate_platform_fence_gate(
        {
            "id": "FRAGMENT-1",
            "point_count": 4000,
            "span_s_c_z_m": [1.8, 0.3, 1.2],
        },
        {
            "trusted_reviewable_view_count": 3,
            "longitudinal_support_fraction": 0.35,
            "visible_platform_fence": True,
            "base_support_confirmed": True,
            "outside_platform_edge_confirmed": True,
        },
    )
    assert report["passed"] is False
    assert "plausible_longitudinal_span_at_least_3m" in report["failed_checks"]
    assert "longitudinal_support_at_least_90_percent" in report["failed_checks"]
