import pytest

from railway_recon.track_assembly import (
    bed_interface_metrics,
    interval_gap,
    metrics_pass,
    sleeper_interface_metrics,
)


def test_interval_gap_distinguishes_gap_and_overlap() -> None:
    assert interval_gap((0.0, 1.0), (1.2, 2.0)) == pytest.approx(0.2)
    assert interval_gap((0.0, 1.0), (0.8, 2.0)) == pytest.approx(-0.2)


def test_sleeper_interface_metrics_use_center_phase_and_clear_gap() -> None:
    new = {
        "center_x_m": 1.0,
        "center_y_m": -99.6,
        "minimum_y_m": -99.72,
        "maximum_y_m": -99.48,
        "maximum_z_m": 0.18,
    }
    retained = {
        "center_x_m": 1.01,
        "center_y_m": -100.2,
        "minimum_y_m": -100.32,
        "maximum_y_m": -100.08,
        "maximum_z_m": 0.17,
    }
    metrics = sleeper_interface_metrics(new, retained, 0.6)
    assert metrics["center_spacing_m"] == pytest.approx(0.6)
    assert metrics["spacing_residual_m"] == pytest.approx(0.0)
    assert abs(metrics["clear_gap_m"] - 0.36) < 1e-12
    assert abs(metrics["lateral_offset_m"] - 0.01) < 1e-12


def test_bed_interface_metrics_and_gate() -> None:
    new = {
        "chainage_range_m": [-99.80, 100.0],
        "top_center_x_m": 2.0,
        "top_z_m": 0.1,
        "bottom_z_m": -0.5,
        "top_width_m": 3.0,
        "bottom_width_m": 4.0,
        "material_base_color": [0.4, 0.4, 0.4, 1.0],
    }
    retained = {
        "chainage_range_m": [-150.0, -99.76],
        "top_center_x_m": 2.03,
        "top_z_m": 0.11,
        "bottom_z_m": -0.52,
        "top_width_m": 3.02,
        "bottom_width_m": 4.04,
        "material_base_color": [0.42, 0.4, 0.39, 1.0],
    }
    metrics = bed_interface_metrics(new, retained)
    assert metrics["longitudinal_gap_m"] == pytest.approx(-0.04)
    assert metrics_pass(
        metrics,
        {
            "longitudinal_gap_m": 0.05,
            "top_center_lateral_offset_m": 0.05,
            "top_elevation_offset_m": 0.02,
            "bottom_elevation_offset_m": 0.03,
            "top_width_offset_m": 0.03,
            "bottom_width_offset_m": 0.05,
            "material_color_distance": 0.03,
        },
    )


def test_bed_endpoint_projection_and_skew_are_used_when_available() -> None:
    new = {
        "chainage_range_m": [-100.2, 50.0],
        "endpoint_longitudinal_m": -99.80,
        "endpoint_skew_m": 0.002,
        "top_center_x_m": 2.0,
        "top_z_m": 0.1,
        "bottom_z_m": -0.5,
        "top_width_m": 3.0,
        "bottom_width_m": 4.0,
        "material_base_color": [0.4, 0.4, 0.4, 1.0],
    }
    retained = {
        **new,
        "endpoint_longitudinal_m": -100.0,
        "endpoint_skew_m": 0.5,
    }
    metrics = bed_interface_metrics(new, retained)
    assert metrics["longitudinal_gap_m"] == pytest.approx(0.2)
    assert metrics["endpoint_skew_m"] == pytest.approx(0.5)
