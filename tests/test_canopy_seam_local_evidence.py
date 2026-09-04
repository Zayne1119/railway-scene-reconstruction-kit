from __future__ import annotations

import numpy as np

from railway_recon.canopy_seam_local_evidence import (
    classify_edge_change,
    estimate_surface_support,
)


def _section() -> dict:
    return {
        "cross_min_m": 1.0,
        "cross_max_m": 3.0,
        "top_z_at_min_cross_m": 5.0,
        "top_z_at_max_cross_m": 5.0,
    }


def test_surface_support_recovers_observed_cross_extent() -> None:
    station = np.repeat(np.linspace(-1.0, -0.2, 10), 200)
    lateral = np.tile(np.linspace(1.1, 2.9, 200), 10)
    points = np.column_stack((station, lateral, np.full(len(station), 5.01)))
    result = estimate_surface_support(
        points,
        _section(),
        station_min_m=-1.1,
        station_max_m=-0.1,
    )
    assert result["status"] == "supported"
    assert 1.0 < result["observed_cross_min_m"] < 1.2
    assert 2.8 < result["observed_cross_max_m"] < 3.0


def test_edge_classifier_separates_model_error_from_physical_transition() -> None:
    assert (
        classify_edge_change(model_change_m=-0.42, observed_change_m=-0.03)
        == "model_segmentation_mismatch"
    )
    assert (
        classify_edge_change(model_change_m=-0.42, observed_change_m=-0.38)
        == "point_supported_physical_transition"
    )
