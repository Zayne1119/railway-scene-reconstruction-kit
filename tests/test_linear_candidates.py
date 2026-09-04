from __future__ import annotations

import numpy as np

from railway_recon.algorithms.linear_candidates import (
    _vertical_candidates,
    _vertical_cell_continuity,
)


def test_vertical_cell_continuity_rejects_disconnected_top_and_bottom_surfaces() -> None:
    keys = np.zeros(42, dtype=np.int64)
    z = np.concatenate(
        [np.linspace(0.0, 0.2, 21), np.linspace(5.8, 6.0, 21)]
    )
    ratio, maximum_gap, occupied_bins = _vertical_cell_continuity(
        keys,
        z,
        np.asarray([True]),
        cell_count=1,
        z_bin_m=0.25,
    )
    assert 2 <= occupied_bins[0] <= 4
    assert ratio[0] < 0.20
    assert maximum_gap[0] > 5.0


def test_vertical_candidates_require_continuous_shaft_returns() -> None:
    rng = np.random.default_rng(7)
    shaft_z = np.linspace(0.0, 6.0, 180)
    shaft_x = rng.normal(0.05, 0.012, len(shaft_z))
    shaft_y = rng.normal(0.05, 0.012, len(shaft_z))

    surface_count = 120
    surface_x = rng.normal(1.05, 0.012, surface_count)
    surface_y = rng.normal(0.05, 0.012, surface_count)
    surface_z = np.concatenate(
        [rng.normal(0.05, 0.02, surface_count // 2), rng.normal(6.0, 0.02, surface_count // 2)]
    )
    config = {
        "xy_bin_m": 0.15,
        "minimum_z_span_m": 4.0,
        "minimum_cell_points": 12,
        "vertical_z_bin_m": 0.25,
        "minimum_vertical_occupied_ratio": 0.45,
        "maximum_vertical_gap_m": 1.0,
        "maximum_footprint_m": 1.5,
        "minimum_component_points": 30,
    }
    mask, records = _vertical_candidates(
        np.concatenate([shaft_x, surface_x]),
        np.concatenate([shaft_y, surface_y]),
        np.concatenate([shaft_z, surface_z]),
        config,
    )
    assert len(records) == 1
    assert records[0]["center_x"] < 0.2
    assert records[0]["features"]["vertical_occupied_ratio"] >= 0.45
    assert np.count_nonzero(mask[: len(shaft_z)]) > 0
    assert np.count_nonzero(mask[len(shaft_z) :]) == 0
