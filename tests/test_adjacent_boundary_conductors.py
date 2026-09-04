from __future__ import annotations

import numpy as np

from railway_recon.adjacent_boundary_conductors import (
    continuation_centerline,
    match_next_boundary_conductors,
)


def _fit(cross: float, z: float) -> dict[str, object]:
    return {
        "station_center_m": 125.0,
        "cross_polynomial_relative_to_center": [0.0, cross],
        "z_polynomial_relative_to_center": [0.0, 0.0, z],
    }


def test_match_next_boundary_conductors_uses_cross_and_height_not_ids() -> None:
    current = [
        {"candidate_id": "OLD-1", "fit": _fit(-16.5, 27.5)},
        {"candidate_id": "OLD-2", "fit": _fit(-3.8, 26.8)},
        {"candidate_id": "OLD-3", "fit": _fit(-3.7, 27.6)},
    ]
    adjacent = [
        {
            "candidate_id": "NEW-13",
            "station_range_m": [126.8, 132.2],
            "cross_range_m": [-3.95, -3.65],
            "z_range_m": [26.7, 26.9],
        },
        {
            "candidate_id": "NEW-02",
            "station_range_m": [126.8, 137.3],
            "cross_range_m": [-16.55, -16.45],
            "z_range_m": [27.3, 27.7],
        },
        {
            "candidate_id": "NEW-14",
            "station_range_m": [126.8, 132.2],
            "cross_range_m": [-3.75, -3.70],
            "z_range_m": [27.45, 27.7],
        },
    ]
    matches = match_next_boundary_conductors(
        current, adjacent, seam_station_m=126.5
    )
    mapping = {
        item["candidate"]["candidate_id"]: item["current_record"]["candidate_id"]
        for item in matches
    }
    assert mapping == {
        "NEW-02": "OLD-1",
        "NEW-13": "OLD-2",
        "NEW-14": "OLD-3",
    }


def test_continuation_centerline_keeps_small_nonoverlap_clearance() -> None:
    local = continuation_centerline(
        _fit(-4.0, 27.0),
        seam_station_m=126.5,
        observed_start_m=126.8,
        build_end_m=132.0,
        seam_endpoint_station_cross_z=[126.5, -4.1, 26.9],
    )
    assert np.isclose(local[0, 0], 126.502)
    assert local[-1, 0] == 132.0
    assert np.all(np.diff(local[:, 0]) > 0.0)
    assert np.linalg.norm(local[0, 1:] - np.asarray([-4.1, 26.9])) < 0.01
    assert np.allclose(local[-1, 1:], [-4.0, 27.0])
