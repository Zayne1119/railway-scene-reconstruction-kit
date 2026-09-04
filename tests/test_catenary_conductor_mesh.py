from __future__ import annotations

import numpy as np
import pytest

from railway_recon.catenary_conductor_mesh import (
    audit_conductor_track_alignment,
    circular_profile,
    clip_observed_intervals,
    conductor_centerline,
)


def test_clip_observed_intervals_preserves_real_gaps() -> None:
    kept, rejected = clip_observed_intervals(
        [[-30.0, -28.0], [-9.5, 12.0], [21.0, 21.5]],
        [-20.0, 30.0],
        minimum_length_m=2.0,
    )
    assert kept == [[-9.5, 12.0]]
    assert len(rejected) == 2
    assert all(item["reason"] == "outside_core_or_below_minimum_supported_length" for item in rejected)


def test_circular_profile_rejects_unusable_mesh_settings() -> None:
    with pytest.raises(ValueError):
        circular_profile(0.0)
    with pytest.raises(ValueError):
        circular_profile(0.01, 5)
    profile = circular_profile(0.02, 8)
    assert profile.shape == (8, 2)
    np.testing.assert_allclose(np.linalg.norm(profile, axis=1), 0.02)


def test_conductor_centerline_applies_only_fitted_small_slopes() -> None:
    centerline = conductor_centerline(
        {
            "origin_xy": [100.0, 200.0],
            "along_xy": [1.0, 0.0],
            "cross_xy": [0.0, 1.0],
        },
        {
            "cross_position_m": -7.5,
            "elevation_z_m": 26.4,
            "cross_slope_m_per_m": 0.001,
            "elevation_slope_m_per_m": -0.002,
        },
        [-10.0, 10.0],
        reference_s_m=0.0,
        sample_spacing_m=2.0,
    )
    np.testing.assert_allclose(centerline[[0, -1], 0], [90.0, 110.0])
    np.testing.assert_allclose(centerline[[0, -1], 1], [192.49, 192.51])
    np.testing.assert_allclose(centerline[[0, -1], 2], [26.42, 26.38])


def test_conductor_alignment_requires_accepted_close_track() -> None:
    context = {
        "candidates": [
            {
                "candidate_id": "WIRE-A",
                "cross_position_m": -12.7,
                "nearest_track_center_distance_m": 0.12,
                "height_above_reference_rail_m": 6.2,
                "nearest_track_accepted_for_model": True,
            },
            {
                "candidate_id": "WIRE-B",
                "cross_position_m": -1.1,
                "nearest_track_center_distance_m": 0.02,
                "height_above_reference_rail_m": 6.3,
                "nearest_track_accepted_for_model": False,
            },
        ]
    }
    passed = audit_conductor_track_alignment(
        context,
        {
            "approved_conductors": [
                {"candidate_id": "WIRE-A", "linked_track_id": "TRACK-A"}
            ]
        },
    )
    assert passed["passed"] is True
    failed = audit_conductor_track_alignment(
        context,
        {
            "approved_conductors": [
                {"candidate_id": "WIRE-B", "linked_track_id": "TRACK-B"}
            ]
        },
    )
    assert failed["passed"] is False
