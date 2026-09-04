from __future__ import annotations

import numpy as np
import pytest

from railway_recon.model_point_support import ObjModel
from railway_recon.track_boundary_reconciliation import (
    reconcile_track_boundary_vertices,
    smoothstep_weights,
)


def _quad(offset: int) -> list[tuple[int, ...]]:
    return [(offset, offset + 1, offset + 2), (offset, offset + 2, offset + 3)]


def test_smoothstep_is_fixed_at_both_ends() -> None:
    result = smoothstep_weights(
        np.asarray([0.0, 5.0, 10.0]),
        transition_start_m=0.0,
        seam_station_m=10.0,
    )
    assert result.tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_reconciliation_matches_nearest_track_without_moving_transition_start() -> None:
    vertices = np.asarray(
        [
            [0.0, -1.1, 0.0],
            [10.0, -1.1, 0.0],
            [10.0, -0.9, 0.0],
            [0.0, -0.9, 0.0],
            [0.0, 0.9, 0.0],
            [10.0, 0.9, 0.0],
            [10.0, 1.1, 0.0],
            [0.0, 1.1, 0.0],
            [0.0, -1.5, -0.4],
            [10.0, -1.5, -0.4],
            [10.0, 1.5, -0.4],
            [0.0, 1.5, -0.4],
            [10.002, -1.2, 0.2],
            [20.0, -1.2, 0.2],
            [20.0, -1.0, 0.2],
            [10.002, -1.0, 0.2],
            [10.002, 0.8, 0.2],
            [20.0, 0.8, 0.2],
            [20.0, 1.0, 0.2],
            [10.002, 1.0, 0.2],
            [10.002, -1.6, -0.2],
            [20.0, -1.6, -0.2],
            [20.0, 1.4, -0.2],
            [10.002, 1.4, -0.2],
        ],
        dtype=np.float64,
    )
    model = ObjModel(
        vertices=vertices,
        faces_by_object={
            "TRACKGRAPH--TRACK-0001-RAIL-LEFT": _quad(0),
            "TRACKGRAPH--TRACK-0001-RAIL-RIGHT": _quad(4),
            "TRACKGRAPH--TRACK-0001-BED": _quad(8),
            "TRACK-0007-RAIL-LEFT": _quad(12),
            "TRACK-0007-RAIL-RIGHT": _quad(16),
            "TRACK-0007-BED": _quad(20),
        },
    )
    transformed, report = reconcile_track_boundary_vertices(
        model,
        origin_xyz=np.zeros(3),
        frame={
            "origin_xy": [0.0, 0.0],
            "along_xy": [1.0, 0.0],
            "cross_xy": [0.0, 1.0],
        },
        seam_station_m=10.0,
        transition_length_m=10.0,
    )
    assert report["mapping_count"] == 1
    assert transformed[0].tolist() == pytest.approx(vertices[0].tolist())
    assert transformed[1, 1] == pytest.approx(-1.2)
    assert transformed[1, 2] == pytest.approx(0.2)


def test_reconciliation_rejects_unsafe_vertical_jump() -> None:
    vertices = np.asarray(
        [
            [0.0, -1.1, 0.0],
            [10.0, -1.1, 0.0],
            [10.0, -0.9, 0.0],
            [0.0, -0.9, 0.0],
            [0.0, 0.9, 0.0],
            [10.0, 0.9, 0.0],
            [10.0, 1.1, 0.0],
            [0.0, 1.1, 0.0],
            [10.002, -1.1, 0.3],
            [20.0, -1.1, 0.3],
            [20.0, -0.9, 0.3],
            [10.002, -0.9, 0.3],
            [10.002, 0.9, 0.3],
            [20.0, 0.9, 0.3],
            [20.0, 1.1, 0.3],
            [10.002, 1.1, 0.3],
        ]
    )
    model = ObjModel(
        vertices,
        {
            "TRACKGRAPH--TRACK-0001-RAIL-LEFT": _quad(0),
            "TRACKGRAPH--TRACK-0001-RAIL-RIGHT": _quad(4),
            "TRACK-0001-RAIL-LEFT": _quad(8),
            "TRACK-0001-RAIL-RIGHT": _quad(12),
        },
    )
    with pytest.raises(ValueError, match="Unsafe vertical correction"):
        reconcile_track_boundary_vertices(
            model,
            origin_xyz=np.zeros(3),
            frame={
                "origin_xy": [0.0, 0.0],
                "along_xy": [1.0, 0.0],
                "cross_xy": [0.0, 1.0],
            },
            seam_station_m=10.0,
            transition_length_m=5.0,
        )
