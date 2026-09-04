from __future__ import annotations

import numpy as np

from railway_recon.adjacent_surface_extensions import (
    endpoint_cross_plane,
    extension_slab_mesh,
    robust_plane_refit,
)
from railway_recon.geometry import CorridorFrame


def test_robust_plane_refit_rejects_sparse_outliers() -> None:
    station, cross = np.meshgrid(np.linspace(0.0, 5.0, 30), np.linspace(1.0, 3.0, 20))
    z = 0.01 * station - 0.1 * cross + 20.0
    points = np.column_stack((station.ravel(), cross.ravel(), z.ravel()))
    points = np.vstack((points, [[2.0, 2.0, 21.0], [3.0, 2.0, 19.0]]))
    report = robust_plane_refit(points, np.asarray([0.01, -0.1, 20.0]))
    assert report["passed"] is True
    assert report["absolute_residual_p90_m"] < 1.0e-8


def test_extension_slab_matches_seed_at_start_and_is_closed() -> None:
    frame = CorridorFrame(
        np.asarray([100.0, 200.0]), np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])
    )
    seed = np.asarray([0.0, 0.1, 20.0])
    fitted = np.asarray([0.01, 0.1, 20.0])
    vertices, faces = extension_slab_mesh(
        frame,
        station_start_m=10.0,
        station_end_m=15.0,
        cross_range_m=(1.0, 3.0),
        start_plane=seed,
        end_plane=fitted,
        thickness_m=0.12,
    )
    assert vertices.shape == (8, 3)
    assert len(faces) == 6
    assert np.allclose(vertices[4:6, 2], [20.1, 20.3])
    assert np.allclose(vertices[4:, 2] - vertices[:4, 2], 0.12)


def test_endpoint_cross_plane_matches_endpoint_corners() -> None:
    envelope = np.asarray(
        [[9.0, 1.0, 20.0], [10.0, 1.0, 20.1], [10.0, 3.0, 20.5]]
    )
    plane, report = endpoint_cross_plane(envelope, edge_station_m=10.0)
    predicted = plane[1] * np.asarray([1.0, 3.0]) + plane[2]
    assert np.allclose(predicted, [20.1, 20.5])
    assert report["endpoint_line_residual_maximum_m"] < 1.0e-9
