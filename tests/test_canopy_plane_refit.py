from __future__ import annotations

import numpy as np

from railway_recon.canopy_plane_refit import fit_plane, robust_roof_plane_fit, top_envelope


def test_top_envelope_keeps_upper_vertex() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.1], [1.0, 0.0, 1.2], [0.0, 1.0, 1.3]]
    )
    envelope = top_envelope(vertices)
    assert len(envelope) == 3
    assert np.max(envelope[:, 2]) == 1.3
    assert np.min(envelope[:, 2]) == 1.1


def test_robust_roof_plane_fit_recovers_vertical_shift() -> None:
    roof = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [2.0, 2.0, 1.0],
            [0.0, 2.0, 1.0],
        ]
    )
    grid_x, grid_y = np.meshgrid(np.linspace(0.0, 2.0, 40), np.linspace(0.0, 2.0, 40))
    cloud = np.column_stack(
        (grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, 1.08))
    )
    result = robust_roof_plane_fit(cloud, roof)
    assert result["passed"]
    assert abs(result["vertical_correction_at_center_m"] - 0.08) < 1e-6
    plane = fit_plane(cloud)
    assert np.allclose(plane.coefficients, [0.0, 0.0, 1.08])
