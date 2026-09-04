from __future__ import annotations

import numpy as np

from railway_recon.adjacent_platform_transition import (
    transition_slab_mesh,
    transition_surface_residuals,
)
from railway_recon.geometry import CorridorFrame


def test_transition_mesh_tapers_between_sections() -> None:
    frame = CorridorFrame(
        np.asarray([0.0, 0.0]), np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])
    )
    vertices, faces = transition_slab_mesh(
        frame,
        start_station_m=0.0,
        end_station_m=2.0,
        start_section=(1.0, 10.0, 20.0, 20.1),
        end_section=(1.1, 9.5, 20.0, 20.05),
        start_thickness_m=0.35,
        end_thickness_m=0.30,
    )
    assert vertices.shape == (8, 3)
    assert len(faces) == 6
    assert np.allclose(vertices[4:, 1], [1.0, 10.0, 9.5, 1.1])


def test_transition_residual_is_zero_on_bilinear_surface() -> None:
    points = np.asarray([[0.0, 1.0, 20.0], [1.0, 5.4, 20.0375], [2.0, 9.5, 20.05]])
    residual, within = transition_surface_residuals(
        points,
        start_station_m=0.0,
        end_station_m=2.0,
        start_section=(1.0, 10.0, 20.0, 20.1),
        end_section=(1.1, 9.5, 20.0, 20.05),
    )
    assert np.all(within)
    assert np.max(np.abs(residual)) < 1.0e-9
