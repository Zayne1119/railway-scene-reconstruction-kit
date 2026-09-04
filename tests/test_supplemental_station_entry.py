from __future__ import annotations

import numpy as np

from railway_recon.supplemental_station_entry import (
    _panel_prism_local,
    _roof_prism_local,
)


def test_panel_prism_is_closed_and_non_degenerate() -> None:
    vertices, faces = _panel_prism_local(
        np.asarray(((1.0, 2.0), (4.0, 2.0), (4.0, 5.0), (1.0, 5.0))),
        cross_m=6.0,
        thickness_m=0.1,
    )
    assert vertices.shape == (8, 3)
    assert len(faces) == 6
    assert np.allclose(np.unique(vertices[:, 1]), (6.0, 6.1))


def test_sloped_roof_prism_preserves_endpoint_heights() -> None:
    vertices, faces = _roof_prism_local(
        station_start_m=10.0,
        station_end_m=20.0,
        cross_start_m=2.0,
        cross_end_m=5.0,
        start_top_z_m=8.0,
        end_top_z_m=4.0,
        thickness_m=0.2,
    )
    assert vertices.shape == (8, 3)
    assert len(faces) == 6
    assert np.allclose(vertices[4:, 2], (8.0, 4.0, 4.0, 8.0))
