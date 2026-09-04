from __future__ import annotations

import numpy as np

from railway_recon.supplemental_remote_context_facade import (
    _facade_prism_local,
    fit_remote_context_facade,
)


def test_fit_remote_context_facade_recovers_observed_plane() -> None:
    station, z = np.meshgrid(np.linspace(95.0, 100.0, 70), np.linspace(19.2, 25.2, 70))
    rng = np.random.default_rng(7)
    points = np.column_stack(
        (
            station.ravel(),
            42.585 + rng.normal(0.0, 0.008, station.size),
            z.ravel(),
        )
    )
    fit = fit_remote_context_facade(points)
    assert 42.56 <= fit["cross_peak_m"] <= 42.61
    assert fit["plane_point_count"] >= 4_000
    vertices, faces = _facade_prism_local(fit)
    assert vertices.shape == (8, 3)
    assert len(faces) == 6
