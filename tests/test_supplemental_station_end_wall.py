from __future__ import annotations

import numpy as np

from railway_recon.supplemental_station_end_wall import (
    _wall_prism_local,
    fit_station_end_wall,
)


def test_fit_station_end_wall_and_closed_prism() -> None:
    rng = np.random.default_rng(7)
    points = np.column_stack(
        (
            rng.normal(95.78, 0.04, 4_000),
            rng.uniform(22.2, 28.5, 4_000),
            rng.uniform(19.2, 24.7, 4_000),
        )
    )
    fit = fit_station_end_wall(points)
    assert abs(float(fit["station_peak_m"]) - 95.78) < 0.05
    assert float(fit["cross_max_m"]) - float(fit["cross_min_m"]) > 5.5
    vertices, faces = _wall_prism_local(fit)
    assert vertices.shape == (8, 3)
    assert len(faces) == 6
