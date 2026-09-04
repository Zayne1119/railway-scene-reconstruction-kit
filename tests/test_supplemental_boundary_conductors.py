from __future__ import annotations

import numpy as np

from railway_recon.supplemental_boundary_conductors import (
    fit_boundary_conductor_fragment,
)


def test_fit_boundary_conductor_fragment() -> None:
    rng = np.random.default_rng(7)
    station = np.repeat(np.linspace(120.0, 126.0, 31), 8)
    relative = station - 123.0
    cross = -4.0 + 0.01 * relative + rng.normal(0.0, 0.008, len(station))
    z = 27.0 + 0.012 * relative + 0.002 * relative**2
    z += rng.normal(0.0, 0.008, len(station))
    fit = fit_boundary_conductor_fragment(
        np.column_stack((station, cross, z)),
        station_start_m=120.0,
        station_end_m=126.0,
    )
    assert fit["passed"] is True
    assert fit["residual_p90_m"] < 0.03
    assert fit["occupied_station_bin_count"] >= 25
