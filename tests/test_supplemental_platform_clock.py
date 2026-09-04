from __future__ import annotations

import numpy as np

from railway_recon.supplemental_platform_clock import (
    _clock_disc_local,
    fit_platform_clock,
)


def test_fit_platform_clock_recovers_round_blue_grey_component() -> None:
    rng = np.random.default_rng(7)
    angle = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=False)
    station = 84.0 + rng.normal(0.0, 0.025, len(angle))
    cross = -27.5 + 0.42 * np.cos(angle) + rng.normal(0.0, 0.008, len(angle))
    z = 25.0 + 0.42 * np.sin(angle) + rng.normal(0.0, 0.008, len(angle))
    points = np.column_stack((station, cross, z))
    rgb = np.tile(np.asarray((120.0, 135.0, 165.0)), (len(points), 1))
    fit = fit_platform_clock(points, rgb, station_hint_m=84.0)
    assert abs(float(fit["cross_center_m"]) + 27.5) < 0.05
    assert abs(float(fit["z_center_m"]) - 25.0) < 0.05
    assert abs(float(fit["radius_m"]) - 0.42) < 0.05
    vertices, faces = _clock_disc_local(fit)
    assert len(vertices) == 64
    assert len(faces) == 34
