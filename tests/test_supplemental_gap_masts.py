from __future__ import annotations

import numpy as np
import pytest

from railway_recon.supplemental_gap_masts import fit_observed_mast_shaft


def test_fit_observed_mast_shaft_recovers_robust_envelope() -> None:
    rng = np.random.default_rng(7)
    z = np.linspace(20.0, 28.0, 240)
    station = 61.5 + rng.uniform(-0.12, 0.12, len(z))
    cross = -21.2 + rng.uniform(-0.10, 0.10, len(z))
    result = fit_observed_mast_shaft(
        np.column_stack((station, cross, z)),
        minimum_z_m=20.0,
        maximum_z_m=28.0,
    )
    assert result["station_m"] == pytest.approx(61.5, abs=0.03)
    assert result["cross_m"] == pytest.approx(-21.2, abs=0.03)
    assert 0.18 <= result["along_width_m"] <= 0.25
    assert 0.18 <= result["cross_depth_m"] <= 0.25


def test_fit_observed_mast_shaft_rejects_short_fragment() -> None:
    points = np.column_stack(
        (np.zeros(100), np.zeros(100), np.linspace(20.0, 24.0, 100))
    )
    with pytest.raises(ValueError, match="too short"):
        fit_observed_mast_shaft(points, minimum_z_m=20.0, maximum_z_m=24.0)
