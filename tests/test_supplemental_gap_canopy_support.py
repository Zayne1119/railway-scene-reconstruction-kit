from __future__ import annotations

import numpy as np
import pytest

from railway_recon.supplemental_gap_canopy_support import fit_occluded_canopy_support


def test_fit_occluded_support_uses_base_and_upper_bands() -> None:
    rng = np.random.default_rng(10)
    base = np.column_stack(
        (
            84.15 + rng.normal(0.0, 0.03, 40),
            -27.90 + rng.normal(0.0, 0.05, 40),
            rng.uniform(21.0, 21.3, 40),
        )
    )
    upper = np.column_stack(
        (
            84.18 + rng.normal(0.0, 0.03, 120),
            -27.86 + rng.normal(0.0, 0.05, 120),
            rng.uniform(24.5, 25.6, 120),
        )
    )
    result = fit_occluded_canopy_support(
        np.vstack((base, upper)), minimum_z_m=21.0, maximum_z_m=25.6
    )
    assert result["station_m"] == pytest.approx(84.165, abs=0.04)
    assert result["cross_m"] == pytest.approx(-27.88, abs=0.05)
    assert result["continuous_shaft_observed"] is False


def test_fit_occluded_support_rejects_missing_upper_band() -> None:
    points = np.column_stack(
        (np.zeros(120), np.zeros(120), np.linspace(21.0, 21.3, 120))
    )
    with pytest.raises(ValueError, match="stable base or upper"):
        fit_occluded_canopy_support(points, minimum_z_m=21.0, maximum_z_m=25.6)
