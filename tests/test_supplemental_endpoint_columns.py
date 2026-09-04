from __future__ import annotations

import numpy as np

from railway_recon.supplemental_endpoint_columns import robust_column_target


def test_robust_column_target_recovers_observed_shaft_center() -> None:
    rng = np.random.default_rng(20260902)
    points = np.column_stack(
        (
            rng.uniform(9.6, 10.4, 4000),
            rng.uniform(-5.3, -4.7, 4000),
            rng.uniform(20.0, 25.0, 4000),
        )
    )
    fit = robust_column_target(points)
    assert abs(fit["station_m"] - 10.0) < 0.03
    assert abs(fit["cross_m"] + 5.0) < 0.03
    assert 0.70 < fit["observed_along_extent_m"] < 0.85
    assert 0.50 < fit["observed_cross_extent_m"] < 0.65
