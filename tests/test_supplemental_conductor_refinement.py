from __future__ import annotations

import numpy as np

from railway_recon.supplemental_conductor_refinement import (
    evaluate_span_fit,
    fit_sagged_conductor_span,
)


def test_sagged_conductor_fit_recovers_parabolic_span() -> None:
    station = np.linspace(0.0, 50.0, 1001)
    cross = 3.0 + 0.001 * station
    z = 27.0 + 0.0012 * (station - 25.0) ** 2
    points = np.column_stack((station, cross, z))
    fit = fit_sagged_conductor_span(points, 0.0, 50.0, minimum_points_per_bin=2)
    assert fit["passed"] is True
    assert fit["sag_depth_m"] > 0.70
    predicted = evaluate_span_fit(fit, np.asarray((0.0, 25.0, 50.0)))
    assert np.allclose(predicted[:, 1], (3.0, 3.025, 3.05), atol=0.01)
    assert np.allclose(predicted[:, 2], (27.75, 27.0, 27.75), atol=0.02)
