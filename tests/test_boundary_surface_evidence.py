from __future__ import annotations

import numpy as np

from railway_recon.boundary_surface_evidence import summarize_station_bins


def test_summarize_station_bins_returns_separate_supported_runs() -> None:
    station = np.concatenate(
        (
            np.repeat(np.asarray([0.25, 0.75, 1.25]), 5),
            np.repeat(np.asarray([2.25, 2.75]), 5),
        )
    )
    report = summarize_station_bins(
        station,
        station_minimum_m=0.0,
        station_maximum_m=3.0,
        station_bin_m=0.5,
        minimum_points_per_bin=5,
    )
    assert report["occupied_station_runs_m"] == [[0.0, 1.5], [2.0, 3.0]]
    assert report["longest_occupied_run_m"] == 1.5
