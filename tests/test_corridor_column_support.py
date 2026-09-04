import numpy as np

from railway_recon.corridor_column_support import _support_metrics

SETTINGS = {
    "support_radius_m": 0.65,
    "vertical_bin_m": 0.4,
    "minimum_occupied_ratio": 0.2,
    "minimum_points": 10,
    "bottom_clearance_m": 0.1,
    "top_tolerance_m": 0.3,
}


def test_column_support_passes_for_vertically_distributed_points() -> None:
    z = np.linspace(0.2, 4.8, 50)
    metrics = _support_metrics(
        np.zeros_like(z),
        np.zeros_like(z),
        z,
        {
            "longitudinal_position_m": 0.0,
            "cross_position_m": 0.0,
            "minimum_z": 0.0,
            "maximum_z": 5.0,
        },
        SETTINGS,
    )

    assert metrics["supported"] is True
    assert metrics["occupied_vertical_ratio"] > 0.8


def test_column_support_rejects_ground_only_cluster() -> None:
    z = np.full(100, 0.2)
    metrics = _support_metrics(
        np.zeros_like(z),
        np.zeros_like(z),
        z,
        {
            "longitudinal_position_m": 0.0,
            "cross_position_m": 0.0,
            "minimum_z": 0.0,
            "maximum_z": 5.0,
        },
        SETTINGS,
    )

    assert metrics["supported"] is False
    assert metrics["point_count"] == 100
