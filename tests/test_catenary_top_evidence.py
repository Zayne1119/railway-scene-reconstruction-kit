from __future__ import annotations

import numpy as np

from railway_recon.catenary_top_evidence import evaluate_top_equipment_evidence


def _trace(z0: float, slope: float, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = np.linspace(0.25, 3.25, 80)
    return np.column_stack(
        (
            rng.normal(0.0, 0.12, len(u)),
            u,
            z0 + slope * u + rng.normal(0.0, 0.025, len(u)),
        )
    )


def test_complete_local_hardware_evidence_passes() -> None:
    top = 30.45
    upper = _trace(top - 1.75, 0.03, seed=1)
    u_down = np.linspace(0.55, 1.75, 60)
    lower = np.column_stack(
        (np.zeros_like(u_down), u_down, top - 1.8 - 0.65 * (u_down - 0.55))
    )
    u_up = np.linspace(1.70, 3.20, 70)
    positioner = np.column_stack(
        (np.zeros_like(u_up), u_up, top - 2.6 + 0.50 * (u_up - 1.70))
    )
    anchor_z = np.linspace(top - 2.02, top - 1.52, 24)
    anchor_upper = np.column_stack(
        (np.zeros_like(anchor_z), np.full_like(anchor_z, 0.65), anchor_z)
    )
    result = evaluate_top_equipment_evidence(
        np.vstack((upper, lower, positioner, anchor_upper)),
        shaft_top_z_m=top,
    )
    assert result["passed"] is True
    assert result["attachment_clusters"]


def test_longitudinal_wire_is_not_accepted_as_hardware() -> None:
    top = 30.45
    station = np.linspace(-2.25, 2.25, 180)
    wire = np.column_stack(
        (station, np.full_like(station, 2.8), np.full_like(station, top - 1.7))
    )
    result = evaluate_top_equipment_evidence(wire, shaft_top_z_m=top)
    assert result["passed"] is False
    assert result["conductor_like_point_count"] > 0
