from __future__ import annotations

import numpy as np

from railway_recon.adjacent_context_seam_audit import audit_subsystem_handoff


def test_subsystem_handoff_passes_small_station_and_cross_z_gap() -> None:
    current = np.asarray(
        [[9.95, -1.0, 20.0], [10.0, 1.0, 20.0], [10.0, 2.0, 20.1]]
    )
    adjacent = np.asarray(
        [[10.002, -1.0, 20.01], [10.002, 1.0, 20.01], [10.04, 2.0, 20.11]]
    )
    report = audit_subsystem_handoff(
        current,
        adjacent,
        seam_station_m=10.0,
        edge_depth_m=0.08,
        station_gap_max_m=0.05,
        cross_z_p90_max_m=0.05,
    )
    assert report["passed"] is True
    assert report["signed_station_gap_m"] < 0.01


def test_subsystem_handoff_blocks_large_cross_section_mismatch() -> None:
    current = np.asarray([[10.0, -1.0, 20.0], [10.0, 1.0, 20.0]])
    adjacent = np.asarray([[10.002, -1.0, 21.0], [10.002, 1.0, 21.0]])
    report = audit_subsystem_handoff(
        current,
        adjacent,
        seam_station_m=10.0,
        edge_depth_m=0.08,
        station_gap_max_m=0.05,
        cross_z_p90_max_m=0.20,
    )
    assert report["passed"] is False
    assert report["status"] == "handoff_reconciliation_required"


def test_subsystem_handoff_reports_overlap_larger_than_acceptance_tolerance() -> None:
    current = np.asarray([[10.25, -1.0, 20.0], [10.25, 1.0, 20.0]])
    adjacent = np.asarray([[10.002, -1.0, 20.0], [10.002, 1.0, 20.0]])
    report = audit_subsystem_handoff(
        current,
        adjacent,
        seam_station_m=10.0,
        edge_depth_m=0.08,
        station_gap_max_m=0.05,
        cross_z_p90_max_m=0.20,
    )
    assert report["signed_station_gap_m"] < -0.20
    assert report["gates"]["absolute_station_gap_within_limit"] is False
    assert report["passed"] is False


def test_slab_sampling_ignores_different_edge_tessellation() -> None:
    current = np.asarray(
        [
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 1.0],
            [10.0, 5.0, 1.0],
            [10.0, 5.0, 2.0],
            [10.0, 10.0, 2.0],
            [10.0, 10.0, 3.0],
        ]
    )
    adjacent = np.asarray(
        [
            [10.002, 0.0, 0.0],
            [10.002, 0.0, 1.0],
            [10.002, 10.0, 2.0],
            [10.002, 10.0, 3.0],
        ]
    )
    report = audit_subsystem_handoff(
        current,
        adjacent,
        seam_station_m=10.0,
        edge_depth_m=0.08,
        station_gap_max_m=0.05,
        cross_z_p90_max_m=0.03,
        sample_slab_section=True,
    )
    assert report["passed"] is True
    assert report["symmetric_cross_z_distance"]["p90_m"] < 0.01
