from __future__ import annotations

from copy import deepcopy

from railway_recon.track_boundary_closure import evaluate_track_boundary_closure


def _seams(track_p90: float, track_p95: float) -> dict:
    return {
        "passed": True,
        "subsystems": {
            "track": {
                "signed_station_gap_m": 0.002,
                "symmetric_cross_z_distance": {
                    "p90_m": track_p90,
                    "p95_m": track_p95,
                    "maximum_m": 0.17,
                },
            }
        },
    }


def _support(p90: float, coverage: float) -> dict:
    return {
        "objects": [
            {
                "object_name": f"TRACKGRAPH--TRACK-000{track}-RAIL-{side}",
                "p50_m": 0.05,
                "p90_m": p90,
                "coverage_at_0_10m": coverage,
            }
            for track in range(1, 5)
            for side in ("LEFT", "RIGHT")
        ]
    }


def _evaluate() -> dict:
    return evaluate_track_boundary_closure(
        candidate_build={
            "transform": {"mapping_count": 4},
            "gates": {"identities": True, "mesh": True},
        },
        mesh={
            "passed": True,
            "degenerate_triangle_count": 0,
            "duplicate_face_count": 0,
        },
        baseline_seams=_seams(0.137, 0.15),
        candidate_seams=_seams(0.0005, 0.013),
        baseline_support=_support(0.0841, 0.9620),
        candidate_support=_support(0.0843, 0.9614),
        fixed_views={"view_count": 6, "views": [{} for _ in range(6)]},
    )


def test_track_boundary_closure_passes_small_support_regression() -> None:
    result = _evaluate()
    assert result["passed"] is True
    assert result["metrics"]["track_seam_p90_improvement_m"] > 0.13
    assert result["status"].endswith("not_promoted")


def test_track_boundary_closure_blocks_point_support_regression() -> None:
    result = _evaluate()
    support = _support(0.10, 0.94)
    blocked = evaluate_track_boundary_closure(
        candidate_build={
            "transform": {"mapping_count": 4},
            "gates": {"identities": True},
        },
        mesh={
            "passed": True,
            "degenerate_triangle_count": 0,
            "duplicate_face_count": 0,
        },
        baseline_seams=_seams(0.137, 0.15),
        candidate_seams=_seams(0.0005, 0.013),
        baseline_support=_support(0.0841, 0.9620),
        candidate_support=deepcopy(support),
        fixed_views={"view_count": 6, "views": [{} for _ in range(6)]},
    )
    assert result["passed"] is True
    assert blocked["passed"] is False
    assert blocked["gates"]["rail_point_support_p90_not_regressed"] is False
