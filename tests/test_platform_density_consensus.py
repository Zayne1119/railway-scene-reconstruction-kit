from __future__ import annotations

from railway_recon.platform_density_consensus import evaluate_platform_density_consensus


def _candidate(identifier: str, s: list[float], c: list[float], z: list[float]) -> dict:
    return {
        "configuration_id": identifier,
        "settings": {"surface_grid_m": 0.5},
        "report": {
            "rail_reference_z_m": 20.0,
            "platform_components": [
                {
                    "id": f"PLATFORM-{identifier}",
                    "side": "left",
                    "occupied_cell_count": 100,
                    "longitudinal_range_m": s,
                    "cross_range_m": c,
                    "z_range_m": z,
                    "median_height_above_rail_m": 1.1,
                    "fit_segments": [{"id": "FIT"}],
                }
            ],
        },
    }


def test_consensus_accepts_stable_multi_resolution_components() -> None:
    result = evaluate_platform_density_consensus(
        [
            _candidate("A", [-19.0, 30.0], [-28.8, -21.8], [21.08, 21.16]),
            _candidate("B", [-18.5, 30.5], [-28.6, -22.0], [21.09, 21.17]),
            _candidate("C", [-18.0, 29.5], [-28.5, -22.2], [21.07, 21.17]),
        ],
        side="left",
    )
    assert result["passed"] is True
    assert result["accepted_configuration_count"] == 3
    assert result["consensus"]["longitudinal_range_m"] == [-18.5, 30.0]


def test_consensus_rejects_large_z_span_before_boundary_vote() -> None:
    result = evaluate_platform_density_consensus(
        [
            _candidate("A", [-19.0, 30.0], [-28.8, -21.8], [21.08, 21.16]),
            _candidate("B", [-18.5, 30.5], [-28.6, -22.0], [20.5, 21.2]),
        ],
        side="left",
        minimum_accepted_configurations=2,
    )
    assert result["passed"] is False
    assert result["accepted_configuration_count"] == 1
    rejected = next(item for item in result["configurations"] if item["configuration_id"] == "B")
    assert rejected["accepted_for_consensus"] is False


def test_consensus_rejects_unstable_cross_boundary() -> None:
    result = evaluate_platform_density_consensus(
        [
            _candidate("A", [-19.0, 30.0], [-28.8, -21.8], [21.08, 21.16]),
            _candidate("B", [-18.5, 30.5], [-28.6, -22.0], [21.09, 21.17]),
            _candidate("C", [-18.0, 29.5], [-31.0, -24.0], [21.07, 21.17]),
        ],
        side="left",
    )
    assert result["passed"] is False
    assert result["checks"]["outer_edge_stable"] is False
