from railway_recon.rail_recovery_selection import (
    _candidate_segment_ids,
    _trial_is_pareto_improvement,
)


def _audit(**failures: int) -> dict:
    return {
        "checks": [
            {"id": check, "failure_count": count}
            for check, count in failures.items()
        ]
    }


def test_pareto_recovery_accepts_non_regressing_seam_improvement() -> None:
    graph = {"tracks": [{"id": "TRACK-0001"}, {"id": "TRACK-0002"}]}
    baseline = _audit(segment_seams=4, rail_gauge=1, track_graph_structure=0)
    trial = _audit(segment_seams=3, rail_gauge=1, track_graph_structure=0)

    accepted, reasons, deltas = _trial_is_pareto_improvement(
        graph, baseline, graph, trial
    )

    assert accepted is True
    assert reasons == []
    assert deltas["segment_seams"] == -1


def test_pareto_recovery_rejects_quality_regression() -> None:
    graph = {"tracks": [{"id": "TRACK-0001"}, {"id": "TRACK-0002"}]}
    baseline = _audit(segment_seams=4, rail_top_crosslevel=0)
    trial = _audit(segment_seams=3, rail_top_crosslevel=1)

    accepted, reasons, _ = _trial_is_pareto_improvement(graph, baseline, graph, trial)

    assert accepted is False
    assert "quality_check_regressed:rail_top_crosslevel:+1" in reasons


def test_candidate_segments_include_interior_empty_and_graph_gap() -> None:
    manifest = {
        "segments": [
            {
                "id": f"s{start:04d}_{start + 50:04d}m",
                "chainage_start_m": float(start),
                "chainage_end_m": float(start + 50),
            }
            for start in range(0, 250, 50)
        ]
    }
    graph = {
        "tracks": [
            {"chainage_start_m": 0.0, "chainage_end_m": 250.0},
        ],
        "sources": [
            {"segment_id": "s0000_0050m", "rail_pair_count": 1},
            {"segment_id": "s0050_0100m", "rail_pair_count": 0},
            {"segment_id": "s0100_0150m", "rail_pair_count": 1},
            {"segment_id": "s0150_0200m", "rail_pair_count": 1},
            {"segment_id": "s0200_0250m", "rail_pair_count": 1},
        ],
        "observations": [
            {"id": "left", "chainage_end_m": 100.0},
            {"id": "right", "chainage_start_m": 200.0},
        ],
        "seams": [
            {
                "from_observation_id": "left",
                "to_observation_id": "right",
            }
        ],
    }

    assert _candidate_segment_ids(graph, manifest) == [
        "s0050_0100m",
        "s0100_0150m",
        "s0150_0200m",
    ]
