from railway_recon.corridor_canopy_candidate import _consecutive_runs


def test_consecutive_grid_runs_isolate_supported_outlier() -> None:
    items = [
        {"id": "A", "grid_index": -20},
        {"id": "B", "grid_index": 0},
        {"id": "C", "grid_index": 1},
        {"id": "D", "grid_index": 2},
    ]
    assert [[item["id"] for item in run] for run in _consecutive_runs(items)] == [
        ["A"],
        ["B", "C", "D"],
    ]
