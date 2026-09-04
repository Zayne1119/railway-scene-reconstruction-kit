from railway_recon.station_platform_pipeline import _contiguous_runs


def test_contiguous_runs_isolate_blocked_segments() -> None:
    order = ["A", "B", "C", "D", "E", "F"]
    assert _contiguous_runs(order, {"A", "B", "D", "E", "F"}) == [
        ["A", "B"],
        ["D", "E", "F"],
    ]
