from railway_recon.relationships import (
    aggregate_status,
    contact_status,
    interval_gap,
    percentile,
)


def test_interval_gap_handles_overlap_and_separation() -> None:
    assert interval_gap((0.0, 2.0), (1.0, 3.0)) == 0.0
    assert interval_gap((0.0, 1.0), (1.25, 2.0)) == 0.25
    assert interval_gap((3.0, 2.0), (1.0, 1.5)) == 0.5


def test_contact_status_is_signed_and_fail_closed() -> None:
    assert contact_status(0.02, maximum_gap_m=0.05, maximum_penetration_m=0.2) == "pass"
    assert contact_status(0.06, maximum_gap_m=0.05, maximum_penetration_m=0.2) == "gap"
    assert (
        contact_status(-0.21, maximum_gap_m=0.05, maximum_penetration_m=0.2)
        == "excessive_penetration"
    )
    assert (
        contact_status(None, maximum_gap_m=0.05, maximum_penetration_m=0.2)
        == "no_measurement"
    )


def test_percentile_and_aggregate_status() -> None:
    assert percentile([0.0, 1.0, 2.0, 3.0], 0.5) == 1.5
    assert percentile([], 0.9) is None
    assert aggregate_status(["pass", "pass"]) == "pass"
    assert aggregate_status(["pass", "gap"]) == "fail"
    assert aggregate_status([]) == "fail"
