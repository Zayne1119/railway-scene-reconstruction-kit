from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from railway_recon.assembly_interface_gate import (
    audit_obj_vertical_interfaces,
    evaluate_vertical_interface,
)


def _plate(z: float, *, start: float = 0.0, stop: float = 1.0) -> np.ndarray:
    axis = np.linspace(start, stop, 11)
    return np.asarray([(x, y, z) for x in axis for y in axis], dtype=np.float64)


def test_flush_top_interface_passes_without_claiming_acceptance() -> None:
    result = evaluate_vertical_interface(
        _plate(2.0),
        _plate(2.0),
        source_side="top",
    )

    assert result["candidate_geometry_passed"] is True
    assert result["formal_acceptance"] is False
    assert result["status"] == "candidate_geometry_interface_pass_pending_review"
    assert result["contact_ratio"] == pytest.approx(1.0)


def test_detached_top_interface_fails_open_gap_gate() -> None:
    result = evaluate_vertical_interface(
        _plate(2.0),
        _plate(2.12),
        source_side="top",
        contact_tolerance_m=0.03,
    )

    assert result["candidate_geometry_passed"] is False
    assert result["gates"]["open_gap"] is False
    assert result["metrics"]["open_gap_p90_m"] == pytest.approx(0.12)


def test_shallow_bottom_insertion_requires_explicit_permission() -> None:
    source = _plate(0.96)
    bearing = _plate(1.0)

    conservative = evaluate_vertical_interface(
        source,
        bearing,
        source_side="bottom",
        contact_tolerance_m=0.02,
        maximum_insertion_m=0.08,
    )
    declared = evaluate_vertical_interface(
        source,
        bearing,
        source_side="bottom",
        contact_tolerance_m=0.02,
        insertion_allowed=True,
        maximum_insertion_m=0.08,
    )

    assert conservative["candidate_geometry_passed"] is False
    assert conservative["gates"]["insertion"] is False
    assert declared["candidate_geometry_passed"] is True
    assert declared["metrics"]["insertion_p90_m"] == pytest.approx(0.04)


def test_excessive_insertion_and_no_xy_overlap_fail_closed() -> None:
    excessive = evaluate_vertical_interface(
        _plate(0.80),
        _plate(1.0),
        source_side="bottom",
        insertion_allowed=True,
        maximum_insertion_m=0.08,
    )
    absent = evaluate_vertical_interface(
        _plate(1.0),
        _plate(1.0, start=5.0, stop=6.0),
        source_side="bottom",
    )

    assert excessive["gates"]["insertion"] is False
    assert absent["candidate_geometry_passed"] is False
    assert absent["target_overlap_sample_count"] == 0
    assert absent["metrics"]["open_gap_p90_m"] is None


def test_localized_contact_allows_sloped_surface_to_touch_at_one_edge() -> None:
    source = _plate(2.0)
    target = source.copy()
    target[:, 2] = 2.0 + 0.20 * target[:, 0]

    area = evaluate_vertical_interface(
        source,
        target,
        source_side="top",
        contact_tolerance_m=0.03,
        minimum_contact_ratio=0.05,
    )
    localized = evaluate_vertical_interface(
        source,
        target,
        source_side="top",
        contact_mode="localized",
        contact_tolerance_m=0.03,
        minimum_contact_ratio=0.05,
    )

    assert area["candidate_geometry_passed"] is False
    assert localized["candidate_geometry_passed"] is True
    assert localized["metrics"]["closest_signed_separation_m"] == pytest.approx(0.0)


def test_obj_batch_audit_samples_faces_and_reports_missing_objects(tmp_path: Path) -> None:
    model = tmp_path / "interface.obj"
    model.write_text(
        """o SUPPORT
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3 4
o BEARING
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 5 6 7 8
""",
        encoding="utf-8",
    )

    report = audit_obj_vertical_interfaces(
        model,
        [
            {
                "interface_id": "base-bearing",
                "declared_relation": "rests_on",
                "source_object": "SUPPORT",
                "target_objects": ["BEARING"],
                "source_side": "bottom",
            },
            {
                "interface_id": "missing-target",
                "source_object": "SUPPORT",
                "target_objects": ["ABSENT"],
                "source_side": "bottom",
            },
        ],
        sampling_spacing_m=0.25,
    )

    assert report["summary"] == {
        "interface_count": 2,
        "candidate_geometry_pass_count": 1,
        "review_required_count": 1,
        "formal_acceptance": False,
    }
    assert report["interfaces"][0]["candidate_geometry_passed"] is True
    assert report["interfaces"][1]["missing_objects"] == ["ABSENT"]


def test_invalid_thresholds_are_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be less"):
        evaluate_vertical_interface(
            _plate(0.0),
            _plate(0.0),
            source_side="bottom",
            contact_tolerance_m=0.05,
            maximum_insertion_m=0.04,
        )
