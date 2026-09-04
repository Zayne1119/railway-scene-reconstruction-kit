from __future__ import annotations

import pytest

from railway_recon.supplemental_gap_mast_gate import interval_separation


def test_interval_separation_reports_clearance() -> None:
    assert interval_separation((-21.4, -21.1), (-20.85, -17.3)) == pytest.approx(0.25)
    assert interval_separation((-21.4, -21.1), (-28.5, -21.8)) == pytest.approx(0.4)


def test_interval_separation_returns_zero_for_overlap() -> None:
    assert interval_separation((-21.4, -21.1), (-21.2, -20.0)) == 0.0
