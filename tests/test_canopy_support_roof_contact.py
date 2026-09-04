from __future__ import annotations

import numpy as np

from railway_recon.canopy_support_roof_contact import contiguous_true_ranges


def test_contiguous_true_ranges() -> None:
    values = np.asarray([False, True, True, False, True, True, True, False])
    assert contiguous_true_ranges(values) == [(1, 3), (4, 7)]


def test_contiguous_true_ranges_handles_open_end() -> None:
    assert contiguous_true_ranges(np.asarray([False, True, True])) == [(1, 3)]
