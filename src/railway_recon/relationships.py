"""Small, dependency-free helpers for geometric relationship gates."""

from __future__ import annotations

import math
from collections.abc import Iterable


def interval_gap(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Return the non-negative separation between two closed intervals."""
    first_min, first_max = sorted(first)
    second_min, second_max = sorted(second)
    if first_max < second_min:
        return second_min - first_max
    if second_max < first_min:
        return first_min - second_max
    return 0.0


def contact_status(
    signed_gap_m: float | None,
    *,
    maximum_gap_m: float,
    maximum_penetration_m: float,
) -> str:
    """Classify a signed gap: positive is air, negative is penetration."""
    if signed_gap_m is None or not math.isfinite(signed_gap_m):
        return "no_measurement"
    if signed_gap_m > maximum_gap_m:
        return "gap"
    if signed_gap_m < -maximum_penetration_m:
        return "excessive_penetration"
    return "pass"


def percentile(values: Iterable[float], quantile: float) -> float | None:
    """Return a linearly interpolated percentile for finite values."""
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    position = (len(finite) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return finite[lower]
    weight = position - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def aggregate_status(statuses: Iterable[str]) -> str:
    """Fail closed unless every relationship measurement passes."""
    values = list(statuses)
    return "pass" if values and all(value == "pass" for value in values) else "fail"
