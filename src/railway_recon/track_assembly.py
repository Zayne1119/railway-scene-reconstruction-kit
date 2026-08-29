"""Pure metrics for rail/sleeper/ballast assembly interfaces."""

from __future__ import annotations

import math
from typing import Any


def interval_gap(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Return positive separation or negative overlap between closed intervals."""

    first_min, first_max = sorted(first)
    second_min, second_max = sorted(second)
    if first_max < second_min:
        return second_min - first_max
    if second_max < first_min:
        return first_min - second_max
    return -min(first_max, second_max) + max(first_min, second_min)


def sleeper_interface_metrics(
    new: dict[str, float],
    retained: dict[str, float],
    nominal_spacing_m: float,
) -> dict[str, float]:
    """Compare the last new sleeper and first retained sleeper."""

    center_spacing = abs(new["center_y_m"] - retained["center_y_m"])
    return {
        "center_spacing_m": center_spacing,
        "spacing_residual_m": abs(center_spacing - nominal_spacing_m),
        "clear_gap_m": max(
            0.0,
            interval_gap(
                (new["minimum_y_m"], new["maximum_y_m"]),
                (retained["minimum_y_m"], retained["maximum_y_m"]),
            ),
        ),
        "lateral_offset_m": abs(new["center_x_m"] - retained["center_x_m"]),
        "top_offset_m": abs(new["maximum_z_m"] - retained["maximum_z_m"]),
    }


def bed_interface_metrics(
    new: dict[str, Any], retained: dict[str, Any]
) -> dict[str, float]:
    """Compare two ballast-bed endpoint cross-sections."""

    new_color = new.get("material_base_color", [0.0, 0.0, 0.0])
    retained_color = retained.get("material_base_color", [0.0, 0.0, 0.0])
    color_distance = math.sqrt(
        sum(
            (float(new_color[index]) - float(retained_color[index])) ** 2
            for index in range(3)
        )
    )
    if "endpoint_longitudinal_m" in new and "endpoint_longitudinal_m" in retained:
        longitudinal_gap = abs(
            float(new["endpoint_longitudinal_m"])
            - float(retained["endpoint_longitudinal_m"])
        )
    else:
        longitudinal_gap = interval_gap(
            tuple(new["chainage_range_m"]), tuple(retained["chainage_range_m"])
        )
    return {
        "longitudinal_gap_m": longitudinal_gap,
        "endpoint_skew_m": max(
            float(new.get("endpoint_skew_m", 0.0)),
            float(retained.get("endpoint_skew_m", 0.0)),
        ),
        "top_center_lateral_offset_m": abs(
            float(new["top_center_x_m"]) - float(retained["top_center_x_m"])
        ),
        "top_elevation_offset_m": abs(
            float(new["top_z_m"]) - float(retained["top_z_m"])
        ),
        "bottom_elevation_offset_m": abs(
            float(new["bottom_z_m"]) - float(retained["bottom_z_m"])
        ),
        "top_width_offset_m": abs(
            float(new["top_width_m"]) - float(retained["top_width_m"])
        ),
        "bottom_width_offset_m": abs(
            float(new["bottom_width_m"]) - float(retained["bottom_width_m"])
        ),
        "material_color_distance": color_distance,
    }


def metrics_pass(metrics: dict[str, float], thresholds: dict[str, float]) -> bool:
    """Return whether every metric satisfies its same-named maximum threshold."""

    return all(abs(float(metrics[key])) <= float(limit) for key, limit in thresholds.items())
