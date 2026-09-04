from __future__ import annotations

from typing import Any

import numpy as np


def evaluate_platform_density_consensus(
    candidates: list[dict[str, Any]],
    *,
    side: str,
    minimum_accepted_configurations: int = 3,
    maximum_component_z_span_m: float = 0.20,
    maximum_longitudinal_boundary_spread_m: float = 2.0,
    maximum_cross_boundary_spread_m: float = 1.0,
    maximum_median_elevation_spread_m: float = 0.10,
) -> dict[str, Any]:
    """Require a low-density platform to agree across multiple grid settings."""
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    if minimum_accepted_configurations < 2:
        raise ValueError("At least two configurations are required for consensus")
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        report = candidate["report"]
        components = [
            item for item in report.get("platform_components", []) if item.get("side") == side
        ]
        if not components:
            records.append(
                {
                    "configuration_id": str(candidate["configuration_id"]),
                    "accepted_for_consensus": False,
                    "rejection_reason": "no_component_on_requested_side",
                }
            )
            continue
        component = max(components, key=lambda item: int(item.get("occupied_cell_count", 0)))
        s_range = [float(value) for value in component["longitudinal_range_m"]]
        c_range = [float(value) for value in component["cross_range_m"]]
        z_range = [float(value) for value in component["z_range_m"]]
        z_span = z_range[1] - z_range[0]
        median_elevation = float(report["rail_reference_z_m"]) + float(
            component["median_height_above_rail_m"]
        )
        accepted = z_span <= maximum_component_z_span_m and bool(component.get("fit_segments"))
        records.append(
            {
                "configuration_id": str(candidate["configuration_id"]),
                "settings": candidate.get("settings", {}),
                "component_id": component["id"],
                "occupied_cell_count": int(component["occupied_cell_count"]),
                "fit_segment_count": len(component.get("fit_segments", [])),
                "longitudinal_range_m": s_range,
                "cross_range_m": c_range,
                "z_range_m": z_range,
                "component_z_span_m": z_span,
                "median_elevation_m": median_elevation,
                "rail_side_edge_cross_m": c_range[1] if side == "left" else c_range[0],
                "outer_edge_cross_m": c_range[0] if side == "left" else c_range[1],
                "accepted_for_consensus": accepted,
                "rejection_reason": (
                    None
                    if accepted
                    else "component_z_span_or_fit_segment_gate_failed"
                ),
            }
        )

    accepted_records = [item for item in records if item["accepted_for_consensus"]]
    criteria = {
        "minimum_accepted_configurations": minimum_accepted_configurations,
        "maximum_component_z_span_m": maximum_component_z_span_m,
        "maximum_longitudinal_boundary_spread_m": maximum_longitudinal_boundary_spread_m,
        "maximum_cross_boundary_spread_m": maximum_cross_boundary_spread_m,
        "maximum_median_elevation_spread_m": maximum_median_elevation_spread_m,
    }
    if not accepted_records:
        return {
            "side": side,
            "configurations": records,
            "accepted_configuration_count": 0,
            "criteria": criteria,
            "passed": False,
            "status": "no_accepted_configuration",
        }

    s_low = np.asarray([item["longitudinal_range_m"][0] for item in accepted_records])
    s_high = np.asarray([item["longitudinal_range_m"][1] for item in accepted_records])
    outer = np.asarray([item["outer_edge_cross_m"] for item in accepted_records])
    rail_side = np.asarray([item["rail_side_edge_cross_m"] for item in accepted_records])
    elevation = np.asarray([item["median_elevation_m"] for item in accepted_records])
    spreads = {
        "longitudinal_low_m": float(np.ptp(s_low)),
        "longitudinal_high_m": float(np.ptp(s_high)),
        "outer_edge_cross_m": float(np.ptp(outer)),
        "rail_side_edge_cross_m": float(np.ptp(rail_side)),
        "median_elevation_m": float(np.ptp(elevation)),
    }
    checks = {
        "enough_configurations": len(accepted_records) >= minimum_accepted_configurations,
        "longitudinal_low_stable": spreads["longitudinal_low_m"]
        <= maximum_longitudinal_boundary_spread_m,
        "longitudinal_high_stable": spreads["longitudinal_high_m"]
        <= maximum_longitudinal_boundary_spread_m,
        "outer_edge_stable": spreads["outer_edge_cross_m"]
        <= maximum_cross_boundary_spread_m,
        "rail_side_edge_stable": spreads["rail_side_edge_cross_m"]
        <= maximum_cross_boundary_spread_m,
        "median_elevation_stable": spreads["median_elevation_m"]
        <= maximum_median_elevation_spread_m,
    }
    passed = all(checks.values())
    return {
        "side": side,
        "configurations": records,
        "accepted_configuration_count": len(accepted_records),
        "criteria": criteria,
        "spreads": spreads,
        "checks": checks,
        "consensus": {
            "longitudinal_range_m": [float(np.median(s_low)), float(np.median(s_high))],
            "cross_range_m": [
                float(np.median(outer)) if side == "left" else float(np.median(rail_side)),
                float(np.median(rail_side)) if side == "left" else float(np.median(outer)),
            ],
            "median_elevation_m": float(np.median(elevation)),
        },
        "passed": passed,
        "status": "pass" if passed else "review_required",
    }
