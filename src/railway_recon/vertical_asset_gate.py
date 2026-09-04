from __future__ import annotations

from typing import Any


def evaluate_vertical_asset_gate(
    candidate: dict[str, Any],
    interpretation: dict[str, Any],
    *,
    asset_class: str,
) -> dict[str, Any]:
    """Require a real visible shaft before a vertical hypothesis becomes geometry."""
    if asset_class not in {"catenary_support", "canopy_column", "sign_or_marker_pole"}:
        raise ValueError(f"Unsupported vertical asset class: {asset_class}")
    trusted_views = int(interpretation.get("trusted_reviewable_view_count", 0))
    visible_shaft = bool(interpretation.get("visible_full_height_shaft", False))
    base_supported = bool(interpretation.get("base_support_confirmed", False))
    occupancy = float(candidate.get("features", {}).get("vertical_occupied_ratio", 0.0))
    minimum_occupancy = {
        "catenary_support": 0.65,
        "canopy_column": 0.65,
        "sign_or_marker_pole": 0.40,
    }[asset_class]
    checks: dict[str, bool] = {
        "at_least_two_trusted_reviewable_views": trusted_views >= 2,
        "visible_full_height_shaft": visible_shaft,
        "base_support_confirmed": base_supported,
        "vertical_point_occupancy": occupancy >= minimum_occupancy,
    }
    rail_distance = candidate.get("rail_context", {}).get("nearest_rail_distance_m")
    if asset_class == "catenary_support":
        checks["minimum_rail_clearance_1_0m"] = (
            rail_distance is not None and float(rail_distance) >= 1.0
        )
    passed = all(checks.values())
    return {
        "schema_version": "railway.vertical-asset-gate.v1",
        "candidate_id": candidate.get("id"),
        "asset_class": asset_class,
        "checks": checks,
        "vertical_occupied_ratio": occupancy,
        "minimum_vertical_occupied_ratio": minimum_occupancy,
        "nearest_rail_distance_m": (
            float(rail_distance) if rail_distance is not None else None
        ),
        "passed": passed,
        "status": "pass" if passed else "reject_or_hold_no_geometry",
        "failed_checks": [name for name, value in checks.items() if not value],
        "limitations": [
            "Photo projection indicates a review location; only a visible physical shaft counts as semantic confirmation.",
            "The rail-clearance threshold is a reconstruction sanity gate, not a design-code compliance statement.",
        ],
    }
