from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from .config import ProjectConfig
from .io import load_json, write_json
from .targeted_canopy_recovery import infer_column_grid


def _resource_settings() -> dict[str, Any]:
    path = resources.files("railway_recon.resources").joinpath("corridor-column-grid.default.json")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _canonical_position(candidate: dict[str, Any], frame: dict[str, Any]) -> tuple[float, float]:
    if "center_x" not in candidate or "center_y" not in candidate:
        raise ValueError("Column candidates require center_x and center_y")
    point = np.asarray(
        [float(candidate["center_x"]), float(candidate["center_y"])],
        dtype=np.float64,
    )
    origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    delta = point - origin
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    lateral = np.asarray(frame["lateral_xy"], dtype=np.float64)
    return float(np.dot(delta, along)), float(np.dot(delta, lateral))


def collect_canopy_column_seeds(
    ownership_plan: dict[str, Any],
    vertical_reports: dict[str, dict[str, Any]],
    *,
    side: str,
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect high-quality automatic seeds in one canonical corridor frame."""
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    segments = {item["segment_id"]: item for item in ownership_plan["segments"]}
    if set(vertical_reports) != set(segments):
        raise ValueError("Vertical report IDs must exactly match the ownership plan")
    frame = ownership_plan["frame"]
    minimum_height = float(settings["minimum_column_height_m"])
    maximum_height = float(settings["maximum_column_height_m"])
    minimum_vertical_ratio = float(settings["minimum_vertical_occupied_ratio"])
    minimum_abs_cross = float(settings["minimum_absolute_cross_position_m"])
    maximum_abs_cross = float(settings["maximum_absolute_cross_position_m"])
    reviewed_ids = {str(value) for value in settings.get("reviewed_seed_ids", [])}
    require_reviewed = bool(settings.get("require_reviewed_seed_ids", False))
    seeds: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for segment_id, report in vertical_reports.items():
        owned_low, owned_high = (float(value) for value in segments[segment_id]["owned_interval_m"])
        for candidate in report.get("candidates", []):
            candidate_id = str(candidate["id"])
            global_id = f"{segment_id}::{candidate_id}"
            station, cross = _canonical_position(candidate, frame)
            height = float(candidate.get("height_m", 0.0))
            vertical_ratio = float(
                candidate.get("features", {}).get("vertical_occupied_ratio", 0.0)
            )
            class_ok = candidate.get("predicted_class") == "canopy_column"
            side_ok = cross < 0.0 if side == "left" else cross > 0.0
            cross_ok = minimum_abs_cross <= abs(cross) <= maximum_abs_cross
            owned = owned_low <= station <= owned_high
            reviewed = candidate_id in reviewed_ids or global_id in reviewed_ids
            checks = {
                "predicted_canopy_column": class_ok,
                "requested_side": side_ok,
                "cross_range": cross_ok,
                "owned_by_segment": owned,
                "height_range": minimum_height <= height <= maximum_height,
                "strong_vertical_support": vertical_ratio >= minimum_vertical_ratio,
                "review_requirement": reviewed or not require_reviewed,
            }
            accepted = all(checks.values())
            decisions.append(
                {
                    "id": global_id,
                    "segment_id": segment_id,
                    "source_candidate_id": candidate_id,
                    "canonical_station_m": station,
                    "canonical_cross_m": cross,
                    "checks": checks,
                    "accepted_as_seed": accepted,
                }
            )
            if not accepted:
                continue
            seeds.append(
                {
                    "id": global_id,
                    "segment_id": segment_id,
                    "source_candidate_id": candidate_id,
                    "center_x": float(candidate["center_x"]),
                    "center_y": float(candidate["center_y"]),
                    "longitudinal_position_m": station,
                    "cross_position_m": cross,
                    "minimum_z": float(candidate["minimum_z"]),
                    "maximum_z": float(candidate["maximum_z"]),
                    "height_m": height,
                    "footprint_m": float(candidate["footprint_m"]),
                    "automatic_seed": not reviewed,
                    "reviewed_seed": reviewed,
                }
            )
    seeds.sort(key=lambda item: float(item["longitudinal_position_m"]))
    return seeds, decisions


def assign_column_grid_ownership(
    grid: list[dict[str, Any]], ownership_plan: dict[str, Any]
) -> list[dict[str, Any]]:
    segments = list(ownership_plan["segments"])
    if not segments:
        raise ValueError("Ownership plan contains no segments")
    result: list[dict[str, Any]] = []
    for item in grid:
        station = float(item["longitudinal_position_m"])
        owner: dict[str, Any] | None = None
        for index, segment in enumerate(segments):
            low, high = (float(value) for value in segment["owned_interval_m"])
            if low <= station < high or (index == len(segments) - 1 and station <= high):
                owner = segment
                break
        if owner is None:
            raise ValueError(f"Column grid position lies outside ownership coverage: {station}")
        record = dict(item)
        record["owner_segment_id"] = str(owner["segment_id"])
        low, high = (float(value) for value in owner["owned_interval_m"])
        record["distance_to_nearest_ownership_boundary_m"] = min(
            abs(station - low), abs(high - station)
        )
        result.append(record)
    return result


def recover_corridor_column_grid_data(
    ownership_plan: dict[str, Any],
    vertical_reports: dict[str, dict[str, Any]],
    *,
    side: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    seeds, decisions = collect_canopy_column_seeds(
        ownership_plan, vertical_reports, side=side, settings=settings
    )
    minimum_seeds = int(settings["minimum_seed_count"])
    if len(seeds) < minimum_seeds:
        raise ValueError("Too few canopy-column seeds for corridor grid recovery")
    coverage = [float(value) for value in ownership_plan["coverage_interval_m"]]
    grid, fit = infer_column_grid(
        seeds,
        coverage,
        float(settings["nominal_column_spacing_m"]),
        float(settings["minimum_column_spacing_m"]),
        float(settings["maximum_column_spacing_m"]),
        float(settings["seed_grid_match_tolerance_m"]),
    )
    maximum_residual = float(settings["maximum_seed_grid_residual_m"])
    if float(fit["maximum_absolute_seed_grid_residual_m"]) > maximum_residual:
        raise ValueError("Column-grid residual exceeds the configured gate")
    prefix = f"{side.upper()}-CANOPY-GRID"
    for index, item in enumerate(grid, start=1):
        item["id"] = f"{prefix}-{index:03d}"
        if item["reviewed_seed_id"]:
            seed = next(seed for seed in seeds if seed["id"] == item["reviewed_seed_id"])
            item["evidence_level"] = (
                "photo_interpreted_seed" if seed["reviewed_seed"] else "automatic_point_seed"
            )
            item["status"] = "seed_supported_position"
        else:
            item["status"] = "grid_inferred_pending_local_point_support"
    owned_grid = assign_column_grid_ownership(grid, ownership_plan)
    by_segment = {
        str(segment["segment_id"]): sum(
            item["owner_segment_id"] == segment["segment_id"] for item in owned_grid
        )
        for segment in ownership_plan["segments"]
    }
    inferred_count = sum(item["reviewed_seed_id"] is None for item in owned_grid)
    return {
        "schema_version": "railway.corridor-column-grid-recovery.v1",
        "side": side,
        "coverage_interval_m": coverage,
        "seed_count": len(seeds),
        "grid_position_count": len(owned_grid),
        "inferred_position_count": inferred_count,
        "seed_decisions": decisions,
        "seeds": seeds,
        "fit": fit,
        "grid": owned_grid,
        "grid_count_by_segment": by_segment,
        "passed": True,
        "status": "corridor_column_grid_recovered_geometry_support_still_required",
        "limitations": [
            "Grid-inferred positions are not geometry until local point or photo support passes.",
            "Automatic seeds are predictions; set require_reviewed_seed_ids for formal acceptance.",
        ],
    }


def recover_corridor_column_grid(
    project: ProjectConfig,
    ownership_plan_path: str | Path,
    vertical_report_paths: dict[str, str | Path],
    output_path: str | Path,
    *,
    side: str,
    settings_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    plan_path = project.resolve(ownership_plan_path)
    output = project.resolve(output_path)
    settings_source = project.resolve(settings_path) if settings_path else None
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    settings = load_json(settings_source) if settings_source else _resource_settings()
    if settings.get("schema_version") != "railway.corridor-column-grid-settings.v1":
        raise ValueError("Unsupported corridor-column-grid settings")
    reports: dict[str, dict[str, Any]] = {}
    resolved: dict[str, str] = {}
    for segment_id, raw_path in vertical_report_paths.items():
        path = project.resolve(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        reports[segment_id] = load_json(path)
        resolved[segment_id] = str(path)
    result = recover_corridor_column_grid_data(
        load_json(plan_path), reports, side=side, settings=settings
    )
    result.update(
        {
            "project_id": project.project_id,
            "ownership_plan": str(plan_path),
            "vertical_reports": resolved,
            "settings_source": str(settings_source) if settings_source else "bundled_default",
        }
    )
    write_json(output, result)
    return result
