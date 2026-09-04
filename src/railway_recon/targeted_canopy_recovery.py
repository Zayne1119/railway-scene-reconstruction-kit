from __future__ import annotations

import json
import math
import os
from importlib import resources
from itertools import pairwise
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_closing, binary_dilation, label

from .algorithms.mesh import ObjWriter
from .camera import load_camera_rows
from .canopy_structure import _project_points, _surface_record
from .config import ProjectConfig
from .geometry import CorridorFrame
from .io import load_json, write_json
from .mesh_audit import audit_obj
from .vertical_conflict_photo_evidence import (
    _draw_cross,
    _permutation_matrix,
    _render_candidate_sheet,
    analyze_vertical_conflict_photo_evidence_data,
)
from .vertical_conflict_photo_evidence import (
    _project_markers as _conflict_project_markers,
)
from .vertical_conflict_photo_evidence import (
    _resource_settings as _photo_resource_settings,
)


def _resource_settings() -> dict[str, Any]:
    path = resources.files("railway_recon.resources").joinpath(
        "targeted-canopy-recovery.default.json"
    )
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def reviewed_canopy_seeds(
    vertical: dict[str, Any], review: dict[str, Any], minimum_confidence: float
) -> list[dict[str, Any]]:
    """Apply a review overlay without mutating the original semantic predictions."""
    by_id = {item["id"]: item for item in vertical["candidates"]}
    seeds: list[dict[str, Any]] = []
    for decision in review.get("decisions", []):
        if decision.get("reviewed_class") != "canopy_column":
            continue
        if float(decision.get("confidence", 0.0)) < minimum_confidence:
            continue
        candidate_id = str(decision["candidate_id"])
        if candidate_id not in by_id:
            raise ValueError(f"Reviewed candidate is absent from vertical report: {candidate_id}")
        seed = dict(by_id[candidate_id])
        seed["review_overlay"] = {
            "reviewed_class": "canopy_column",
            "confidence": float(decision["confidence"]),
            "evidence_level": decision.get("evidence_level"),
            "evidence_sheet": decision.get("evidence_sheet"),
        }
        seeds.append(seed)
    return sorted(seeds, key=lambda item: float(item["longitudinal_position_m"]))


def infer_column_grid(
    seeds: list[dict[str, Any]],
    longitudinal_range: list[float],
    nominal_spacing: float,
    spacing_minimum: float,
    spacing_maximum: float,
    seed_match_tolerance: float,
    phase_support_positions: list[float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not seeds:
        raise ValueError("At least one reviewed canopy-column seed is required")
    seed_s = np.asarray(
        [float(item["longitudinal_position_m"]) for item in seeds], dtype=np.float64
    )
    spacing_samples: list[float] = []
    for first in range(seed_s.size):
        for second in range(first + 1, seed_s.size):
            difference = abs(float(seed_s[second] - seed_s[first]))
            divisions = max(1, round(difference / nominal_spacing))
            candidate = difference / divisions
            if spacing_minimum <= candidate <= spacing_maximum:
                spacing_samples.append(candidate)
    fit_basis = "reviewed_column_seed_pairs"
    if not spacing_samples and len(seeds) == 1:
        support = np.asarray(sorted(phase_support_positions or []), dtype=np.float64)
        if support.size < 2:
            raise ValueError(
                "One reviewed canopy-column seed requires at least two periodic "
                "platform-gap phase supports"
            )
        support_differences = np.diff(support)
        spacing_samples = [
            float(value)
            for value in support_differences
            if spacing_minimum <= float(value) <= spacing_maximum
        ]
        fit_basis = "single_reviewed_seed_plus_periodic_platform_gap_support"
    if not spacing_samples:
        raise ValueError("Available evidence does not support a plausible column spacing")
    initial_spacing = float(np.median(spacing_samples))
    if seed_s.size == 1:
        integers = np.zeros(1, dtype=np.int64)
        phase = float(seed_s[0])
        spacing = initial_spacing
        residuals = np.zeros(1, dtype=np.float64)
    else:
        anchor_index = int(np.argmin(np.abs(seed_s - np.median(seed_s))))
        integers = np.rint((seed_s - seed_s[anchor_index]) / initial_spacing).astype(
            np.int64
        )
        design = np.column_stack((np.ones(seed_s.size), integers.astype(np.float64)))
        phase, spacing = np.linalg.lstsq(design, seed_s, rcond=None)[0]
        spacing = float(spacing)
        phase = float(phase)
        residuals = seed_s - (phase + spacing * integers)
    if not spacing_minimum <= spacing <= spacing_maximum:
        raise ValueError("Refined canopy-column spacing is outside the accepted range")

    start, end = (float(value) for value in longitudinal_range)
    first_index = math.ceil((start - phase) / spacing)
    last_index = math.floor((end - phase) / spacing)
    default_cross = float(np.median([item["cross_position_m"] for item in seeds]))
    default_bottom = float(np.median([item["minimum_z"] for item in seeds]))
    default_top = float(np.median([item["maximum_z"] for item in seeds]))
    default_footprint = float(np.median([item["footprint_m"] for item in seeds]))
    grid: list[dict[str, Any]] = []
    for grid_index in range(first_index, last_index + 1):
        predicted_s = phase + grid_index * spacing
        distances = np.abs(seed_s - predicted_s)
        closest = int(np.argmin(distances))
        matched = float(distances[closest]) <= seed_match_tolerance
        seed = seeds[closest] if matched else None
        grid.append(
            {
                "id": f"RIGHT-CANOPY-GRID-{len(grid) + 1:03d}",
                "grid_index": int(grid_index),
                "longitudinal_position_m": (
                    float(seed["longitudinal_position_m"]) if seed else predicted_s
                ),
                "predicted_longitudinal_position_m": predicted_s,
                "cross_position_m": float(seed["cross_position_m"]) if seed else default_cross,
                "minimum_z": float(seed["minimum_z"]) if seed else default_bottom,
                "maximum_z": float(seed["maximum_z"]) if seed else default_top,
                "footprint_m": float(seed["footprint_m"]) if seed else default_footprint,
                "reviewed_seed_id": seed["id"] if seed else None,
                "evidence_level": "photo_interpreted_seed" if seed else "grid_inferred_pending_support",
            }
        )
    return grid, {
        "fit_basis": fit_basis,
        "nominal_spacing_m": nominal_spacing,
        "spacing_samples_m": spacing_samples,
        "refined_spacing_m": spacing,
        "phase_m": phase,
        "seed_grid_residuals_m": [float(value) for value in residuals],
        "maximum_absolute_seed_grid_residual_m": float(np.max(np.abs(residuals))),
    }


def _platform_elevation(component: dict[str, Any], longitudinal: float, cross: float) -> float:
    fits = [
        fit
        for fit in component["fit_segments"]
        if float(fit["longitudinal_range_m"][0]) - 1e-8
        <= longitudinal
        <= float(fit["longitudinal_range_m"][1]) + 1e-8
    ]
    if not fits:
        fits = [
            min(
                component["fit_segments"],
                key=lambda item: abs(
                    longitudinal
                    - sum(float(value) for value in item["longitudinal_range_m"]) / 2.0
                ),
            )
        ]
    values = []
    for fit in fits:
        a, b, d = (float(value) for value in fit["plane_z_equals_a_s_plus_b_c_plus_d"])
        values.append(a * longitudinal + b * cross + d)
    return float(np.mean(values))


def measure_column_support(
    grid: list[dict[str, Any]],
    component: dict[str, Any],
    s: np.ndarray,
    c: np.ndarray,
    z: np.ndarray,
    settings: dict[str, Any],
) -> None:
    radius = float(settings["column_support_radius_m"])
    bin_size = float(settings["column_support_vertical_bin_m"])
    minimum_ratio = float(settings["column_support_minimum_occupied_ratio"])
    minimum_points = int(settings["column_support_minimum_points"])
    for item in grid:
        longitudinal = float(item["longitudinal_position_m"])
        cross = float(item["cross_position_m"])
        bottom = _platform_elevation(component, longitudinal, cross)
        top = float(item["maximum_z"])
        mask = (
            ((s - longitudinal) ** 2 + (c - cross) ** 2 <= radius**2)
            & (z >= bottom + float(settings["column_support_bottom_clearance_m"]))
            & (z <= top + float(settings["column_support_top_tolerance_m"]))
        )
        values = z[mask]
        bin_count = max(1, math.ceil((top - bottom) / bin_size))
        if values.size:
            indexes = np.clip(((values - bottom) / bin_size).astype(np.int64), 0, bin_count - 1)
            occupied = int(np.unique(indexes).size)
        else:
            occupied = 0
        ratio = occupied / bin_count
        supported = values.size >= minimum_points and ratio >= minimum_ratio
        item["minimum_z"] = bottom
        item["point_support"] = {
            "radius_m": radius,
            "point_count": int(values.size),
            "vertical_bin_count": bin_count,
            "occupied_vertical_bin_count": occupied,
            "occupied_vertical_ratio": ratio,
            "supported": supported,
        }
        if item["reviewed_seed_id"]:
            item["status"] = "confirmed_photo_seed"
        elif supported:
            item["evidence_level"] = "point_supported_grid_inference"
            item["status"] = "candidate_geometry_allowed_review_required"
        else:
            item["status"] = "grid_position_only_no_geometry"


def extract_target_roofs(
    seeds: list[dict[str, Any]],
    grid: list[dict[str, Any]],
    component: dict[str, Any],
    s: np.ndarray,
    c: np.ndarray,
    z: np.ndarray,
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[np.ndarray], dict[str, Any]]:
    grid_size = float(settings["roof_grid_m"])
    s_min = math.floor(float(component["longitudinal_range_m"][0]) / grid_size) * grid_size
    s_max = math.ceil(float(component["longitudinal_range_m"][1]) / grid_size) * grid_size
    c_min = math.floor(
        (float(component["cross_range_m"][0]) - float(settings["roof_cross_padding_m"]))
        / grid_size
    ) * grid_size
    c_max = math.ceil(
        (float(component["cross_range_m"][1]) + float(settings["roof_cross_padding_m"]))
        / grid_size
    ) * grid_size
    seed_top = np.asarray([float(item["maximum_z"]) for item in seeds], dtype=np.float64)
    z_min = float(np.min(seed_top)) - float(settings["roof_minimum_z_below_seed_top_m"])
    z_max = float(np.max(seed_top)) + float(settings["roof_maximum_z_above_seed_top_m"])
    valid = (
        (s >= s_min)
        & (s <= s_max)
        & (c >= c_min)
        & (c <= c_max)
        & (z >= z_min)
        & (z <= z_max)
    )
    source_indexes = np.flatnonzero(valid)
    s_bins = math.ceil((s_max - s_min) / grid_size) + 1
    c_bins = math.ceil((c_max - c_min) / grid_size) + 1
    counts = np.zeros((c_bins, s_bins), dtype=np.int64)
    if source_indexes.size:
        si = np.clip(((s[valid] - s_min) / grid_size).astype(np.int64), 0, s_bins - 1)
        ci = np.clip(((c[valid] - c_min) / grid_size).astype(np.int64), 0, c_bins - 1)
        np.add.at(counts, (ci, si), 1)
    else:
        si = np.asarray([], dtype=np.int64)
        ci = np.asarray([], dtype=np.int64)
    occupied = counts >= int(settings["roof_minimum_cell_points"])
    connected = binary_dilation(
        binary_closing(occupied, iterations=int(settings["roof_binary_closing_iterations"])),
        iterations=int(settings["roof_binary_dilation_iterations"]),
    )
    labels, component_count = label(connected)
    seed_radius = int(settings["roof_seed_cell_radius"])
    seeded_labels: set[int] = set()
    for item in grid:
        grid_s = float(item["predicted_longitudinal_position_m"])
        grid_c = float(item["cross_position_m"])
        column_si = int(np.clip((grid_s - s_min) / grid_size, 0, s_bins - 1))
        column_ci = int(np.clip((grid_c - c_min) / grid_size, 0, c_bins - 1))
        window = labels[
            max(0, column_ci - seed_radius) : min(c_bins, column_ci + seed_radius + 1),
            max(0, column_si - seed_radius) : min(s_bins, column_si + seed_radius + 1),
        ]
        seeded_labels.update(int(value) for value in np.unique(window) if value)

    records: list[dict[str, Any]] = []
    indexes_by_component: list[np.ndarray] = []
    for label_value in sorted(seeded_labels):
        cells = labels == label_value
        indexes = source_indexes[labels[ci, si] == label_value]
        if indexes.size < int(settings["roof_minimum_component_points"]):
            continue
        record = _surface_record(
            f"RIGHT-CANOPY-ROOF-CANDIDATE-{len(records) + 1:03d}",
            "RIGHT-PLATFORM-REVIEWED",
            indexes,
            cells,
            s,
            c,
            z,
            s_min,
            c_min,
            grid_size,
            settings,
        )
        for profile in record["footprint_profiles"]:
            low, high = (float(value) for value in profile["longitudinal_range_m"])
            mask = (s[indexes] >= low) & (s[indexes] < high)
            profile_z = z[indexes][mask]
            profile_c = c[indexes][mask]
            if profile_z.size >= int(settings["minimum_profile_points"]):
                design = np.column_stack((profile_c, np.ones(profile_c.size)))
                slope, intercept = np.linalg.lstsq(design, profile_z, rcond=None)[0]
                residual = np.abs(profile_z - design @ np.asarray([slope, intercept]))
                profile["z_equals_a_cross_plus_d"] = [float(slope), float(intercept)]
                profile["absolute_residual_p90_m"] = float(np.percentile(residual, 90.0))
                profile["z_p50_m"] = float(np.median(profile_z))
        records.append(record)
        indexes_by_component.append(indexes)
    return records, indexes_by_component, {
        "search_envelope": {
            "longitudinal_range_m": [s_min, s_max],
            "cross_range_m": [c_min, c_max],
            "z_range_m": [z_min, z_max],
        },
        "source_point_count": int(source_indexes.size),
        "occupied_cell_count": int(np.count_nonzero(occupied)),
        "connected_component_count": int(component_count),
        "seeded_component_count": len(seeded_labels),
    }


def split_roof_surfaces(
    component: dict[str, Any],
    indexes: np.ndarray,
    s: np.ndarray,
    c: np.ndarray,
    z: np.ndarray,
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    """Split a connected plan component at unsupported or stepped cross bands."""
    cross_bin = float(settings["roof_surface_cross_bin_m"])
    minimum_points = int(settings["roof_surface_cross_bin_minimum_points"])
    jump = float(settings["roof_surface_median_jump_m"])
    minimum_width = float(settings["roof_surface_minimum_width_m"])
    component_s = s[indexes]
    component_c = c[indexes]
    component_z = z[indexes]
    cell_size = float(settings["roof_grid_m"])
    cell_s = np.floor(component_s / cell_size).astype(np.int64)
    cell_c = np.floor(component_c / cell_size).astype(np.int64)
    pairs = np.column_stack((cell_s, cell_c))
    _, inverse = np.unique(pairs, axis=0, return_inverse=True)
    representative_s: list[float] = []
    representative_c: list[float] = []
    representative_z: list[float] = []
    percentile = float(settings["roof_surface_cell_z_percentile"])
    for cell_index in range(int(inverse.max()) + 1):
        mask = inverse == cell_index
        representative_s.append(float(np.median(component_s[mask])))
        representative_c.append(float(np.median(component_c[mask])))
        representative_z.append(float(np.percentile(component_z[mask], percentile)))
    representative_s_array = np.asarray(representative_s, dtype=np.float64)
    representative_c_array = np.asarray(representative_c, dtype=np.float64)
    representative_z_array = np.asarray(representative_z, dtype=np.float64)
    start = math.floor(float(component_c.min()) / cross_bin) * cross_bin
    end = math.ceil(float(component_c.max()) / cross_bin) * cross_bin
    bands: list[dict[str, float]] = []
    for low in np.arange(start, end, cross_bin):
        mask = (component_c >= low) & (component_c < low + cross_bin)
        count = int(np.count_nonzero(mask))
        if count < minimum_points:
            continue
        bands.append(
            {
                "low": float(low),
                "high": float(low + cross_bin),
                "median_z": float(np.median(component_z[mask])),
                "point_count": float(count),
            }
        )
    groups: list[list[dict[str, float]]] = []
    for band in bands:
        starts_new = not groups
        if groups:
            previous = groups[-1][-1]
            starts_new = (
                band["low"] > previous["high"] + 1e-8
                or abs(band["median_z"] - previous["median_z"]) > jump
            )
        if starts_new:
            groups.append([band])
        else:
            groups[-1].append(band)

    surfaces: list[dict[str, Any]] = []
    profile_bin = float(settings["outline_profile_bin_m"])
    for group in groups:
        group_low = group[0]["low"]
        group_high = group[-1]["high"]
        if group_high - group_low < minimum_width:
            continue
        raw_surface_mask = (component_c >= group_low) & (component_c < group_high)
        raw_surface_point_count = int(np.count_nonzero(raw_surface_mask))
        representative_mask = (
            (representative_c_array >= group_low)
            & (representative_c_array < group_high)
        )
        surface_s = representative_s_array[representative_mask]
        surface_c = representative_c_array[representative_mask]
        surface_z = representative_z_array[representative_mask]
        if not surface_s.size:
            continue
        profiles: list[dict[str, Any]] = []
        profile_start = math.floor(float(surface_s.min()) / profile_bin) * profile_bin
        profile_end = math.ceil(float(surface_s.max()) / profile_bin) * profile_bin
        for low_s in np.arange(profile_start, profile_end, profile_bin):
            mask = (surface_s >= low_s) & (surface_s < low_s + profile_bin)
            if int(np.count_nonzero(mask)) < int(settings["roof_surface_profile_minimum_cells"]):
                continue
            values_c = surface_c[mask]
            values_z = surface_z[mask]
            design = np.column_stack((values_c, np.ones(values_c.size)))
            slope, intercept = np.linalg.lstsq(design, values_z, rcond=None)[0]
            residual = np.abs(values_z - design @ np.asarray([slope, intercept]))
            profiles.append(
                {
                    "longitudinal_range_m": [float(low_s), float(low_s + profile_bin)],
                    "cross_range_m": [
                        float(np.percentile(values_c, 1.0)),
                        float(np.percentile(values_c, 99.0)),
                    ],
                    "point_count": int(values_c.size),
                    "z_equals_a_cross_plus_d": [float(slope), float(intercept)],
                    "absolute_residual_p90_m": float(np.percentile(residual, 90.0)),
                    "z_p50_m": float(np.median(values_z)),
                }
            )
        design = np.column_stack((surface_s, surface_c, np.ones(surface_s.size)))
        coefficients = np.linalg.lstsq(design, surface_z, rcond=None)[0]
        residual = np.abs(surface_z - design @ coefficients)
        residual_p90 = float(np.percentile(residual, 90.0))
        if residual_p90 <= float(settings["planar_residual_p90_maximum_m"]):
            surface_type = "planar"
        elif residual_p90 <= float(settings["shallow_curve_residual_p90_maximum_m"]):
            surface_type = "piecewise_planar_or_shallow_curve"
        else:
            surface_type = "curved_or_multi_surface_review_required"
        surfaces.append(
            {
                "id": f"{component['id']}-SURFACE-{len(surfaces) + 1:02d}",
                "point_count": raw_surface_point_count,
                "representative_cell_count": int(surface_s.size),
                "cross_range_m": [group_low, group_high],
                "longitudinal_range_m": [float(surface_s.min()), float(surface_s.max())],
                "z_range_m": [float(surface_z.min()), float(surface_z.max())],
                "profile_count": len(profiles),
                "footprint_profiles": profiles,
                "plane_diagnostic": {
                    "z_equals_a_s_plus_b_c_plus_d": [float(value) for value in coefficients],
                    "absolute_residual_p90_m": residual_p90,
                },
                "surface_type": surface_type,
                "status": "observed_cross_band_candidate_review_required",
                "surface_sampling": f"cell_{percentile:g}th_percentile_top_envelope",
            }
        )
    return surfaces


def _profile_transition_deltas(
    previous: dict[str, Any], current: dict[str, Any]
) -> tuple[float, float]:
    previous_low_c, previous_high_c = (
        float(value) for value in previous["cross_range_m"]
    )
    current_low_c, current_high_c = (
        float(value) for value in current["cross_range_m"]
    )
    previous_slope, previous_intercept = (
        float(value) for value in previous["z_equals_a_cross_plus_d"]
    )
    current_slope, current_intercept = (
        float(value) for value in current["z_equals_a_cross_plus_d"]
    )
    previous_heights = (
        previous_slope * previous_low_c + previous_intercept,
        previous_slope * previous_high_c + previous_intercept,
    )
    current_heights = (
        current_slope * current_low_c + current_intercept,
        current_slope * current_high_c + current_intercept,
    )
    cross_shift = max(
        abs(current_low_c - previous_low_c),
        abs(current_high_c - previous_high_c),
    )
    height_shift = max(
        abs(current_heights[0] - previous_heights[0]),
        abs(current_heights[1] - previous_heights[1]),
    )
    return cross_shift, height_shift


def _profile_runs(
    component: dict[str, Any],
    tolerance: float = 1e-6,
    maximum_cross_shift_m: float = 0.75,
    maximum_height_shift_m: float = 0.25,
) -> list[list[dict[str, Any]]]:
    profiles = sorted(
        (
            item
            for item in component["footprint_profiles"]
            if "z_equals_a_cross_plus_d" in item
        ),
        key=lambda item: float(item["longitudinal_range_m"][0]),
    )
    runs: list[list[dict[str, Any]]] = []
    for profile in profiles:
        previous = runs[-1][-1] if runs else None
        separated = previous is None or float(profile["longitudinal_range_m"][0]) > float(
            previous["longitudinal_range_m"][1]
        ) + tolerance
        unsafe_transition = False
        if previous is not None and not separated:
            cross_shift, height_shift = _profile_transition_deltas(previous, profile)
            unsafe_transition = (
                cross_shift > maximum_cross_shift_m
                or height_shift > maximum_height_shift_m
            )
        if separated or unsafe_transition:
            runs.append([profile])
        else:
            runs[-1].append(profile)
    return [run for run in runs if len(run) >= 2]


def audit_roof_profile_mesh_continuity(
    surfaces: list[dict[str, Any]],
    maximum_cross_shift_m: float = 0.75,
    maximum_height_shift_m: float = 0.25,
) -> dict[str, Any]:
    surface_records: list[dict[str, Any]] = []
    total_unsafe = 0
    total_discarded = 0
    for surface in surfaces:
        profiles = sorted(
            (
                item
                for item in surface["footprint_profiles"]
                if "z_equals_a_cross_plus_d" in item
            ),
            key=lambda item: float(item["longitudinal_range_m"][0]),
        )
        unsafe = 0
        maximum_cross_shift = 0.0
        maximum_height_shift = 0.0
        for previous, current in pairwise(profiles):
            if float(current["longitudinal_range_m"][0]) > float(
                previous["longitudinal_range_m"][1]
            ) + 1e-6:
                continue
            cross_shift, height_shift = _profile_transition_deltas(previous, current)
            maximum_cross_shift = max(maximum_cross_shift, cross_shift)
            maximum_height_shift = max(maximum_height_shift, height_shift)
            unsafe += int(
                cross_shift > maximum_cross_shift_m
                or height_shift > maximum_height_shift_m
            )
        runs = _profile_runs(
            surface,
            maximum_cross_shift_m=maximum_cross_shift_m,
            maximum_height_shift_m=maximum_height_shift_m,
        )
        emitted_profiles = sum(len(run) for run in runs)
        discarded = len(profiles) - emitted_profiles
        total_unsafe += unsafe
        total_discarded += discarded
        surface_records.append(
            {
                "surface_id": surface["id"],
                "source_profile_count": len(profiles),
                "emitted_run_count": len(runs),
                "emitted_profile_count": emitted_profiles,
                "discarded_isolated_profile_count": discarded,
                "unsafe_transition_split_count": unsafe,
                "maximum_cross_endpoint_shift_m": maximum_cross_shift,
                "maximum_height_endpoint_shift_m": maximum_height_shift,
            }
        )
    return {
        "schema_version": "railway.targeted-canopy-profile-mesh-audit.v1",
        "maximum_allowed_cross_endpoint_shift_m": maximum_cross_shift_m,
        "maximum_allowed_height_endpoint_shift_m": maximum_height_shift_m,
        "unsafe_transition_split_count": total_unsafe,
        "discarded_isolated_profile_count": total_discarded,
        "surface_records": surface_records,
        "passed": bool(surfaces),
        "status": (
            "pass_discontinuous_transitions_split_before_meshing"
            if surfaces
            else "fail_no_observed_roof_surfaces"
        ),
    }


def _mesh_roof_run(
    run: list[dict[str, Any]], frame: CorridorFrame, thickness: float
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    stations: list[tuple[float, float, float, float, float]] = []
    for index, profile in enumerate(run):
        low_s, high_s = (float(value) for value in profile["longitudinal_range_m"])
        low_c, high_c = (float(value) for value in profile["cross_range_m"])
        slope, intercept = (float(value) for value in profile["z_equals_a_cross_plus_d"])
        if index == 0:
            stations.append((low_s, low_c, high_c, slope * low_c + intercept, slope * high_c + intercept))
        stations.append((high_s, low_c, high_c, slope * low_c + intercept, slope * high_c + intercept))
    top: list[list[float]] = []
    for longitudinal, low_c, high_c, low_z, high_z in stations:
        xy = frame.world_xy(
            np.asarray([longitudinal, longitudinal]), np.asarray([low_c, high_c])
        )
        top.extend(
            ([float(xy[0, 0]), float(xy[0, 1]), low_z], [float(xy[1, 0]), float(xy[1, 1]), high_z])
        )
    top_array = np.asarray(top, dtype=np.float64)
    bottom = top_array.copy()
    bottom[:, 2] -= thickness
    vertices = np.vstack((top_array, bottom))
    offset = len(top_array)
    faces: list[tuple[int, ...]] = []
    for index in range(len(stations) - 1):
        lo_a, hi_a, lo_b, hi_b = 2 * index, 2 * index + 1, 2 * index + 2, 2 * index + 3
        faces.extend(
            (
                (lo_a, lo_b, hi_b, hi_a),
                (offset + lo_a, offset + hi_a, offset + hi_b, offset + lo_b),
                (lo_a, offset + lo_a, offset + lo_b, lo_b),
                (hi_a, hi_b, offset + hi_b, offset + hi_a),
            )
        )
    final_lo, final_hi = len(top_array) - 2, len(top_array) - 1
    faces.extend(
        (
            (0, 1, offset + 1, offset),
            (final_lo, offset + final_lo, offset + final_hi, final_hi),
        )
    )
    return vertices, faces


def _column_box(
    item: dict[str, Any],
    frame: CorridorFrame,
    *,
    open_top: bool = False,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    half = max(float(item["footprint_m"]) / 2.0, 0.12)
    longitudinal = float(item["longitudinal_position_m"])
    cross = float(item["cross_position_m"])
    bottom = float(item["minimum_z"])
    top = float(item["maximum_z"])
    local = np.asarray(
        [
            [longitudinal + ds, cross + dc, height]
            for height in (bottom, top)
            for ds, dc in ((-half, -half), (half, -half), (half, half), (-half, half))
        ],
        dtype=np.float64,
    )
    xy = frame.world_xy(local[:, 0], local[:, 1])
    vertices = np.column_stack((xy, local[:, 2]))
    faces = [(0, 3, 2, 1)]
    if not open_top:
        faces.append((4, 5, 6, 7))
    faces.extend(
        ((0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
    )
    return vertices, faces


def _rectangular_box(
    longitudinal: float,
    cross: float,
    bottom: float,
    top: float,
    along_size: float,
    cross_size: float,
    frame: CorridorFrame,
    *,
    open_bottom: bool = False,
    open_top: bool = False,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    local = np.asarray(
        [
            [longitudinal + ds, cross + dc, height]
            for height in (bottom, top)
            for ds, dc in (
                (-along_size / 2.0, -cross_size / 2.0),
                (along_size / 2.0, -cross_size / 2.0),
                (along_size / 2.0, cross_size / 2.0),
                (-along_size / 2.0, cross_size / 2.0),
            )
        ],
        dtype=np.float64,
    )
    xy = frame.world_xy(local[:, 0], local[:, 1])
    vertices = np.column_stack((xy, local[:, 2]))
    faces: list[tuple[int, ...]] = []
    if not open_bottom:
        faces.append((0, 3, 2, 1))
    if not open_top:
        faces.append((4, 5, 6, 7))
    faces.extend(
        ((0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
    )
    return vertices, faces


def _local_underroof_mesh(
    node: dict[str, Any], frame: CorridorFrame
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    longitudinal = float(node["longitudinal_position_m"])
    half_along = float(node["along_half_extent_m"])
    sections = node["cross_sections"]
    local: list[list[float]] = []
    for section in sections:
        cross = float(section["cross_m"])
        for height in (float(section["bottom_z_m"]), float(section["top_z_m"])):
            local.extend(
                (
                    [longitudinal - half_along, cross, height],
                    [longitudinal + half_along, cross, height],
                )
            )
    local_array = np.asarray(local, dtype=np.float64)
    xy = frame.world_xy(local_array[:, 0], local_array[:, 1])
    vertices = np.column_stack((xy, local_array[:, 2]))
    faces: list[tuple[int, ...]] = []
    # Per section: bottom-left, bottom-right, top-left, top-right.
    for index in range(len(sections) - 1):
        first = 4 * index
        second = 4 * (index + 1)
        faces.extend(
            (
                (first + 2, first + 3, second + 3, second + 2),
                (first, second, second + 1, first + 1),
                (first, first + 2, second + 2, second),
                (first + 1, second + 1, second + 3, first + 3),
            )
        )
    # Keep the free cross end closed, but leave the observed-roof end open.
    # The observed roof already owns that boundary face; emitting it twice
    # creates a coplanar internal surface and can flicker in UE.
    last = 4 * (len(sections) - 1)
    roof_cross = float(node["nearest_observed_roof_cross_m"])
    first_is_roof = abs(float(sections[0]["cross_m"]) - roof_cross) <= 1e-6
    last_is_roof = abs(float(sections[-1]["cross_m"]) - roof_cross) <= 1e-6
    if not first_is_roof:
        faces.append((0, 1, 3, 2))
    if not last_is_roof:
        faces.append((last, last + 2, last + 3, last + 1))
    return vertices, faces


def _write_materials(path: Path) -> None:
    content = """# Targeted evidence-aware canopy candidate materials
newmtl CanopyRoofObserved
Kd 0.58 0.78 0.82
Ks 0.12 0.12 0.12
Ns 18

newmtl CanopyColumnPhotoConfirmed
Kd 0.18 0.82 0.95
Ks 0.08 0.08 0.08
Ns 12

newmtl CanopyColumnPointSupported
Kd 0.72 1.00 0.24
Ks 0.08 0.08 0.08
Ns 12

newmtl CanopyCapitalPhotoInterpreted
Kd 0.28 0.12 0.055
Ks 0.08 0.08 0.08
Ns 10

newmtl CanopyUnderroofLocalRecovery
Kd 0.52 0.34 0.76
Ks 0.10 0.10 0.10
Ns 14
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _render_diagnostic(report: dict[str, Any], output: Path) -> None:
    image = Image.new("RGB", (1700, 900), "#06141d")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((30, 25), "Targeted right-platform canopy recovery", fill="#eaf9ff", font=font)
    draw.text(
        (30, 50),
        f"reviewed seeds={report['reviewed_seed_count']} | grid={report['column_grid_count']} | "
        f"geometry columns={report['geometry_column_count']} | roof surfaces={report['roof_surface_count']}",
        fill="#b7ff3c",
        font=font,
    )
    panel = (55, 105, 1645, 780)
    draw.rectangle(panel, outline="#28505e", width=2)
    search = report["roof_search"]["search_envelope"]
    s_min, s_max = search["longitudinal_range_m"]
    c_min, c_max = search["cross_range_m"]

    def pixel(longitudinal: float, cross: float) -> tuple[int, int]:
        x = panel[0] + 25 + int((longitudinal - s_min) / max(s_max - s_min, 1e-9) * (panel[2] - panel[0] - 50))
        y = panel[3] - 25 - int((cross - c_min) / max(c_max - c_min, 1e-9) * (panel[3] - panel[1] - 50))
        return x, y

    palette = ("#244f59", "#35495f", "#38523d", "#59412d")
    for surface_index, surface in enumerate(report["roof_surfaces"]):
        polygon = []
        profiles = surface["footprint_profiles"]
        for profile in profiles:
            low, _ = profile["longitudinal_range_m"]
            cross_low, _ = profile["cross_range_m"]
            polygon.append(pixel(float(low), float(cross_low)))
        for profile in reversed(profiles):
            _, high = profile["longitudinal_range_m"]
            _, cross_high = profile["cross_range_m"]
            polygon.append(pixel(float(high), float(cross_high)))
        if len(polygon) >= 3:
            draw.polygon(
                polygon,
                fill=palette[surface_index % len(palette)],
                outline="#73e6ff",
            )
    for item in report["column_grid"]:
        point = pixel(float(item["longitudinal_position_m"]), float(item["cross_position_m"]))
        if item["status"] == "confirmed_photo_seed":
            color, radius = "#29d3ff", 7
        elif item["status"] == "candidate_geometry_allowed_review_required":
            color, radius = "#b7ff3c", 5
        else:
            color, radius = "#ff7a35", 4
        draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=color)
    draw.text(
        (55, 820),
        "cyan roof = point-cloud observed; blue columns = photo confirmed; lime = grid + point supported; orange = no geometry",
        fill="#8ca5ad",
        font=font,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def recover_targeted_canopy_data(
    vertical: dict[str, Any],
    review: dict[str, Any],
    platform: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    settings: dict[str, Any],
) -> tuple[dict[str, Any], list[np.ndarray]]:
    if review.get("next_gate") != "ready_for_targeted_right_platform_canopy_recovery":
        raise ValueError("Semantic review has not opened the targeted canopy recovery gate")
    components = [
        item for item in platform["platform_components"] if item.get("side") == settings["platform_side"]
    ]
    if len(components) != 1:
        raise ValueError("Targeted recovery requires exactly one reviewed platform component")
    component = components[0]
    seeds = reviewed_canopy_seeds(vertical, review, float(settings["minimum_review_confidence"]))
    phase_support_positions = [
        float(item["longitudinal_center_m"])
        for item in component.get("interior_gap_candidates", [])
        if item.get("gap_classification")
        == "periodic_occlusion_pattern_not_opening_candidate"
    ]
    grid, grid_fit = infer_column_grid(
        seeds,
        component["longitudinal_range_m"],
        float(settings["nominal_column_spacing_m"]),
        float(settings["minimum_column_spacing_m"]),
        float(settings["maximum_column_spacing_m"]),
        float(settings["seed_grid_match_tolerance_m"]),
        phase_support_positions=phase_support_positions,
    )
    s, c = _project_points(x, y, vertical["frame"])
    measure_column_support(grid, component, s, c, z, settings)
    roofs, roof_indexes, roof_search = extract_target_roofs(
        seeds, grid, component, s, c, z, settings
    )
    surfaces: list[dict[str, Any]] = []
    for roof, indexes in zip(roofs, roof_indexes, strict=True):
        surfaces.extend(split_roof_surfaces(roof, indexes, s, c, z, settings))
    geometry_columns = [item for item in grid if item["status"] != "grid_position_only_no_geometry"]
    return {
        "schema_version": "railway.targeted-canopy-recovery.v1",
        "project_id": vertical.get("project_id"),
        "segment_id": vertical["segment_id"],
        "source": vertical["source"],
        "frame": vertical["frame"],
        "platform_component_id": component["id"],
        "reviewed_seed_count": len(seeds),
        "reviewed_seed_ids": [item["id"] for item in seeds],
        "column_grid_fit": grid_fit,
        "column_grid_count": len(grid),
        "geometry_column_count": len(geometry_columns),
        "point_supported_inferred_column_count": sum(
            item["status"] == "candidate_geometry_allowed_review_required" for item in grid
        ),
        "unsupported_grid_position_count": sum(
            item["status"] == "grid_position_only_no_geometry" for item in grid
        ),
        "column_grid": grid,
        "roof_component_count": len(roofs),
        "roof_components": roofs,
        "roof_surface_count": len(surfaces),
        "roof_surfaces": surfaces,
        "roof_profile_mesh_continuity_audit": audit_roof_profile_mesh_continuity(
            surfaces
        ),
        "roof_search": roof_search,
        "status": (
            "targeted_canopy_candidate_geometry_ready_for_review"
            if surfaces and geometry_columns
            else "insufficient_targeted_canopy_evidence"
        ),
        "evidence_policy": {
            "original_semantic_predictions_mutated": False,
            "unsupported_grid_positions_written_as_geometry": False,
            "roof_forced_across_unobserved_intervals": False,
            "asset_registry_write": False,
        },
        "limitations": [
            "Grid inference supplies expected column positions but geometry is emitted only for reviewed or point-supported positions.",
            "Roof geometry follows point-cloud-observed profile runs and is not a design or clearance model.",
            "This candidate is independent from the formal asset registry until a separate approval gate.",
        ],
        "settings": settings,
    }, roof_indexes


def apply_grid_review(report: dict[str, Any], review: dict[str, Any]) -> None:
    if review.get("schema_version") != "railway.targeted-canopy-grid-review.v1":
        raise ValueError("Unsupported targeted canopy grid review schema")
    if review.get("segment_id") != report.get("segment_id"):
        raise ValueError("Targeted canopy grid review belongs to another segment")
    decision = review.get("decision")
    if decision != "reject_constant_spacing_grid":
        raise ValueError(f"Unsupported targeted canopy grid decision: {decision}")
    for item in report["column_grid"]:
        if item.get("reviewed_seed_id") is None:
            item["status_before_grid_review"] = item["status"]
            item["status"] = "grid_position_rejected_by_photo_review_no_geometry"
    report["geometry_column_count"] = sum(
        item["status"] == "confirmed_photo_seed" for item in report["column_grid"]
    )
    report["point_supported_inferred_column_count"] = 0
    report["unsupported_grid_position_count"] = sum(
        item.get("reviewed_seed_id") is None for item in report["column_grid"]
    )
    report["column_grid_review"] = review
    report["evidence_policy"]["rejected_grid_positions_written_as_geometry"] = False
    report["status"] = "targeted_canopy_roof_and_confirmed_seed_geometry_ready_for_review"


def audit_confirmed_column_roof_interfaces(report: dict[str, Any]) -> dict[str, Any]:
    tolerance = float(report["settings"].get("column_roof_interface_tolerance_m", 0.15))
    thickness = float(report["settings"]["roof_candidate_thickness_m"])
    results: list[dict[str, Any]] = []
    for column in report["column_grid"]:
        if column["status"] != "confirmed_photo_seed":
            continue
        longitudinal = float(column["longitudinal_position_m"])
        cross = float(column["cross_position_m"])
        options: list[tuple[float, dict[str, Any], dict[str, Any], float]] = []
        for surface in report["roof_surfaces"]:
            profiles = [
                profile
                for profile in surface["footprint_profiles"]
                if float(profile["longitudinal_range_m"][0]) - 1e-8
                <= longitudinal
                <= float(profile["longitudinal_range_m"][1]) + 1e-8
            ]
            for profile in profiles:
                low, high = (float(value) for value in profile["cross_range_m"])
                nearest_cross = min(max(cross, low), high)
                cross_distance = abs(cross - nearest_cross)
                slope, intercept = (
                    float(value) for value in profile["z_equals_a_cross_plus_d"]
                )
                top_z = slope * nearest_cross + intercept
                options.append((cross_distance, surface, profile, top_z))
        if not options:
            results.append(
                {
                    "column_grid_id": column["id"],
                    "reviewed_seed_id": column["reviewed_seed_id"],
                    "status": "no_roof_profile_at_longitudinal_position",
                }
            )
            continue
        cross_distance, surface, profile, roof_top = min(options, key=lambda value: value[0])
        roof_underside = roof_top - thickness
        vertical_gap = roof_underside - float(column["maximum_z"])
        covered = cross_distance <= 1e-8
        passed = covered and abs(vertical_gap) <= tolerance
        results.append(
            {
                "column_grid_id": column["id"],
                "reviewed_seed_id": column["reviewed_seed_id"],
                "nearest_roof_surface_id": surface["id"],
                "roof_profile_longitudinal_range_m": profile["longitudinal_range_m"],
                "nearest_roof_cross_distance_m": cross_distance,
                "column_top_z_m": float(column["maximum_z"]),
                "nearest_roof_top_z_m": roof_top,
                "assumed_roof_underside_z_m": roof_underside,
                "vertical_gap_to_assumed_underside_m": vertical_gap,
                "maximum_allowed_absolute_gap_m": tolerance,
                "status": (
                    "pass"
                    if passed
                    else (
                        "column_axis_outside_observed_roof_surface"
                        if not covered
                        else "column_roof_vertical_gap"
                    )
                ),
            }
        )
    unresolved = sum(item["status"] != "pass" for item in results)
    return {
        "confirmed_column_count": len(results),
        "pass_count": len(results) - unresolved,
        "unresolved_count": unresolved,
        "interfaces": results,
        "merge_gate": "passed" if unresolved == 0 else "blocked",
        "required_action": (
            "none"
            if unresolved == 0
            else "recover_photo_interpreted_capital_or_underroof_geometry_without_extending_unobserved_roof"
        ),
    }


def recover_local_column_roof_nodes(report: dict[str, Any]) -> list[dict[str, Any]]:
    settings = report["settings"]
    maximum_cross_distance = float(settings["local_node_maximum_cross_distance_m"])
    capital_height = float(settings["capital_height_m"])
    scale = float(settings["capital_footprint_scale"])
    minimum_size = float(settings["capital_minimum_footprint_m"])
    maximum_size = float(settings["capital_maximum_footprint_m"])
    minimum_top_thickness = float(settings["local_underroof_minimum_thickness_m"])
    nodes: list[dict[str, Any]] = []
    for column in report["column_grid"]:
        if column["status"] != "confirmed_photo_seed":
            continue
        longitudinal = float(column["longitudinal_position_m"])
        cross = float(column["cross_position_m"])
        column_top = float(column["maximum_z"])
        options: list[tuple[float, float, dict[str, Any], dict[str, Any]]] = []
        for surface in report["roof_surfaces"]:
            for profile in surface["footprint_profiles"]:
                low_s, high_s = (float(value) for value in profile["longitudinal_range_m"])
                if not low_s - 1e-8 <= longitudinal <= high_s + 1e-8:
                    continue
                low_c, high_c = (float(value) for value in profile["cross_range_m"])
                nearest_cross = min(max(cross, low_c), high_c)
                slope, intercept = (
                    float(value) for value in profile["z_equals_a_cross_plus_d"]
                )
                roof_top = slope * nearest_cross + intercept
                if roof_top < column_top + minimum_top_thickness * 0.5:
                    continue
                options.append(
                    (abs(nearest_cross - cross), nearest_cross, surface, profile)
                )
        if not options:
            continue
        cross_distance, roof_cross, surface, profile = min(options, key=lambda value: value[0])
        if cross_distance > maximum_cross_distance:
            continue
        slope, intercept = (
            float(value) for value in profile["z_equals_a_cross_plus_d"]
        )
        roof_top = slope * roof_cross + intercept
        capital_size = min(
            max(float(column["footprint_m"]) * scale, minimum_size), maximum_size
        )
        capital_half = capital_size / 2.0
        if roof_cross >= cross:
            sections = [
                {
                    "cross_m": cross - capital_half,
                    "bottom_z_m": column_top,
                    "top_z_m": column_top + minimum_top_thickness,
                },
                {
                    "cross_m": cross + capital_half,
                    "bottom_z_m": column_top,
                    "top_z_m": column_top + minimum_top_thickness,
                },
            ]
            if roof_cross > cross + capital_half + 1e-6:
                sections.append(
                    {
                        "cross_m": roof_cross,
                        "bottom_z_m": roof_top - minimum_top_thickness,
                        "top_z_m": roof_top,
                    }
                )
        else:
            sections = []
            if roof_cross < cross - capital_half - 1e-6:
                sections.append(
                    {
                        "cross_m": roof_cross,
                        "bottom_z_m": roof_top - minimum_top_thickness,
                        "top_z_m": roof_top,
                    }
                )
            sections.extend(
                (
                    {
                        "cross_m": cross - capital_half,
                        "bottom_z_m": column_top,
                        "top_z_m": column_top + minimum_top_thickness,
                    },
                    {
                        "cross_m": cross + capital_half,
                        "bottom_z_m": column_top,
                        "top_z_m": column_top + minimum_top_thickness,
                    },
                )
            )
        nodes.append(
            {
                "id": f"{column['id']}-LOCAL-NODE",
                "column_grid_id": column["id"],
                "reviewed_seed_id": column["reviewed_seed_id"],
                "longitudinal_position_m": longitudinal,
                "cross_position_m": cross,
                "column_top_z_m": column_top,
                "capital_bottom_z_m": column_top - capital_height,
                "capital_top_z_m": column_top,
                "capital_along_size_m": capital_size,
                "capital_cross_size_m": capital_size,
                "along_half_extent_m": float(settings["local_underroof_along_half_extent_m"]),
                "cross_sections": sections,
                "nearest_observed_roof_surface_id": surface["id"],
                "nearest_observed_roof_cross_m": roof_cross,
                "nearest_observed_roof_top_z_m": roof_top,
                "cross_recovery_distance_m": cross_distance,
                "evidence_level": "photo_interpreted_plus_observed_roof_boundary",
                "confidence": float(settings["local_node_candidate_confidence"]),
                "status": "local_node_candidate_visual_review_required",
                "limitations": [
                    "Capital existence and visible appearance are photo interpreted.",
                    "The underroof connector interpolates only to the nearest observed roof boundary.",
                    "No connector is extended longitudinally beyond the local confirmed column node.",
                ],
            }
        )
    return nodes


def audit_local_node_chains(report: dict[str, Any]) -> dict[str, Any]:
    nodes = report.get("local_column_roof_nodes", [])
    confirmed = [
        item for item in report["column_grid"] if item["status"] == "confirmed_photo_seed"
    ]
    node_ids = {item["column_grid_id"] for item in nodes}
    missing = [item["id"] for item in confirmed if item["id"] not in node_ids]
    invalid = [
        item["id"]
        for item in nodes
        if len(item["cross_sections"]) < 2
        or any(
            float(section["top_z_m"]) <= float(section["bottom_z_m"])
            for section in item["cross_sections"]
        )
    ]
    candidate_pass = not missing and not invalid and bool(nodes)
    return {
        "confirmed_column_count": len(confirmed),
        "local_node_count": len(nodes),
        "missing_column_node_ids": missing,
        "invalid_node_ids": invalid,
        "candidate_chain_gate": "passed" if candidate_pass else "blocked",
        "formal_merge_gate": "blocked_pending_fixed_view_visual_review",
        "contact_surface_policy": {
            "column_top_cap": "single_surface_capital_bottom_only",
            "capital_underroof": "single_surface_underroof_bottom_only",
            "underroof_observed_roof": "open_connector_end_uses_observed_roof_boundary",
            "coplanar_internal_contact_faces_emitted": False,
        },
        "status": "candidate_nodes_complete" if candidate_pass else "candidate_nodes_incomplete",
    }


def recover_targeted_canopy(
    project: ProjectConfig,
    segment_id: str,
    semantic_review_path: str | Path,
    overwrite: bool = False,
    vertical_report_path: str | Path | None = None,
    platform_report_path: str | Path | None = None,
    settings_path: str | Path | None = None,
    grid_review_path: str | Path | None = None,
) -> dict[str, Any]:
    reports = project.workspace_path("reports")
    vertical_path = project.resolve(vertical_report_path) if vertical_report_path else reports / f"{segment_id}_vertical_hypotheses.json"
    platform_path = project.resolve(platform_report_path) if platform_report_path else reports / f"{segment_id}_platform_surface.json"
    review_path = project.resolve(semantic_review_path)
    for path in (vertical_path, platform_path, review_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    vertical = load_json(vertical_path)
    platform = load_json(platform_path)
    review = load_json(review_path)
    configured = project.value.get("algorithms", {}).get("targeted_canopy_recovery")
    resolved_settings = project.resolve(settings_path) if settings_path else (project.resolve(configured) if configured else None)
    settings = load_json(resolved_settings) if resolved_settings else _resource_settings()

    source = Path(vertical["source"])
    if not source.is_file():
        raise FileNotFoundError(source)
    output_dir = project.workspace_path("exports") / segment_id / "targeted_canopy_candidate"
    obj_path = output_dir / "targeted_canopy_candidate.obj"
    mtl_path = output_dir / "targeted_canopy_candidate.mtl"
    origin_path = output_dir / "model_origin.json"
    report_path = reports / f"{segment_id}_targeted_canopy_recovery.json"
    audit_path = reports / f"{segment_id}_targeted_canopy_mesh_audit.json"
    diagnostic_path = reports / f"{segment_id}_targeted_canopy_recovery.png"
    for path in (obj_path, mtl_path, origin_path, report_path, audit_path, diagnostic_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite: {path}")

    cloud = laspy.read(source)
    report, _ = recover_targeted_canopy_data(
        vertical,
        review,
        platform,
        np.asarray(cloud.x, dtype=np.float64),
        np.asarray(cloud.y, dtype=np.float64),
        np.asarray(cloud.z, dtype=np.float64),
        settings,
    )
    resolved_grid_review = project.resolve(grid_review_path) if grid_review_path else None
    if resolved_grid_review:
        if not resolved_grid_review.is_file():
            raise FileNotFoundError(resolved_grid_review)
        apply_grid_review(report, load_json(resolved_grid_review))
    report["local_column_roof_nodes"] = recover_local_column_roof_nodes(report)
    report["confirmed_column_roof_interface_audit"] = audit_confirmed_column_roof_interfaces(
        report
    )
    report["local_node_chain_audit"] = audit_local_node_chains(report)
    if report["confirmed_column_roof_interface_audit"]["merge_gate"] == "blocked":
        report["status"] = "targeted_canopy_candidate_generated_observed_interface_gate_blocked"
    frame = CorridorFrame.from_json(vertical["frame"])
    all_world_xy = frame.world_xy(
        np.asarray([item["longitudinal_position_m"] for item in report["column_grid"]]),
        np.asarray([item["cross_position_m"] for item in report["column_grid"]]),
    )
    origin = np.asarray([float(np.mean(all_world_xy[:, 0])), float(np.mean(all_world_xy[:, 1])), 0.0])
    writer = ObjWriter(origin, material_library=mtl_path.name)
    for surface in report["roof_surfaces"]:
        for run_number, run in enumerate(_profile_runs(surface), start=1):
            vertices, faces = _mesh_roof_run(run, frame, float(settings["roof_candidate_thickness_m"]))
            writer.add_mesh(f"{surface['id']}-RUN-{run_number:02d}", vertices, faces, "CanopyRoofObserved")
    for item in report["column_grid"]:
        if item["status"] not in {
            "confirmed_photo_seed",
            "candidate_geometry_allowed_review_required",
        }:
            continue
        node = next(
            (
                candidate
                for candidate in report["local_column_roof_nodes"]
                if candidate["column_grid_id"] == item["id"]
            ),
            None,
        )
        shaft = dict(item)
        if node:
            shaft["maximum_z"] = node["capital_bottom_z_m"]
        vertices, faces = _column_box(shaft, frame, open_top=bool(node))
        material = "CanopyColumnPhotoConfirmed" if item["status"] == "confirmed_photo_seed" else "CanopyColumnPointSupported"
        writer.add_mesh(item["id"], vertices, faces, material)
        if node:
            capital_vertices, capital_faces = _rectangular_box(
                float(node["longitudinal_position_m"]),
                float(node["cross_position_m"]),
                float(node["capital_bottom_z_m"]),
                float(node["capital_top_z_m"]),
                float(node["capital_along_size_m"]),
                float(node["capital_cross_size_m"]),
                frame,
                open_top=True,
            )
            writer.add_mesh(
                f"{item['id']}-CAPITAL",
                capital_vertices,
                capital_faces,
                "CanopyCapitalPhotoInterpreted",
            )
            underroof_vertices, underroof_faces = _local_underroof_mesh(node, frame)
            writer.add_mesh(
                f"{item['id']}-LOCAL-UNDERROOF",
                underroof_vertices,
                underroof_faces,
                "CanopyUnderroofLocalRecovery",
            )
    if writer.face_count == 0:
        raise ValueError("No evidence-supported canopy candidate geometry was generated")
    writer.write(obj_path)
    _write_materials(mtl_path)
    write_json(origin_path, {"origin_xyz": origin.tolist(), "units": "metre", "axis": "Z-up"})
    audit = audit_obj(obj_path)
    audit["status"] = "pass" if audit["passed"] else "fail"
    write_json(audit_path, audit)
    if not audit["passed"]:
        raise ValueError("Generated targeted canopy OBJ failed mesh audit")

    report.update(
        {
            "semantic_review": str(review_path),
            "vertical_report": str(vertical_path),
            "platform_report": str(platform_path),
            "settings_source": str(resolved_settings) if resolved_settings else "bundled_default",
            "grid_review": str(resolved_grid_review) if resolved_grid_review else None,
            "output_obj": str(obj_path),
            "output_mtl": str(mtl_path),
            "output_origin": str(origin_path),
            "output_mesh_audit": str(audit_path),
            "output_diagnostic": str(diagnostic_path),
            "mesh_vertex_count": writer.vertex_count,
            "mesh_face_count": writer.face_count,
        }
    )
    write_json(report_path, report)
    _render_diagnostic(report, diagnostic_path)
    return report


def _render_grid_photo_summary(
    report: dict[str, Any], crops: dict[str, dict[int, np.ndarray]], output: Path
) -> None:
    candidates = report["candidates"]
    columns = 3
    tile_width = 640
    tile_height = 390
    rows = math.ceil(len(candidates) / columns)
    sheet = Image.new("RGB", (columns * tile_width, 75 + rows * tile_height), "#06141d")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((20, 15), "INFERRED RIGHT-CANOPY GRID / BEST PHOTO REVIEW VIEW", fill="#eaf9ff", font=font)
    draw.text((20, 40), "orange line is a grid hypothesis, not accepted geometry", fill="#ff9f31", font=font)
    for index, candidate in enumerate(candidates):
        reviewable = [
            item
            for item in candidate["views"]
            if item["trusted_for_consensus"] and item["reviewable_resolution"]
        ]
        options = reviewable or list(candidate["views"])
        best = min(options, key=lambda item: float(item["camera_distance_m"]))
        crop = Image.fromarray(crops[candidate["candidate_id"]][int(best["camera_index"])], mode="RGB")
        layer = ImageDraw.Draw(crop)
        markers = best["crop_markers"]
        layer.line((*markers["base"], *markers["top"]), fill="#ff9f31", width=5)
        _draw_cross(layer, markers["base"], "#b7ff3c")
        _draw_cross(layer, markers["top"], "#24d7ff")
        crop.thumbnail((tile_width - 20, tile_height - 62), Image.Resampling.LANCZOS)
        tile_x = (index % columns) * tile_width
        tile_y = 75 + (index // columns) * tile_height
        sheet.paste(crop, (tile_x + (tile_width - crop.width) // 2, tile_y + 42))
        draw.text(
            (tile_x + 10, tile_y + 8),
            f"{candidate['candidate_id']} | S {candidate['longitudinal_position_m']:.2f} | cam {best['camera_index']} | {best['camera_distance_m']:.1f} m",
            fill="#b7ff3c" if reviewable else "#9bb1b9",
            font=font,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=94)


def _sample_column_axis_score(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
    height, width = image.shape[:2]
    samples: list[np.ndarray] = []
    for px, py in zip(u, v, strict=True):
        x = round(float(px)) % width
        y = round(float(py))
        if y < 2 or y >= height - 2:
            continue
        xs = np.mod(np.arange(x - 2, x + 3), width)
        patch = image[y - 2 : y + 3][:, xs].astype(np.float64) / 255.0
        samples.append(np.median(patch.reshape(-1, 3), axis=0))
    if len(samples) < 6:
        return 0.0
    rgb = np.asarray(samples)
    maximum = rgb.max(axis=1)
    minimum = rgb.min(axis=1)
    saturation = (maximum - minimum) / np.maximum(maximum, 1e-6)
    luminance = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    neutral = np.clip(1.0 - saturation / 0.30, 0.0, 1.0)
    visible = np.clip((luminance - 0.12) / 0.25, 0.0, 1.0)
    not_sky_white = np.clip((0.98 - luminance) / 0.18, 0.0, 1.0)
    return float(np.mean(neutral * visible * not_sky_white))


def scan_photo_column_axis(
    recovery: dict[str, Any],
    consensus: dict[str, Any],
    camera_rows: list[dict[str, str]],
    image_loader: Any,
    half_width_m: float = 2.5,
    step_m: float = 0.1,
) -> dict[str, Any]:
    convention = consensus["best_shared_convention"]
    permutation = _permutation_matrix(str(convention["canonical_axes_from_local"]))
    rows = {int(row["index"]): row for row in camera_rows}
    trusted_cameras = [
        camera
        for camera in consensus["per_camera"]
        if bool(camera["local_best_matches_consensus"])
    ]
    image_arrays = {
        int(camera["camera_index"]): np.asarray(
            image_loader(Path(camera["photo"])).convert("RGB")
        )
        for camera in trusted_cameras
    }
    offsets = np.arange(-half_width_m, half_width_m + step_m / 2.0, step_m)
    results: list[dict[str, Any]] = []
    for item in recovery["column_grid"]:
        scores: list[float] = []
        per_offset_view_counts: list[int] = []
        for offset in offsets:
            bottom = float(item["minimum_z"]) + 0.45
            top = float(item["maximum_z"]) - 0.45
            local = np.column_stack(
                (
                    np.full(15, float(item["longitudinal_position_m"]) + float(offset)),
                    np.full(15, float(item["cross_position_m"])),
                    np.linspace(bottom, top, 15),
                )
            )
            view_scores: list[float] = []
            for camera in trusted_cameras:
                camera_index = int(camera["camera_index"])
                row = rows[camera_index]
                image = image_arrays[camera_index]
                camera_xyz = np.asarray(
                    [float(row[key]) for key in ("x", "y", "z")], dtype=np.float64
                )
                u, v, distance = _conflict_project_markers(
                    local,
                    recovery["frame"],
                    camera_xyz,
                    camera,
                    convention,
                    permutation,
                    image.shape[1],
                    image.shape[0],
                )
                projected_height = math.dist((float(u[0]), float(v[0])), (float(u[-1]), float(v[-1])))
                if distance > 55.0 or projected_height < 65.0:
                    continue
                view_scores.append(_sample_column_axis_score(image, u, v))
            per_offset_view_counts.append(len(view_scores))
            scores.append(float(np.median(view_scores)) if view_scores else 0.0)
        best_index = int(np.argmax(scores))
        ordered = np.sort(np.asarray(scores))
        baseline = float(np.percentile(ordered, 50.0))
        peak = float(scores[best_index])
        results.append(
            {
                "column_grid_id": item["id"],
                "reviewed_seed_id": item.get("reviewed_seed_id"),
                "original_longitudinal_position_m": float(item["longitudinal_position_m"]),
                "best_offset_m": float(offsets[best_index]),
                "best_longitudinal_position_m": float(item["longitudinal_position_m"] + offsets[best_index]),
                "best_score": peak,
                "median_scan_score": baseline,
                "peak_above_median": peak - baseline,
                "best_offset_trusted_view_count": per_offset_view_counts[best_index],
                "scan_offsets_m": [float(value) for value in offsets],
                "scan_scores": scores,
                "status": "appearance_scan_diagnostic_not_geometry",
            }
        )
    seed_offsets = [
        abs(float(item["best_offset_m"]))
        for item in results
        if item["reviewed_seed_id"] is not None
    ]
    calibrated = bool(seed_offsets) and float(np.median(seed_offsets)) <= 0.5
    return {
        "scan_half_width_m": half_width_m,
        "scan_step_m": step_m,
        "reviewed_seed_count": len(seed_offsets),
        "reviewed_seed_median_absolute_best_offset_m": (
            float(np.median(seed_offsets)) if seed_offsets else None
        ),
        "calibrated_on_reviewed_seeds": calibrated,
        "columns": results,
        "status": (
            "appearance_scan_supports_local_refinement_diagnostic_only"
            if calibrated
            else "appearance_scan_failed_seed_calibration_do_not_use"
        ),
    }


def targeted_canopy_photo_evidence(
    project: ProjectConfig,
    segment_id: str,
    recovery_report_path: str | Path,
    projection_consensus_path: str | Path,
    settings_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    recovery_path = project.resolve(recovery_report_path)
    consensus_path = project.resolve(projection_consensus_path)
    for path in (recovery_path, consensus_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    recovery = load_json(recovery_path)
    consensus = load_json(consensus_path)
    resolved_settings = project.resolve(settings_path) if settings_path else None
    settings = load_json(resolved_settings) if resolved_settings else _photo_resource_settings()
    inferred = [
        item for item in recovery["column_grid"] if item.get("reviewed_seed_id") is None
    ]
    pseudo_vertical = {
        "schema_version": "railway.vertical-hypotheses.v1",
        "project_id": recovery.get("project_id"),
        "segment_id": segment_id,
        "frame": recovery["frame"],
        "candidates": [
            {
                "id": item["id"],
                "predicted_class": "grid_inferred",
                "confidence": "candidate",
                "longitudinal_position_m": item["longitudinal_position_m"],
                "cross_position_m": item["cross_position_m"],
                "minimum_z": item["minimum_z"],
                "maximum_z": item["maximum_z"],
                "footprint_m": item["footprint_m"],
            }
            for item in inferred
        ],
    }
    pseudo_interface = {
        "project_id": recovery.get("project_id"),
        "segment_id": segment_id,
        "vertical_interfaces": [
            {"candidate_id": item["id"], "semantic_conflict": True} for item in inferred
        ],
    }
    cache: dict[Path, Image.Image] = {}

    def load_image(path: Path) -> Image.Image:
        if path not in cache:
            cache[path] = Image.open(path).convert("RGB")
        return cache[path]

    camera_rows = load_camera_rows(Path(consensus["camera_csv"]))
    report, crops = analyze_vertical_conflict_photo_evidence_data(
        pseudo_interface,
        pseudo_vertical,
        consensus,
        camera_rows,
        settings,
        load_image,
    )
    report["schema_version"] = "railway.targeted-canopy-grid-photo-evidence.v1"
    report["status"] = "grid_photo_review_generated_no_geometry_or_registry_write"
    report["recovery_report"] = str(recovery_path)
    report["projection_consensus_report"] = str(consensus_path)
    report["column_axis_appearance_scan"] = scan_photo_column_axis(
        recovery, consensus, camera_rows, load_image
    )
    reports = project.workspace_path("reports")
    output_report = reports / f"{segment_id}_targeted_canopy_grid_photo_evidence.json"
    output_summary = reports / f"{segment_id}_targeted_canopy_grid_photo_evidence.jpg"
    output_directory = reports / f"{segment_id}_targeted_canopy_grid_photo_evidence"
    for path in (output_report, output_summary):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite: {path}")
    for candidate in report["candidates"]:
        sheet_path = output_directory / f"{candidate['candidate_id']}.jpg"
        _render_candidate_sheet(candidate, crops[candidate["candidate_id"]], sheet_path, settings)
        candidate["evidence_sheet"] = str(sheet_path)
    report["output_summary"] = str(output_summary)
    report["output_evidence_directory"] = str(output_directory)
    report["settings_source"] = str(resolved_settings) if resolved_settings else "bundled_default"
    write_json(output_report, report)
    _render_grid_photo_summary(report, crops, output_summary)
    for image in cache.values():
        image.close()
    return report
