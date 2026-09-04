from __future__ import annotations

import json
import math
from importlib import resources
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, label

from .config import ProjectConfig
from .io import load_json, write_json


def _resource_settings() -> dict[str, Any]:
    path = resources.files("railway_recon.resources").joinpath("platform-surface.default.json")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _project_points(
    x: np.ndarray, y: np.ndarray, frame: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    delta = np.column_stack((x, y)) - origin
    return (
        delta @ np.asarray(frame["along_xy"], dtype=np.float64),
        delta @ np.asarray(frame["cross_xy"], dtype=np.float64),
    )


def _intervals(values: np.ndarray, grid: float) -> list[list[float]]:
    if not values.size:
        return []
    indexes = np.unique(np.floor(values / grid).astype(np.int64))
    groups: list[list[int]] = [[int(indexes[0])]]
    for index in indexes[1:]:
        if int(index) == groups[-1][-1] + 1:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])
    return [[group[0] * grid, (group[-1] + 1) * grid] for group in groups]


def _qualified_cells(
    s: np.ndarray,
    c: np.ndarray,
    z: np.ndarray,
    s_min: float,
    s_max: float,
    c_min: float,
    c_max: float,
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], np.ndarray, float, float, int, int]:
    grid = float(settings["surface_grid_m"])
    s_origin = math.floor(s_min / grid) * grid
    c_origin = math.floor(c_min / grid) * grid
    s_bins = math.ceil((s_max - s_origin) / grid) + 1
    c_bins = math.ceil((c_max - c_origin) / grid) + 1
    si = np.clip(((s - s_origin) / grid).astype(np.int64), 0, s_bins - 1)
    ci = np.clip(((c - c_origin) / grid).astype(np.int64), 0, c_bins - 1)
    cell_ids = ci * s_bins + si
    order = np.argsort(cell_ids, kind="stable")
    ordered_ids = cell_ids[order]
    unique, starts, counts = np.unique(ordered_ids, return_index=True, return_counts=True)
    occupied = np.zeros((c_bins, s_bins), dtype=bool)
    records: list[dict[str, Any]] = []
    minimum_points = int(settings["minimum_cell_points"])
    maximum_thickness = float(settings["maximum_cell_z_p90_p10_m"])
    for cell_id, start, count in zip(unique, starts, counts, strict=True):
        if int(count) < minimum_points:
            continue
        indexes = order[int(start) : int(start + count)]
        values = z[indexes]
        z10, z50, z90 = np.percentile(values, (10.0, 50.0, 90.0))
        if float(z90 - z10) > maximum_thickness:
            continue
        row, column = divmod(int(cell_id), s_bins)
        occupied[row, column] = True
        records.append(
            {
                "s_m": s_origin + (column + 0.5) * grid,
                "c_m": c_origin + (row + 0.5) * grid,
                "z_p10_m": float(z10),
                "z_p50_m": float(z50),
                "z_p90_m": float(z90),
                "point_count": int(count),
                "row": row,
                "column": column,
            }
        )
    return records, occupied, s_origin, c_origin, s_bins, c_bins


def _fit_segments(
    records: list[dict[str, Any]], side: str, settings: dict[str, Any]
) -> list[dict[str, Any]]:
    segment_size = float(settings["longitudinal_fit_segment_m"])
    s_values = np.asarray([item["s_m"] for item in records])
    c_values = np.asarray([item["c_m"] for item in records])
    z_values = np.asarray([item["z_p50_m"] for item in records])
    segment_indexes = np.floor(s_values / segment_size).astype(np.int64)
    result: list[dict[str, Any]] = []
    for segment_index in np.unique(segment_indexes):
        mask = segment_indexes == segment_index
        if int(mask.sum()) < int(settings["minimum_fit_segment_cells"]):
            continue
        source_count = int(mask.sum())
        segment_s = s_values[mask]
        segment_c = c_values[mask]
        segment_z = z_values[mask]
        minimum_cells = int(settings["minimum_fit_segment_cells"])
        elevation_gap = float(settings.get("fit_elevation_cluster_gap_m", 0.12))
        elevation_order = np.argsort(segment_z)
        split_locations = np.flatnonzero(np.diff(segment_z[elevation_order]) > elevation_gap) + 1
        elevation_groups = np.split(elevation_order, split_locations)
        dominant_group = max(elevation_groups, key=len)
        if len(dominant_group) >= minimum_cells and len(dominant_group) < source_count:
            segment_s = segment_s[dominant_group]
            segment_c = segment_c[dominant_group]
            segment_z = segment_z[dominant_group]
        fit_source_count = len(segment_z)
        design = np.column_stack((segment_s, segment_c, np.ones(fit_source_count)))
        inliers = np.ones(fit_source_count, dtype=bool)
        iterations = int(settings.get("robust_fit_iterations", 0))
        residual_floor = float(settings.get("robust_fit_residual_floor_m", 0.02))
        mad_multiplier = float(settings.get("robust_fit_mad_multiplier", 3.5))
        for _ in range(iterations):
            coefficients = np.linalg.lstsq(design[inliers], segment_z[inliers], rcond=None)[0]
            residuals_all = np.abs(segment_z - design @ coefficients)
            median_residual = float(np.median(residuals_all[inliers]))
            mad = float(np.median(np.abs(residuals_all[inliers] - median_residual)))
            threshold = max(residual_floor, median_residual + mad_multiplier * 1.4826 * mad)
            updated = residuals_all <= threshold
            if int(updated.sum()) < minimum_cells or np.array_equal(updated, inliers):
                break
            inliers = updated
        coefficients = np.linalg.lstsq(design[inliers], segment_z[inliers], rcond=None)[0]
        residuals = np.abs(segment_z[inliers] - design[inliers] @ coefficients)
        cross_range = [
            float(np.percentile(segment_c[inliers], 2.0)),
            float(np.percentile(segment_c[inliers], 98.0)),
        ]
        rail_side_edge = cross_range[1] if side == "left" else cross_range[0]
        result.append(
            {
                "longitudinal_range_m": [
                    float(segment_index * segment_size),
                    float((segment_index + 1) * segment_size),
                ],
                "observed_cross_range_m": cross_range,
                "rail_side_edge_cross_m": rail_side_edge,
                "median_elevation_m": float(np.median(segment_z[inliers])),
                "plane_z_equals_a_s_plus_b_c_plus_d": [float(value) for value in coefficients],
                "absolute_residual_p90_m": float(np.percentile(residuals, 90.0)),
                "occupied_cell_count": int(inliers.sum()),
                "source_occupied_cell_count": source_count,
                "robust_inlier_fraction": float(inliers.sum() / source_count),
                "status": "observed_segment_fit_not_forced_across_gaps",
            }
        )
    return result


def _interior_gap_candidates(
    component_id: str,
    component_records: list[dict[str, Any]],
    shape: tuple[int, int],
    s_origin: float,
    c_origin: float,
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    observed = np.zeros(shape, dtype=bool)
    for item in component_records:
        observed[item["row"], item["column"]] = True
    enclosed = binary_fill_holes(observed) & ~observed
    labels, _ = label(enclosed, structure=np.ones((3, 3), dtype=np.int8))
    grid = float(settings["surface_grid_m"])
    minimum_area = float(settings["minimum_opening_area_m2"])
    maximum_area = float(settings["maximum_opening_area_m2"])
    minimum_boundary = float(settings["minimum_opening_boundary_coverage"])
    candidates: list[dict[str, Any]] = []
    for gap_label in np.unique(labels):
        if not gap_label:
            continue
        mask = labels == gap_label
        cells = int(mask.sum())
        area = cells * grid * grid
        if area < minimum_area or area > maximum_area:
            continue
        rows, columns = np.nonzero(mask)
        ring = binary_dilation(mask, structure=np.ones((3, 3), dtype=bool)) & ~mask
        boundary_coverage = float(np.count_nonzero(ring & observed)) / max(
            int(np.count_nonzero(ring)), 1
        )
        if boundary_coverage < minimum_boundary:
            continue
        candidates.append(
            {
                "id": f"{component_id}-OPENING-{len(candidates) + 1:03d}",
                "longitudinal_range_m": [
                    s_origin + int(columns.min()) * grid,
                    s_origin + (int(columns.max()) + 1) * grid,
                ],
                "cross_range_m": [
                    c_origin + int(rows.min()) * grid,
                    c_origin + (int(rows.max()) + 1) * grid,
                ],
                "area_m2": area,
                "missing_cell_count": cells,
                "boundary_coverage_ratio": boundary_coverage,
                "interpretation": "possible_opening_or_occlusion_not_asset_confirmation",
                "status": "automatic_enclosed_surface_gap_review_required",
            }
        )
    _classify_repeated_gap_patterns(candidates, settings)
    return candidates


def _classify_repeated_gap_patterns(
    candidates: list[dict[str, Any]], settings: dict[str, Any]
) -> None:
    """Suppress regularly repeated same-cross gaps as likely occlusion footprints."""
    for candidate in candidates:
        longitudinal = candidate["longitudinal_range_m"]
        cross = candidate["cross_range_m"]
        candidate["longitudinal_center_m"] = float(sum(longitudinal) / 2.0)
        candidate["cross_center_m"] = float(sum(cross) / 2.0)
        candidate["gap_classification"] = "unique_enclosed_gap_review_required"
        candidate["mesh_action"] = "do_not_cut_until_independent_evidence_confirms_opening"

    transverse_tolerance = float(settings["periodic_gap_cross_center_tolerance_m"])
    minimum_repetitions = int(settings["periodic_gap_minimum_repetitions"])
    minimum_spacing = float(settings["periodic_gap_minimum_spacing_m"])
    maximum_spacing_cv = float(settings["periodic_gap_maximum_spacing_cv"])
    maximum_outliers = int(settings.get("periodic_gap_maximum_outliers", 1))
    remaining = sorted(candidates, key=lambda item: item["cross_center_m"])
    groups: list[list[dict[str, Any]]] = []
    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        unmatched: list[dict[str, Any]] = []
        for candidate in remaining:
            if abs(candidate["cross_center_m"] - seed["cross_center_m"]) <= transverse_tolerance:
                group.append(candidate)
            else:
                unmatched.append(candidate)
        groups.append(group)
        remaining = unmatched

    pattern_index = 0
    for group in groups:
        if len(group) < minimum_repetitions:
            continue
        ordered = sorted(group, key=lambda item: item["longitudinal_center_m"])

        def spacing_quality(items: list[dict[str, Any]]) -> tuple[float, float]:
            spacings = np.diff([item["longitudinal_center_m"] for item in items])
            median = float(np.median(spacings))
            return median, float(np.std(spacings) / max(median, 1e-9))

        selected = ordered
        median_spacing, spacing_cv = spacing_quality(selected)
        removed: list[dict[str, Any]] = []
        while (
            (median_spacing < minimum_spacing or spacing_cv > maximum_spacing_cv)
            and len(selected) > minimum_repetitions
            and len(removed) < maximum_outliers
        ):
            trials = []
            for index in range(len(selected)):
                trial = selected[:index] + selected[index + 1 :]
                trial_spacing, trial_cv = spacing_quality(trial)
                penalty = 0.0 if trial_spacing >= minimum_spacing else 1.0
                trials.append((penalty + trial_cv, index, trial_spacing, trial_cv))
            _, remove_index, median_spacing, spacing_cv = min(trials)
            removed.append(selected[remove_index])
            selected = selected[:remove_index] + selected[remove_index + 1 :]
        if median_spacing < minimum_spacing or spacing_cv > maximum_spacing_cv:
            continue
        pattern_index += 1
        pattern_id = f"PERIODIC-GAP-PATTERN-{pattern_index:03d}"
        for candidate in selected:
            candidate.update(
                {
                    "gap_classification": "periodic_occlusion_pattern_not_opening_candidate",
                    "periodic_pattern_id": pattern_id,
                    "periodic_pattern_repetition_count": len(selected),
                    "periodic_pattern_spacing_m": median_spacing,
                    "periodic_pattern_spacing_cv": spacing_cv,
                    "periodic_pattern_outlier_count": len(removed),
                    "interpretation": "likely_repeated_asset_or_sensor_occlusion_not_platform_opening",
                    "status": "automatically_suppressed_from_platform_opening_mesh",
                    "mesh_action": "preserve_platform_surface_do_not_create_opening",
                }
            )


def _elevation_connected_record_groups(
    records: list[dict[str, Any]], settings: dict[str, Any]
) -> list[list[dict[str, Any]]]:
    """Keep touching horizontal surfaces separate when their elevations disagree."""

    if not records:
        return []
    tolerance = float(settings.get("component_neighbor_elevation_tolerance_m", 0.08))
    radius = int(settings.get("component_neighbor_radius_cells", 2))
    lookup = {(int(item["row"]), int(item["column"])): index for index, item in enumerate(records)}
    remaining = set(range(len(records)))
    groups: list[list[dict[str, Any]]] = []
    while remaining:
        seed = remaining.pop()
        indexes = [seed]
        queue = [seed]
        while queue:
            current = queue.pop()
            record = records[current]
            row = int(record["row"])
            column = int(record["column"])
            elevation = float(record["z_p50_m"])
            for row_offset in range(-radius, radius + 1):
                for column_offset in range(-radius, radius + 1):
                    if row_offset == 0 and column_offset == 0:
                        continue
                    neighbor = lookup.get((row + row_offset, column + column_offset))
                    if neighbor not in remaining:
                        continue
                    if abs(float(records[neighbor]["z_p50_m"]) - elevation) > tolerance:
                        continue
                    remaining.remove(neighbor)
                    indexes.append(neighbor)
                    queue.append(neighbor)
        groups.append([records[index] for index in indexes])
    return groups


def _side_components(
    side: str,
    s: np.ndarray,
    c: np.ndarray,
    z: np.ndarray,
    rail_reference_z: float,
    rail_edge_cross: float,
    s_range: tuple[float, float],
    settings: dict[str, Any],
    component_start: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clearance = float(settings["minimum_rail_center_clearance_m"])
    maximum_width = float(settings["maximum_platform_search_width_m"])
    if side == "left":
        c_min, c_max = rail_edge_cross - maximum_width, rail_edge_cross - clearance
    else:
        c_min, c_max = rail_edge_cross + clearance, rail_edge_cross + maximum_width
    relative_z = z - rail_reference_z
    selected = (
        (s >= s_range[0])
        & (s <= s_range[1])
        & (c >= c_min)
        & (c <= c_max)
        & (relative_z >= float(settings["minimum_platform_height_above_rail_m"]))
        & (relative_z <= float(settings["maximum_platform_height_above_rail_m"]))
    )
    if not np.any(selected):
        return [], {
            "side": side,
            "status": "no_points_in_platform_search_envelope",
            "search_cross_range_m": [c_min, c_max],
        }
    records, occupied, s_origin, c_origin, _, _ = _qualified_cells(
        s[selected], c[selected], z[selected], s_range[0], s_range[1], c_min, c_max, settings
    )
    if not records:
        return [], {
            "side": side,
            "status": "no_horizontal_surface_cells",
            "search_cross_range_m": [c_min, c_max],
        }
    closed = binary_closing(
        occupied,
        structure=np.ones((3, 3), dtype=bool),
        iterations=int(settings["binary_closing_iterations"]),
    )
    labels, _ = label(closed, structure=np.ones((3, 3), dtype=np.int8))
    components: list[dict[str, Any]] = []
    for component_label in np.unique(labels):
        if not component_label:
            continue
        parent_records = [
            item for item in records if labels[item["row"], item["column"]] == component_label
        ]
        for component_records in _elevation_connected_record_groups(parent_records, settings):
            if len(component_records) < int(settings["minimum_component_cells"]):
                continue
            component_s = np.asarray([item["s_m"] for item in component_records])
            component_c = np.asarray([item["c_m"] for item in component_records])
            component_z = np.asarray([item["z_p50_m"] for item in component_records])
            if float(np.ptp(component_s)) < float(settings["minimum_component_length_m"]):
                continue
            if float(np.ptp(component_c)) < float(settings["minimum_component_width_m"]):
                continue
            component_id = (
                f"PLATFORM-SURFACE-CANDIDATE-{component_start + len(components):03d}"
            )
            opening_candidates = _interior_gap_candidates(
                component_id,
                component_records,
                occupied.shape,
                s_origin,
                c_origin,
                settings,
            )
            components.append(
                {
                    "id": component_id,
                    "side": side,
                    "point_count": int(sum(item["point_count"] for item in component_records)),
                    "occupied_cell_count": len(component_records),
                    "longitudinal_range_m": [float(component_s.min()), float(component_s.max())],
                    "longitudinal_observed_intervals_m": _intervals(
                        component_s, float(settings["surface_grid_m"])
                    ),
                    "cross_range_m": [float(component_c.min()), float(component_c.max())],
                    "z_range_m": [float(component_z.min()), float(component_z.max())],
                    "median_height_above_rail_m": float(
                        np.median(component_z) - rail_reference_z
                    ),
                    "fit_segments": _fit_segments(component_records, side, settings),
                    "interior_gap_candidate_count": len(opening_candidates),
                    "periodic_occlusion_gap_count": sum(
                        item["gap_classification"]
                        == "periodic_occlusion_pattern_not_opening_candidate"
                        for item in opening_candidates
                    ),
                    "unresolved_interior_gap_count": sum(
                        item["gap_classification"] == "unique_enclosed_gap_review_required"
                        for item in opening_candidates
                    ),
                    "interior_gap_candidates": opening_candidates,
                    "status": "automatic_platform_surface_hypothesis_review_required",
                }
            )
    return components, {
        "side": side,
        "search_cross_range_m": [c_min, c_max],
        "source_point_count": int(selected.sum()),
        "qualified_cell_count": len(records),
        "component_count": len(components),
        "status": "platform_components_found" if components else "no_large_platform_component",
    }


def analyze_platform_surface_data(
    vertical_report: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    settings: dict[str, Any],
) -> dict[str, Any]:
    rails = vertical_report.get("rail_lines", [])
    if len(rails) < 2:
        return {
            "schema_version": "railway.platform-surface.v1",
            "project_id": vertical_report.get("project_id"),
            "segment_id": vertical_report["segment_id"],
            "platform_component_count": 0,
            "platform_components": [],
            "status": "insufficient_rail_frame",
        }
    s, c = _project_points(x, y, vertical_report["frame"])
    rail_cross = np.asarray([float(item["cross_position_m"]) for item in rails])
    rail_reference_z = float(np.median([float(item["median_z_m"]) for item in rails]))
    s_padding = float(settings["longitudinal_search_padding_m"])
    s_range = (float(s.min() + s_padding), float(s.max() - s_padding))
    components: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for side, edge in (("left", float(rail_cross.min())), ("right", float(rail_cross.max()))):
        side_components, diagnostic = _side_components(
            side, s, c, z, rail_reference_z, edge, s_range, settings, len(components) + 1
        )
        components.extend(side_components)
        diagnostics.append(diagnostic)
    return {
        "schema_version": "railway.platform-surface.v1",
        "project_id": vertical_report.get("project_id"),
        "segment_id": vertical_report["segment_id"],
        "source": vertical_report["source"],
        "frame": vertical_report["frame"],
        "rail_lines": rails,
        "rail_reference_z_m": rail_reference_z,
        "longitudinal_search_range_m": list(s_range),
        "platform_component_count": len(components),
        "platform_components": components,
        "side_diagnostics": diagnostics,
        "settings": settings,
        "status": (
            "platform_surface_hypotheses_generated_review_required"
            if components
            else "no_platform_surface_hypothesis"
        ),
        "limitations": [
            "Only horizontal surface evidence outside the outermost rail centers is considered.",
            "Enclosed gaps are candidates only; stairs and true openings require independent photo or point evidence.",
            "Repeated same-cross gaps are suppressed from mesh openings as likely periodic asset or sensor occlusion.",
            "Island platforms and vertical platform walls require later evidence stages.",
            "Observed intervals are not bridged into one full-length plane or rectangle.",
            "This stage never writes model geometry or asset registry records.",
        ],
    }


def build_platform_candidate_graph(report: dict[str, Any]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for component in report["platform_components"]:
        nodes.append(
            {
                "id": component["id"],
                "node_type": "platform_surface_hypothesis",
                "side": component["side"],
                "observed_intervals_m": component["longitudinal_observed_intervals_m"],
                "status": component["status"],
            }
        )
        for index, segment in enumerate(component["fit_segments"], start=1):
            segment_id = f"{component['id']}-FIT-{index:03d}"
            nodes.append(
                {
                    "id": segment_id,
                    "node_type": "platform_segment_fit",
                    "longitudinal_range_m": segment["longitudinal_range_m"],
                    "status": segment["status"],
                }
            )
            edges.append(
                {
                    "id": f"PLATFORM-EDGE-{len(edges) + 1:03d}",
                    "type": "platform_has_observed_fit_segment",
                    "source_id": component["id"],
                    "target_id": segment_id,
                    "status": "candidate_evidence_only",
                }
            )
        for opening in component["interior_gap_candidates"]:
            node_type = (
                "platform_periodic_occlusion_hypothesis"
                if opening["gap_classification"]
                == "periodic_occlusion_pattern_not_opening_candidate"
                else "platform_interior_gap_hypothesis"
            )
            nodes.append(
                {
                    "id": opening["id"],
                    "node_type": node_type,
                    "longitudinal_range_m": opening["longitudinal_range_m"],
                    "cross_range_m": opening["cross_range_m"],
                    "area_m2": opening["area_m2"],
                    "gap_classification": opening["gap_classification"],
                    "mesh_action": opening["mesh_action"],
                    "status": opening["status"],
                }
            )
            edges.append(
                {
                    "id": f"PLATFORM-EDGE-{len(edges) + 1:03d}",
                    "type": (
                        "platform_has_periodic_occlusion_pattern"
                        if node_type == "platform_periodic_occlusion_hypothesis"
                        else "platform_contains_enclosed_surface_gap"
                    ),
                    "source_id": component["id"],
                    "target_id": opening["id"],
                    "status": "candidate_evidence_only",
                }
            )
    return {
        "schema_version": "railway.platform-candidate-evidence-graph.v1",
        "project_id": report.get("project_id"),
        "segment_id": report["segment_id"],
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
        "status": "candidate_graph_only_no_asset_registry_or_model_write",
    }


def _render_diagnostic(report: dict[str, Any], output: Path) -> None:
    image = Image.new("RGB", (1800, 1000), "#06141d")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text(
        (30, 24),
        f"{report['segment_id']} platform surfaces | components {report['platform_component_count']}",
        fill="#eaf9ff",
        font=font,
    )
    draw.text(
        (30, 46),
        "Observed cells and segmented fits only; gaps are not bridged into a full-length plane.",
        fill="#ff9f31",
        font=font,
    )
    top = (30, 80, 1230, 950)
    profile = (1260, 80, 1770, 950)
    draw.rectangle(top, outline="#28505e", width=2)
    draw.rectangle(profile, outline="#28505e", width=2)
    components = report["platform_components"]
    rails = report.get("rail_lines", [])
    if not components:
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
        return
    s_values = [value for item in components for value in item["longitudinal_range_m"]]
    c_values = [value for item in components for value in item["cross_range_m"]]
    c_values.extend(float(item["cross_position_m"]) for item in rails)
    s_min, s_max = min(s_values), max(s_values)
    c_min, c_max = min(c_values), max(c_values)

    def pixel(s_value: float, c_value: float) -> tuple[int, int]:
        x = top[0] + 45 + int((s_value - s_min) / max(s_max - s_min, 1e-9) * (top[2] - top[0] - 90))
        y = top[3] - 40 - int((c_value - c_min) / max(c_max - c_min, 1e-9) * (top[3] - top[1] - 90))
        return x, y

    colors = ("#2cb5cf", "#a98cff", "#b7ff3c", "#ff9f31")
    for index, component in enumerate(components):
        color = colors[index % len(colors)]
        for segment in component["fit_segments"]:
            p1 = pixel(segment["longitudinal_range_m"][0], segment["observed_cross_range_m"][0])
            p2 = pixel(segment["longitudinal_range_m"][1], segment["observed_cross_range_m"][1])
            draw.rectangle((p1[0], p2[1], p2[0], p1[1]), fill=color, outline="#dff7ff")
        draw.text(
            (profile[0] + 18, profile[1] + 35 + index * 105),
            f"{component['id']} / {component['side']}",
            fill=color,
            font=font,
        )
        for opening in component["interior_gap_candidates"]:
            p1 = pixel(opening["longitudinal_range_m"][0], opening["cross_range_m"][0])
            p2 = pixel(opening["longitudinal_range_m"][1], opening["cross_range_m"][1])
            color = (
                "#ff9f31"
                if opening["gap_classification"]
                == "periodic_occlusion_pattern_not_opening_candidate"
                else "#ff4f62"
            )
            draw.rectangle((p1[0], p2[1], p2[0], p1[1]), outline=color, width=3)
        draw.text(
            (profile[0] + 18, profile[1] + 55 + index * 105),
            f"S {component['longitudinal_range_m'][0]:.1f}..{component['longitudinal_range_m'][1]:.1f} m",
            fill="#c1d5dc",
            font=font,
        )
        draw.text(
            (profile[0] + 18, profile[1] + 73 + index * 105),
            f"C {component['cross_range_m'][0]:.1f}..{component['cross_range_m'][1]:.1f} m | H {component['median_height_above_rail_m']:.2f} m",
            fill="#c1d5dc",
            font=font,
        )
    for rail in rails:
        p1 = pixel(s_min, float(rail["cross_position_m"]))
        p2 = pixel(s_max, float(rail["cross_position_m"]))
        draw.line((*p1, *p2), fill="#7a8b91", width=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def analyze_platform_surface(
    project: ProjectConfig,
    segment_id: str,
    overwrite: bool = False,
    vertical_report_path: str | Path | None = None,
    settings_path: str | Path | None = None,
) -> dict[str, Any]:
    reports = project.workspace_path("reports")
    vertical_path = (
        project.resolve(vertical_report_path)
        if vertical_report_path
        else reports / f"{segment_id}_vertical_hypotheses.json"
    )
    if not vertical_path.is_file():
        raise FileNotFoundError(vertical_path)
    vertical = load_json(vertical_path)
    source = Path(vertical["source"])
    if not source.is_file():
        raise FileNotFoundError(source)
    configured = project.value.get("algorithms", {}).get("platform_surface")
    resolved_settings = (
        project.resolve(settings_path)
        if settings_path
        else (project.resolve(configured) if configured else None)
    )
    settings = load_json(resolved_settings) if resolved_settings else _resource_settings()
    output_report = reports / f"{segment_id}_platform_surface.json"
    output_graph = reports / f"{segment_id}_platform_candidate_graph.json"
    output_image = reports / f"{segment_id}_platform_surface.png"
    if not overwrite:
        for path in (output_report, output_graph, output_image):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite: {path}")
    cloud = laspy.read(source)
    report = analyze_platform_surface_data(
        vertical,
        np.asarray(cloud.x, dtype=np.float64),
        np.asarray(cloud.y, dtype=np.float64),
        np.asarray(cloud.z, dtype=np.float64),
        settings,
    )
    report.update(
        {
            "vertical_hypotheses_report": str(vertical_path),
            "settings_source": str(resolved_settings) if resolved_settings else "bundled_default",
            "output_report": str(output_report),
            "output_candidate_graph": str(output_graph),
            "output_diagnostic_image": str(output_image),
        }
    )
    write_json(output_report, report)
    write_json(output_graph, build_platform_candidate_graph(report))
    _render_diagnostic(report, output_image)
    return report
