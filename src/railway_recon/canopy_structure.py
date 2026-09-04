from __future__ import annotations

import json
import math
from collections import Counter
from importlib import resources
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_closing, binary_dilation, label
from scipy.spatial import cKDTree

from .config import ProjectConfig
from .io import load_json, write_json


def _resource_settings() -> dict[str, Any]:
    path = resources.files("railway_recon.resources").joinpath(
        "canopy-structure.default.json"
    )
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


def _contiguous_intervals(
    occupied: np.ndarray, minimum: float, bin_size: float
) -> list[list[float]]:
    indexes = np.flatnonzero(occupied)
    if not indexes.size:
        return []
    groups: list[list[int]] = [[int(indexes[0])]]
    for index in indexes[1:]:
        if int(index) == groups[-1][-1] + 1:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])
    return [
        [
            minimum + group[0] * bin_size,
            minimum + (group[-1] + 1) * bin_size,
        ]
        for group in groups
    ]


def _surface_record(
    component_id: str,
    lane_id: str,
    indexes: np.ndarray,
    selected_cells: np.ndarray,
    s: np.ndarray,
    c: np.ndarray,
    z: np.ndarray,
    s_min: float,
    c_min: float,
    grid_size: float,
    settings: dict[str, Any],
) -> dict[str, Any]:
    component_s = s[indexes]
    component_c = c[indexes]
    component_z = z[indexes]
    design = np.column_stack(
        (component_s, component_c, np.ones(component_s.size, dtype=np.float64))
    )
    coefficients = np.linalg.lstsq(design, component_z, rcond=None)[0]
    residuals = np.abs(component_z - design @ coefficients)
    residual_p90 = float(np.percentile(residuals, 90.0))
    if residual_p90 <= float(settings["planar_residual_p90_maximum_m"]):
        surface_type = "planar"
    elif residual_p90 <= float(settings["shallow_curve_residual_p90_maximum_m"]):
        surface_type = "piecewise_planar_or_shallow_curve"
    else:
        surface_type = "curved_or_multi_surface"

    profile_bin = float(settings["outline_profile_bin_m"])
    profile_min = math.floor(float(component_s.min()) / profile_bin) * profile_bin
    profile_max = math.ceil(float(component_s.max()) / profile_bin) * profile_bin
    footprint_profiles: list[dict[str, Any]] = []
    for low in np.arange(profile_min, profile_max, profile_bin):
        mask = (component_s >= low) & (component_s < low + profile_bin)
        if int(np.count_nonzero(mask)) < int(settings["minimum_profile_points"]):
            continue
        footprint_profiles.append(
            {
                "longitudinal_range_m": [float(low), float(low + profile_bin)],
                "cross_range_m": [
                    float(np.percentile(component_c[mask], 1.0)),
                    float(np.percentile(component_c[mask], 99.0)),
                ],
                "point_count": int(np.count_nonzero(mask)),
            }
        )

    cross_bin = float(settings["cross_profile_bin_m"])
    cross_profile_min = math.floor(float(component_c.min()) / cross_bin) * cross_bin
    cross_profile_max = math.ceil(float(component_c.max()) / cross_bin) * cross_bin
    cross_profiles: list[dict[str, Any]] = []
    for low in np.arange(cross_profile_min, cross_profile_max, cross_bin):
        mask = (component_c >= low) & (component_c < low + cross_bin)
        if int(np.count_nonzero(mask)) < int(settings["minimum_profile_points"]):
            continue
        values = component_z[mask]
        cross_profiles.append(
            {
                "cross_range_m": [float(low), float(low + cross_bin)],
                "z_p10_m": float(np.percentile(values, 10.0)),
                "z_p50_m": float(np.percentile(values, 50.0)),
                "z_p90_m": float(np.percentile(values, 90.0)),
                "point_count": int(values.size),
            }
        )

    occupied_longitudinal = np.any(selected_cells, axis=0)
    return {
        "id": component_id,
        "lane_id": lane_id,
        "point_count": int(indexes.size),
        "occupied_cell_count": int(np.count_nonzero(selected_cells)),
        "longitudinal_range_m": [float(component_s.min()), float(component_s.max())],
        "cross_range_m": [float(component_c.min()), float(component_c.max())],
        "z_range_m": [float(component_z.min()), float(component_z.max())],
        "z_percentiles_m": {
            "p10": float(np.percentile(component_z, 10.0)),
            "p50": float(np.percentile(component_z, 50.0)),
            "p90": float(np.percentile(component_z, 90.0)),
        },
        "longitudinal_observed_intervals_m": _contiguous_intervals(
            occupied_longitudinal, s_min, grid_size
        ),
        "footprint_profiles": footprint_profiles,
        "cross_profiles": cross_profiles,
        "plane_diagnostic": {
            "z_equals_a_s_plus_b_c_plus_d": [float(value) for value in coefficients],
            "absolute_residual_p90_m": residual_p90,
            "status": "diagnostic_only_not_forced_geometry",
        },
        "surface_type": surface_type,
        "status": "automatic_roof_surface_hypothesis_review_required",
    }


def _roof_search_cross_range(
    columns: list[dict[str, Any]], settings: dict[str, Any]
) -> tuple[float, float]:
    column_cross = float(np.median([item["cross_position_m"] for item in columns]))
    maximum_span = float(settings["maximum_roof_cross_search_span_m"])
    outside_padding = float(settings["roof_cross_outside_padding_m"])
    rail_ranges = [
        item["rail_context"].get("track_envelope_cross_range_m")
        for item in columns
        if item.get("rail_context", {}).get("available")
    ]
    rail_ranges = [value for value in rail_ranges if value]
    if not rail_ranges:
        return column_cross - maximum_span / 2.0, column_cross + maximum_span / 2.0
    rail_min = float(np.median([value[0] for value in rail_ranges]))
    rail_max = float(np.median([value[1] for value in rail_ranges]))
    if column_cross < rail_min:
        return column_cross - outside_padding, column_cross + maximum_span
    if column_cross > rail_max:
        return column_cross - maximum_span, column_cross + outside_padding
    return column_cross - maximum_span / 2.0, column_cross + maximum_span / 2.0


def _extract_lane_roofs(
    lane_id: str,
    columns: list[dict[str, Any]],
    s: np.ndarray,
    c: np.ndarray,
    z: np.ndarray,
    settings: dict[str, Any],
    component_start: int,
) -> tuple[list[dict[str, Any]], list[np.ndarray], dict[str, Any]]:
    grid_size = float(settings["roof_grid_m"])
    column_s = np.asarray(
        [float(item["longitudinal_position_m"]) for item in columns], dtype=np.float64
    )
    column_c = np.asarray(
        [float(item["cross_position_m"]) for item in columns], dtype=np.float64
    )
    column_top = np.asarray(
        [float(item["maximum_z"]) for item in columns], dtype=np.float64
    )
    lane = columns[0]["lane"]
    spacing = lane.get("strong_vertical_median_spacing_m")
    spacing = float(spacing) if spacing else float(settings["fallback_column_spacing_m"])
    padding = spacing * float(settings["roof_longitudinal_padding_spacing_fraction"])
    s_min = math.floor((float(column_s.min()) - padding) / grid_size) * grid_size
    s_max = math.ceil((float(column_s.max()) + padding) / grid_size) * grid_size
    cross_min, cross_max = _roof_search_cross_range(columns, settings)
    c_min = math.floor(cross_min / grid_size) * grid_size
    c_max = math.ceil(cross_max / grid_size) * grid_size
    median_top = float(np.median(column_top))
    z_min = median_top - float(settings["roof_minimum_z_below_column_top_m"])
    z_max = median_top + float(settings["roof_maximum_z_above_column_top_m"])

    valid = (
        (s >= s_min)
        & (s <= s_max)
        & (c >= c_min)
        & (c <= c_max)
        & (z >= z_min)
        & (z <= z_max)
    )
    source_indexes = np.flatnonzero(valid)
    if not source_indexes.size:
        return [], [], {
            "lane_id": lane_id,
            "status": "no_points_in_roof_search_envelope",
            "search_envelope": {
                "longitudinal_range_m": [s_min, s_max],
                "cross_range_m": [c_min, c_max],
                "z_range_m": [z_min, z_max],
            },
        }

    s_bins = math.ceil((s_max - s_min) / grid_size) + 1
    c_bins = math.ceil((c_max - c_min) / grid_size) + 1
    s_index = np.clip(
        ((s[valid] - s_min) / grid_size).astype(np.int64), 0, s_bins - 1
    )
    c_index = np.clip(
        ((c[valid] - c_min) / grid_size).astype(np.int64), 0, c_bins - 1
    )
    counts = np.zeros((c_bins, s_bins), dtype=np.int64)
    np.add.at(counts, (c_index, s_index), 1)
    occupied = counts >= int(settings["roof_minimum_cell_points"])
    closed = binary_closing(
        occupied, iterations=int(settings["roof_binary_closing_iterations"])
    )
    connected_mask = binary_dilation(
        closed, iterations=int(settings["roof_binary_dilation_iterations"])
    )
    labels, component_count = label(connected_mask)
    seed_radius = int(settings["roof_seed_cell_radius"])
    seeded_labels: set[int] = set()
    for column_longitudinal, column_cross in zip(column_s, column_c, strict=True):
        si = int(np.clip((column_longitudinal - s_min) / grid_size, 0, s_bins - 1))
        ci = int(np.clip((column_cross - c_min) / grid_size, 0, c_bins - 1))
        window = labels[
            max(0, ci - seed_radius) : min(c_bins, ci + seed_radius + 1),
            max(0, si - seed_radius) : min(s_bins, si + seed_radius + 1),
        ]
        seeded_labels.update(int(value) for value in np.unique(window) if value)

    records: list[dict[str, Any]] = []
    point_indexes: list[np.ndarray] = []
    for label_value in sorted(seeded_labels):
        selected_cells = labels == label_value
        point_mask = labels[c_index, s_index] == label_value
        indexes = source_indexes[point_mask]
        if indexes.size < int(settings["roof_minimum_component_points"]):
            continue
        component_id = f"CANOPY-ROOF-CANDIDATE-{component_start + len(records):03d}"
        records.append(
            _surface_record(
                component_id,
                lane_id,
                indexes,
                selected_cells,
                s,
                c,
                z,
                s_min,
                c_min,
                grid_size,
                settings,
            )
        )
        point_indexes.append(indexes)
    return records, point_indexes, {
        "lane_id": lane_id,
        "status": "roof_components_extracted" if records else "no_seeded_roof_component",
        "search_envelope": {
            "longitudinal_range_m": [s_min, s_max],
            "cross_range_m": [c_min, c_max],
            "z_range_m": [z_min, z_max],
        },
        "grid_shape": [c_bins, s_bins],
        "occupied_cell_count": int(np.count_nonzero(occupied)),
        "connected_component_count": int(component_count),
        "seeded_component_count": len(seeded_labels),
    }


def _column_contacts(
    columns: list[dict[str, Any]],
    components: list[dict[str, Any]],
    component_indexes: list[np.ndarray],
    s: np.ndarray,
    c: np.ndarray,
    z: np.ndarray,
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    if not component_indexes:
        return [
            {
                "column_id": item["id"],
                "roof_id": None,
                "nearest_3d_distance_m": None,
                "status": "no_measurement",
            }
            for item in columns
        ]
    trees = [
        cKDTree(np.column_stack((s[indexes], c[indexes], z[indexes])))
        for indexes in component_indexes
    ]
    contacts: list[dict[str, Any]] = []
    threshold = float(settings["column_roof_maximum_nearest_distance_m"])
    for column in columns:
        query = np.asarray(
            [
                float(column["longitudinal_position_m"]),
                float(column["cross_position_m"]),
                float(column["maximum_z"]),
            ],
            dtype=np.float64,
        )
        options = []
        for component_index, (component, tree) in enumerate(
            zip(components, trees, strict=True)
        ):
            distance, point_index = tree.query(query)
            indexes = component_indexes[component_index]
            source_index = int(indexes[int(point_index)])
            options.append((float(distance), component, source_index))
        distance, component, source_index = min(options, key=lambda item: item[0])
        nearest = [float(s[source_index]), float(c[source_index]), float(z[source_index])]
        contacts.append(
            {
                "column_id": column["id"],
                "roof_id": component["id"],
                "column_top": query.tolist(),
                "nearest_roof_point": nearest,
                "nearest_3d_distance_m": distance,
                "vertical_difference_m": nearest[2] - float(query[2]),
                "maximum_allowed_distance_m": threshold,
                "status": "pass" if distance <= threshold else "gap",
                "evidence_scope": "shared_point_cloud_geometry_not_independent_mesh_contact",
            }
        )
    return contacts


def _transverse_member_checks(
    columns: list[dict[str, Any]],
    components: list[dict[str, Any]],
    component_indexes: list[np.ndarray],
    s: np.ndarray,
    c: np.ndarray,
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    if not component_indexes:
        return []
    selected = np.unique(np.concatenate(component_indexes))
    half_width = float(settings["transverse_member_strip_half_width_m"])
    side_offset = float(settings["transverse_member_side_offset_m"])
    minimum_ratio = float(settings["transverse_member_minimum_density_ratio"])
    minimum_points = int(settings["transverse_member_minimum_center_points"])
    checks: list[dict[str, Any]] = []
    for number, column in enumerate(columns, start=1):
        longitudinal = float(column["longitudinal_position_m"])
        center_count = int(
            np.count_nonzero(np.abs(s[selected] - longitudinal) <= half_width)
        )
        before_count = int(
            np.count_nonzero(
                np.abs(s[selected] - (longitudinal - side_offset)) <= half_width
            )
        )
        after_count = int(
            np.count_nonzero(
                np.abs(s[selected] - (longitudinal + side_offset)) <= half_width
            )
        )
        side_mean = (before_count + after_count) / 2.0
        ratio = center_count / max(side_mean, 1.0)
        detected = center_count >= minimum_points and ratio >= minimum_ratio
        roof_candidates = [
            component
            for component in components
            if component["longitudinal_range_m"][0]
            <= longitudinal
            <= component["longitudinal_range_m"][1]
        ]
        checks.append(
            {
                "id": f"TRANSVERSE-MEMBER-CHECK-{number:03d}",
                "column_id": column["id"],
                "roof_ids": [item["id"] for item in roof_candidates],
                "longitudinal_position_m": longitudinal,
                "center_strip_point_count": center_count,
                "side_strip_point_count_mean": side_mean,
                "density_ratio": ratio,
                "minimum_density_ratio": minimum_ratio,
                "result": (
                    "distinct_transverse_member_candidate"
                    if detected
                    else "no_distinct_transverse_member_in_point_cloud"
                ),
                "status": "automatic_hypothesis_review_required",
            }
        )
    return checks


def analyze_canopy_structure_data(
    vertical_report: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    settings: dict[str, Any],
) -> dict[str, Any]:
    if vertical_report.get("schema_version") != "railway.vertical-hypotheses.v1":
        raise ValueError("Expected railway.vertical-hypotheses.v1")
    frame = vertical_report["frame"]
    s, c = _project_points(x, y, frame)
    stable_columns = [
        item
        for item in vertical_report["candidates"]
        if item["predicted_class"] == "canopy_column"
        and item["lane"]["periodic_column_grid_candidate"]
        and item["lane"].get("periodic_pattern_confidence") in {"high", "medium"}
    ]
    columns_by_lane: dict[str, list[dict[str, Any]]] = {}
    for column in stable_columns:
        columns_by_lane.setdefault(str(column["lane"]["id"]), []).append(column)

    components: list[dict[str, Any]] = []
    component_indexes: list[np.ndarray] = []
    lane_diagnostics: list[dict[str, Any]] = []
    for lane_id, columns in sorted(columns_by_lane.items()):
        columns.sort(key=lambda item: float(item["longitudinal_position_m"]))
        records, indexes, diagnostic = _extract_lane_roofs(
            lane_id,
            columns,
            s,
            c,
            z,
            settings,
            component_start=len(components) + 1,
        )
        components.extend(records)
        component_indexes.extend(indexes)
        lane_diagnostics.append(diagnostic)

    contacts = _column_contacts(
        stable_columns, components, component_indexes, s, c, z, settings
    )
    transverse_checks = _transverse_member_checks(
        stable_columns, components, component_indexes, s, c, settings
    )
    contact_counts = Counter(item["status"] for item in contacts)
    distinct_transverse = sum(
        item["result"] == "distinct_transverse_member_candidate"
        for item in transverse_checks
    )
    if not stable_columns:
        status = "no_stable_canopy_column_grid"
    elif not components:
        status = "stable_columns_but_no_roof_component"
    elif distinct_transverse == 0:
        status = "roof_and_column_contacts_found_transverse_members_unresolved"
    else:
        status = "canopy_structure_hypotheses_review_required"
    return {
        "schema_version": "railway.canopy-structure-hypotheses.v1",
        "project_id": vertical_report.get("project_id"),
        "segment_id": vertical_report["segment_id"],
        "source": vertical_report["source"],
        "frame": frame,
        "rail_lines": vertical_report.get("rail_lines", []),
        "stable_column_count": len(stable_columns),
        "stable_column_ids": [item["id"] for item in stable_columns],
        "roof_component_count": len(components),
        "roof_components": components,
        "column_roof_contacts": contacts,
        "column_roof_contact_counts": {
            name: int(contact_counts.get(name, 0))
            for name in ("pass", "gap", "no_measurement")
        },
        "transverse_member_check_count": len(transverse_checks),
        "distinct_transverse_member_candidate_count": distinct_transverse,
        "transverse_member_checks": transverse_checks,
        "lane_diagnostics": lane_diagnostics,
        "settings": settings,
        "status": status,
        "limitations": [
            "Roof extraction is seeded by the periodic column grid and remains a hypothesis.",
            "Plane fitting is diagnostic only; curved or multi-surface roofs are not flattened.",
            "Column-to-roof distance uses shared point-cloud geometry, not an independent mesh interface.",
            "No distinct transverse member is a data conclusion, not proof that a beam is absent.",
        ],
    }


def build_canopy_candidate_graph(report: dict[str, Any]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for column_id in report["stable_column_ids"]:
        nodes.append(
            {
                "id": column_id,
                "node_type": "canopy_column_hypothesis",
                "status": "automatic_hypothesis_review_required",
            }
        )
    for component in report["roof_components"]:
        nodes.append(
            {
                "id": component["id"],
                "node_type": "canopy_roof_surface_hypothesis",
                "surface_type": component["surface_type"],
                "observed_intervals_m": component[
                    "longitudinal_observed_intervals_m"
                ],
                "status": component["status"],
            }
        )
    for contact in report["column_roof_contacts"]:
        if contact["roof_id"] is None:
            continue
        edges.append(
            {
                "id": f"CANOPY-EDGE-{len(edges) + 1:03d}",
                "type": "column_top_near_roof_observation",
                "source_id": contact["column_id"],
                "target_id": contact["roof_id"],
                "nearest_3d_distance_m": contact["nearest_3d_distance_m"],
                "status": contact["status"],
            }
        )
    for check in report["transverse_member_checks"]:
        nodes.append(
            {
                "id": check["id"],
                "node_type": "transverse_member_evidence_check",
                "result": check["result"],
                "density_ratio": check["density_ratio"],
                "status": check["status"],
            }
        )
        edges.append(
            {
                "id": f"CANOPY-EDGE-{len(edges) + 1:03d}",
                "type": "column_has_transverse_member_check",
                "source_id": check["column_id"],
                "target_id": check["id"],
                "status": check["status"],
            }
        )
        for roof_id in check["roof_ids"]:
            edges.append(
                {
                    "id": f"CANOPY-EDGE-{len(edges) + 1:03d}",
                    "type": "transverse_member_check_below_roof_candidate",
                    "source_id": check["id"],
                    "target_id": roof_id,
                    "status": check["status"],
                }
            )
    return {
        "schema_version": "railway.canopy-candidate-evidence-graph.v1",
        "project_id": report.get("project_id"),
        "segment_id": report["segment_id"],
        "source_canopy_report": report.get("output_report"),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
        "status": "candidate_graph_only_no_asset_registry_write",
    }


def _draw_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> None:
    draw.rectangle(box, outline="#28505e", width=2)
    draw.text((box[0] + 14, box[1] + 10), title, fill="#b7ff3c", font=ImageFont.load_default())


def _render_diagnostic(report: dict[str, Any], output: Path) -> None:
    image = Image.new("RGB", (1800, 1000), "#06141d")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text(
        (35, 25),
        (
            f"{report['segment_id']} canopy structure | columns {report['stable_column_count']} | "
            f"roof components {report['roof_component_count']} | "
            f"distinct transverse members {report['distinct_transverse_member_candidate_count']}"
        ),
        fill="#eaf9ff",
        font=font,
    )
    top_box = (35, 70, 1115, 560)
    cross_box = (1140, 70, 1765, 560)
    contact_box = (35, 590, 1765, 950)
    _draw_panel(draw, top_box, "TOP / OBSERVED ROOF FOOTPRINT PROFILES")
    _draw_panel(draw, cross_box, "CROSS SECTION / P10-P90 SURFACE BAND")
    _draw_panel(draw, contact_box, "COLUMN CONTACT DISTANCE + TRANSVERSE DENSITY CHECK")
    components = report["roof_components"]
    contacts = report["column_roof_contacts"]
    checks = report["transverse_member_checks"]
    if not components:
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
        return

    s_values = [value for item in components for value in item["longitudinal_range_m"]]
    c_values = [value for item in components for value in item["cross_range_m"]]
    for rail in report.get("rail_lines", []):
        c_values.append(float(rail["cross_position_m"]))
    s_min, s_max = min(s_values), max(s_values)
    c_min, c_max = min(c_values), max(c_values)

    def top_pixel(s_value: float, c_value: float) -> tuple[int, int]:
        x = top_box[0] + 45 + int(
            (s_value - s_min) / max(s_max - s_min, 1e-9) * (top_box[2] - top_box[0] - 90)
        )
        y = top_box[3] - 35 - int(
            (c_value - c_min) / max(c_max - c_min, 1e-9) * (top_box[3] - top_box[1] - 85)
        )
        return x, y

    for component in components:
        for profile in component["footprint_profiles"]:
            p1 = top_pixel(profile["longitudinal_range_m"][0], profile["cross_range_m"][0])
            p2 = top_pixel(profile["longitudinal_range_m"][1], profile["cross_range_m"][1])
            draw.rectangle((p1[0], p2[1], p2[0], p1[1]), fill="#174755", outline="#2cb5cf")
    for rail in report.get("rail_lines", []):
        p1 = top_pixel(s_min, float(rail["cross_position_m"]))
        p2 = top_pixel(s_max, float(rail["cross_position_m"]))
        draw.line((p1[0], p1[1], p2[0], p2[1]), fill="#7b8d93", width=1)
    for contact in contacts:
        s_value, c_value, _ = contact["column_top"]
        px, py = top_pixel(float(s_value), float(c_value))
        color = "#b7ff3c" if contact["status"] == "pass" else "#ff7a35"
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=color)

    cross_profiles = components[0]["cross_profiles"]
    cross_min = min(item["cross_range_m"][0] for item in cross_profiles)
    cross_max = max(item["cross_range_m"][1] for item in cross_profiles)
    z_min = min(item["z_p10_m"] for item in cross_profiles)
    z_max = max(item["z_p90_m"] for item in cross_profiles)

    def cross_pixel(c_value: float, z_value: float) -> tuple[int, int]:
        x = cross_box[0] + 40 + int(
            (c_value - cross_min) / max(cross_max - cross_min, 1e-9) * (cross_box[2] - cross_box[0] - 80)
        )
        y = cross_box[3] - 35 - int(
            (z_value - z_min) / max(z_max - z_min, 1e-9) * (cross_box[3] - cross_box[1] - 85)
        )
        return x, y

    median_points: list[tuple[int, int]] = []
    for profile in cross_profiles:
        center = sum(profile["cross_range_m"]) / 2.0
        lower = cross_pixel(center, profile["z_p10_m"])
        upper = cross_pixel(center, profile["z_p90_m"])
        draw.line((lower[0], lower[1], upper[0], upper[1]), fill="#2c7f91", width=5)
        median_points.append(cross_pixel(center, profile["z_p50_m"]))
    if len(median_points) >= 2:
        draw.line(median_points, fill="#7de7ff", width=2)
    surface = components[0]
    draw.text(
        (cross_box[0] + 16, cross_box[3] - 22),
        f"surface={surface['surface_type']} plane P90={surface['plane_diagnostic']['absolute_residual_p90_m']:.3f} m",
        fill="#93aeb7",
        font=font,
    )

    chart_left = contact_box[0] + 55
    chart_right = contact_box[2] - 40
    chart_top = contact_box[1] + 55
    chart_bottom = contact_box[3] - 45
    threshold = float(report["settings"]["column_roof_maximum_nearest_distance_m"])
    ratio_threshold = float(report["settings"]["transverse_member_minimum_density_ratio"])
    count = max(len(contacts), 1)
    for index, contact in enumerate(sorted(contacts, key=lambda item: item["column_top"][0])):
        x = chart_left + int(index / max(count - 1, 1) * (chart_right - chart_left))
        distance = float(contact["nearest_3d_distance_m"] or 0.0)
        distance_y = chart_top + int(min(distance / max(threshold * 2.0, 1e-9), 1.0) * 100)
        draw.ellipse((x - 5, distance_y - 5, x + 5, distance_y + 5), fill="#b7ff3c")
        draw.text((x - 12, chart_bottom + 5), f"{contact['column_top'][0]:.0f}", fill="#8ca5ad", font=font)
    distance_threshold_y = chart_top + 50
    draw.line((chart_left, distance_threshold_y, chart_right, distance_threshold_y), fill="#ff7a35", width=1)
    draw.text((chart_left, chart_top - 18), "nearest roof distance (threshold line)", fill="#b7ff3c", font=font)
    ratio_y_base = chart_top + 185
    for index, check in enumerate(sorted(checks, key=lambda item: item["longitudinal_position_m"])):
        x = chart_left + int(index / max(count - 1, 1) * (chart_right - chart_left))
        ratio = float(check["density_ratio"])
        y = ratio_y_base + 80 - int(min(ratio / max(ratio_threshold * 1.5, 1e-9), 1.0) * 80)
        color = "#29d3ff" if check["result"] == "distinct_transverse_member_candidate" else "#a98cff"
        draw.rectangle((x - 5, y, x + 5, ratio_y_base + 80), fill=color)
    ratio_threshold_y = ratio_y_base + 80 - int(80 / 1.5)
    draw.line((chart_left, ratio_threshold_y, chart_right, ratio_threshold_y), fill="#ff7a35", width=1)
    draw.text((chart_left, ratio_y_base - 15), "transverse strip density ratio", fill="#a98cff", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def analyze_canopy_structure(
    project: ProjectConfig,
    segment_id: str,
    overwrite: bool = False,
    vertical_report_path: str | Path | None = None,
    settings_path: str | Path | None = None,
) -> dict[str, Any]:
    vertical_path = (
        project.resolve(vertical_report_path)
        if vertical_report_path
        else project.workspace_path("reports") / f"{segment_id}_vertical_hypotheses.json"
    )
    if not vertical_path.is_file():
        raise FileNotFoundError(vertical_path)
    vertical_report = load_json(vertical_path)
    source = Path(vertical_report["source"])
    if not source.is_file():
        raise FileNotFoundError(source)
    if settings_path:
        resolved_settings = project.resolve(settings_path)
        settings = load_json(resolved_settings)
    else:
        configured = project.value.get("algorithms", {}).get("canopy_structure")
        if configured:
            resolved_settings = project.resolve(configured)
            settings = load_json(resolved_settings)
        else:
            resolved_settings = None
            settings = _resource_settings()

    output_report = project.workspace_path("reports") / f"{segment_id}_canopy_structure.json"
    output_graph = (
        project.workspace_path("reports") / f"{segment_id}_canopy_candidate_graph.json"
    )
    output_image = (
        project.workspace_path("reports") / f"{segment_id}_canopy_structure.png"
    )
    for path in (output_report, output_graph, output_image):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite: {path}")

    cloud = laspy.read(source)
    report = analyze_canopy_structure_data(
        vertical_report,
        np.asarray(cloud.x, dtype=np.float64),
        np.asarray(cloud.y, dtype=np.float64),
        np.asarray(cloud.z, dtype=np.float64),
        settings,
    )
    report["vertical_hypotheses_report"] = str(vertical_path)
    report["settings_source"] = str(resolved_settings) if resolved_settings else "bundled_default"
    report["output_report"] = str(output_report)
    report["output_candidate_graph"] = str(output_graph)
    report["output_diagnostic_image"] = str(output_image)
    write_json(output_report, report)
    write_json(output_graph, build_canopy_candidate_graph(report))
    _render_diagnostic(report, output_image)
    return report
