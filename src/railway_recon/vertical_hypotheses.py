from __future__ import annotations

import json
import math
from collections import Counter
from importlib import resources
from itertools import pairwise
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

from .config import ProjectConfig
from .io import load_json, write_json

CLASS_NAMES = (
    "catenary_support",
    "canopy_column",
    "building_edge",
    "false_positive",
)


def _resource_settings() -> dict[str, Any]:
    path = resources.files("railway_recon.resources").joinpath(
        "vertical-hypotheses.default.json"
    )
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def _project_xy(
    x: float, y: float, frame: dict[str, Any]
) -> tuple[float, float]:
    dx = x - float(frame["origin_xy"][0])
    dy = y - float(frame["origin_xy"][1])
    along = frame["along_xy"]
    cross = frame["cross_xy"]
    return (
        dx * float(along[0]) + dy * float(along[1]),
        dx * float(cross[0]) + dy * float(cross[1]),
    )


def _z_overlap_ratio(first: dict[str, Any], second: dict[str, Any]) -> float:
    low = max(float(first["minimum_z"]), float(second["minimum_z"]))
    high = min(float(first["maximum_z"]), float(second["maximum_z"]))
    overlap = max(0.0, high - low)
    first_span = max(1e-9, float(first["maximum_z"]) - float(first["minimum_z"]))
    second_span = max(1e-9, float(second["maximum_z"]) - float(second["minimum_z"]))
    return overlap / min(first_span, second_span)


def _candidate_point_features(
    candidate: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    tree: cKDTree,
    settings: dict[str, Any],
) -> dict[str, float | int]:
    radius = _clamp(
        float(candidate["footprint_m"])
        * float(settings["evidence_radius_footprint_scale"])
        + float(settings["evidence_radius_margin_m"]),
        float(settings["evidence_radius_min_m"]),
        float(settings["evidence_radius_max_m"]),
    )
    indexes = np.asarray(
        tree.query_ball_point(
            [float(candidate["center_x"]), float(candidate["center_y"])], radius
        ),
        dtype=np.int64,
    )
    z_margin = float(settings["evidence_z_margin_m"])
    if indexes.size:
        indexes = indexes[
            (z[indexes] >= float(candidate["minimum_z"]) - z_margin)
            & (z[indexes] <= float(candidate["maximum_z"]) + z_margin)
        ]
    if indexes.size < 3:
        return {
            "evidence_radius_m": radius,
            "local_point_count": int(indexes.size),
            "vertical_axis_z": 0.0,
            "linearity": 0.0,
            "planarity": 0.0,
            "vertical_occupied_ratio": 0.0,
            "horizontal_drift_m_per_m": float("inf"),
        }

    points = np.column_stack((x[indexes], y[indexes], z[indexes]))
    centered = points - np.median(points, axis=0)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    main_axis = eigenvectors[:, order[0]]
    leading = max(float(eigenvalues[0]), 1e-12)

    local_z = z[indexes]
    bin_size = float(settings["vertical_bin_m"])
    bin_count = max(1, math.ceil(float(np.ptp(local_z)) / bin_size))
    occupied = np.unique(
        np.clip(((local_z - local_z.min()) / bin_size).astype(np.int64), 0, bin_count - 1)
    )

    design = np.column_stack((local_z, np.ones(indexes.size)))
    x_slope = float(np.linalg.lstsq(design, x[indexes], rcond=None)[0][0])
    y_slope = float(np.linalg.lstsq(design, y[indexes], rcond=None)[0][0])
    return {
        "evidence_radius_m": radius,
        "local_point_count": int(indexes.size),
        "vertical_axis_z": abs(float(main_axis[2])),
        "linearity": float((eigenvalues[0] - eigenvalues[1]) / leading),
        "planarity": float((eigenvalues[1] - eigenvalues[2]) / leading),
        "vertical_occupied_ratio": float(occupied.size / bin_count),
        "horizontal_drift_m_per_m": float(math.hypot(x_slope, y_slope)),
    }


def _is_vertical_anchor(candidate: dict[str, Any], settings: dict[str, Any]) -> bool:
    features = candidate["features"]
    return (
        float(features["vertical_occupied_ratio"])
        >= float(settings["anchor_minimum_vertical_occupancy"])
        and float(features["vertical_axis_z"])
        >= float(settings["anchor_minimum_axis_z"])
        and float(features["horizontal_drift_m_per_m"])
        <= float(settings["anchor_maximum_horizontal_drift_m_per_m"])
    )


def _merge_candidate_indexes(
    candidates: list[dict[str, Any]], settings: dict[str, Any]
) -> list[list[int]]:
    remaining = set(range(len(candidates)))
    groups: list[list[int]] = []
    anchor_order = sorted(
        (index for index in remaining if _is_vertical_anchor(candidates[index], settings)),
        key=lambda index: int(candidates[index]["point_count"]),
        reverse=True,
    )
    for anchor_index in anchor_order:
        if anchor_index not in remaining:
            continue
        anchor = candidates[anchor_index]
        group = [anchor_index]
        for other_index in list(remaining):
            if other_index == anchor_index:
                continue
            other = candidates[other_index]
            distance = math.hypot(
                float(anchor["center_x"]) - float(other["center_x"]),
                float(anchor["center_y"]) - float(other["center_y"]),
            )
            if (
                distance <= float(settings["anchor_merge_radius_m"])
                and _z_overlap_ratio(anchor, other)
                >= float(settings["merge_minimum_z_overlap_ratio"])
            ):
                group.append(other_index)
        remaining.difference_update(group)
        groups.append(sorted(group))

    residual_radius = float(settings["residual_merge_radius_m"])
    while remaining:
        seed_index = max(
            remaining, key=lambda index: int(candidates[index]["point_count"])
        )
        seed = candidates[seed_index]
        group = [seed_index]
        for other_index in list(remaining):
            if other_index == seed_index:
                continue
            other = candidates[other_index]
            distance = math.hypot(
                float(seed["center_x"]) - float(other["center_x"]),
                float(seed["center_y"]) - float(other["center_y"]),
            )
            if (
                distance <= residual_radius
                and _z_overlap_ratio(seed, other)
                >= float(settings["merge_minimum_z_overlap_ratio"])
            ):
                group.append(other_index)
        remaining.difference_update(group)
        groups.append(sorted(group))
    return groups


def _aggregate_group(
    group: list[int], candidates: list[dict[str, Any]], frame: dict[str, Any]
) -> dict[str, Any]:
    members = [candidates[index] for index in group]
    weights = np.asarray(
        [max(1, int(member["point_count"])) for member in members], dtype=np.float64
    )
    center_x = float(
        np.average([float(member["center_x"]) for member in members], weights=weights)
    )
    center_y = float(
        np.average([float(member["center_y"]) for member in members], weights=weights)
    )
    minimum_z = min(float(member["minimum_z"]) for member in members)
    maximum_z = max(float(member["maximum_z"]) for member in members)
    longitudinal, cross = _project_xy(center_x, center_y, frame)
    return {
        "source_candidate_ids": [str(member["id"]) for member in members],
        "source_candidate_count": len(members),
        "center_x": center_x,
        "center_y": center_y,
        "minimum_z": minimum_z,
        "maximum_z": maximum_z,
        "height_m": maximum_z - minimum_z,
        "footprint_m": max(float(member["footprint_m"]) for member in members),
        "source_point_count_sum": sum(int(member["point_count"]) for member in members),
        "longitudinal_position_m": longitudinal,
        "cross_position_m": cross,
    }


def _median_and_mad(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    return median, float(np.median(np.abs(array - median)))


def _assign_lane_features(
    candidates: list[dict[str, Any]], settings: dict[str, Any]
) -> list[dict[str, Any]]:
    tolerance = float(settings["lane_cross_tolerance_m"])
    lanes: list[list[int]] = []
    for candidate_index in sorted(
        range(len(candidates)), key=lambda index: float(candidates[index]["cross_position_m"])
    ):
        cross = float(candidates[candidate_index]["cross_position_m"])
        nearest_lane: list[int] | None = None
        nearest_distance = float("inf")
        for lane in lanes:
            lane_cross = float(
                np.median([float(candidates[index]["cross_position_m"]) for index in lane])
            )
            distance = abs(cross - lane_cross)
            if distance <= tolerance and distance < nearest_distance:
                nearest_lane = lane
                nearest_distance = distance
        if nearest_lane is None:
            lanes.append([candidate_index])
        else:
            nearest_lane.append(candidate_index)

    lane_records: list[dict[str, Any]] = []
    for lane_number, lane in enumerate(lanes, start=1):
        longitudinal = sorted(
            float(candidates[index]["longitudinal_position_m"]) for index in lane
        )
        spacings = [second - first for first, second in pairwise(longitudinal)]
        median_spacing, spacing_mad = _median_and_mad(spacings)
        strong = [
            index
            for index in lane
            if _is_vertical_anchor(candidates[index], settings)
        ]
        strong_longitudinal = sorted(
            float(candidates[index]["longitudinal_position_m"]) for index in strong
        )
        strong_spacings = [
            second - first
            for first, second in pairwise(strong_longitudinal)
        ]
        strong_median, strong_mad = _median_and_mad(strong_spacings)
        periodic = (
            len(strong) >= int(settings["periodic_lane_minimum_count"])
            and strong_median is not None
            and float(settings["periodic_spacing_min_m"])
            <= strong_median
            <= float(settings["periodic_spacing_max_m"])
            and strong_mad is not None
            and strong_mad <= float(settings["periodic_spacing_max_mad_m"])
        )
        if periodic and len(strong) >= 5 and strong_mad is not None and strong_mad <= 0.25:
            pattern_confidence = "high"
        elif periodic:
            pattern_confidence = "medium"
        else:
            pattern_confidence = "not_applicable"
        dense_edge = (
            len(lane) >= int(settings["dense_lane_minimum_count"])
            and median_spacing is not None
            and median_spacing <= float(settings["dense_lane_maximum_median_spacing_m"])
        )
        lane_id = f"VERTICAL-LANE-{lane_number:03d}"
        record = {
            "id": lane_id,
            "cross_position_m": float(
                np.median([float(candidates[index]["cross_position_m"]) for index in lane])
            ),
            "candidate_count": len(lane),
            "median_spacing_m": median_spacing,
            "spacing_mad_m": spacing_mad,
            "strong_vertical_count": len(strong),
            "strong_vertical_median_spacing_m": strong_median,
            "strong_vertical_spacing_mad_m": strong_mad,
            "periodic_column_grid_candidate": periodic,
            "periodic_pattern_confidence": pattern_confidence,
            "dense_surface_edge_candidate": dense_edge,
        }
        lane_records.append(record)
        for index in lane:
            candidates[index]["lane"] = record.copy()
    return lane_records


def _rail_context(
    candidate: dict[str, Any], rail_report: dict[str, Any] | None
) -> dict[str, Any]:
    if not rail_report or not rail_report.get("rail_lines"):
        return {
            "available": False,
            "nearest_rail_distance_m": None,
            "track_envelope_distance_m": None,
            "inside_track_envelope": None,
        }
    _, rail_cross = _project_xy(
        float(candidate["center_x"]),
        float(candidate["center_y"]),
        rail_report["frame"],
    )
    lines = [float(line["cross_position_m"]) for line in rail_report["rail_lines"]]
    minimum = min(lines)
    maximum = max(lines)
    if minimum <= rail_cross <= maximum:
        envelope_distance = 0.0
    else:
        envelope_distance = min(abs(rail_cross - minimum), abs(rail_cross - maximum))
    nearest_line_index = min(range(len(lines)), key=lambda index: abs(rail_cross - lines[index]))
    return {
        "available": True,
        "cross_position_in_rail_frame_m": rail_cross,
        "nearest_rail_id": str(rail_report["rail_lines"][nearest_line_index]["id"]),
        "nearest_rail_distance_m": abs(rail_cross - lines[nearest_line_index]),
        "track_envelope_cross_range_m": [minimum, maximum],
        "track_envelope_distance_m": envelope_distance,
        "inside_track_envelope": minimum <= rail_cross <= maximum,
    }


def _rail_line_summary(
    rail_report: dict[str, Any] | None, target_frame: dict[str, Any]
) -> list[dict[str, Any]]:
    if not rail_report or not rail_report.get("rail_lines"):
        return []
    source_frame = rail_report["frame"]
    origin_x, origin_y = (float(value) for value in source_frame["origin_xy"])
    cross_x, cross_y = (float(value) for value in source_frame["cross_xy"])
    result = []
    for line in rail_report["rail_lines"]:
        source_cross = float(line["cross_position_m"])
        world_x = origin_x + source_cross * cross_x
        world_y = origin_y + source_cross * cross_y
        _, target_cross = _project_xy(world_x, world_y, target_frame)
        result.append(
            {
                "id": str(line["id"]),
                "cross_position_m": target_cross,
                "median_z_m": line.get("median_z_m"),
            }
        )
    return result


def _cable_context(
    candidate: dict[str, Any], cable_candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    if not cable_candidates:
        return {
            "available": False,
            "nearest_cable_id": None,
            "cross_difference_m": None,
            "top_elevation_difference_m": None,
        }
    cross = float(candidate["cross_position_m"])
    top = float(candidate["maximum_z"])
    nearest = min(
        cable_candidates,
        key=lambda item: abs(cross - float(item["cross_position_m"])),
    )
    return {
        "available": True,
        "nearest_cable_id": str(nearest["id"]),
        "cross_difference_m": abs(cross - float(nearest["cross_position_m"])),
        "top_elevation_difference_m": top - float(nearest["elevation_z"]),
        "longitudinal_coverage": float(nearest["longitudinal_coverage"]),
    }


def _classify_candidate(
    candidate: dict[str, Any], settings: dict[str, Any]
) -> tuple[str, dict[str, float], str, list[str]]:
    features = candidate["features"]
    lane = candidate["lane"]
    rail = candidate["rail_context"]
    cable = candidate["cable_context"]
    occupancy = float(features["vertical_occupied_ratio"])
    vertical_support = _clamp(
        (occupancy - float(settings["vertical_support_occupancy_floor"]))
        / float(settings["vertical_support_occupancy_span"])
    )
    vertical_support *= _clamp(
        (float(features["vertical_axis_z"]) - 0.90) / 0.09
    )
    low_occupancy = 1.0 - _clamp(
        occupancy / float(settings["anchor_minimum_vertical_occupancy"])
    )
    height = float(candidate["height_m"])
    catenary_height = _clamp(
        (height - float(settings["catenary_minimum_height_m"]))
        / float(settings["catenary_height_ramp_m"])
    )
    canopy_height = 1.0 if (
        float(settings["canopy_minimum_height_m"])
        <= height
        <= float(settings["canopy_maximum_height_m"])
    ) else 0.0
    envelope_distance = rail.get("track_envelope_distance_m")
    track_proximity = 0.0
    platform_band = 0.0
    if envelope_distance is not None:
        distance = float(envelope_distance)
        track_proximity = 1.0 - _clamp(
            distance / float(settings["catenary_track_proximity_falloff_m"])
        )
        platform_band = 1.0 if (
            float(settings["canopy_outside_track_minimum_m"])
            <= distance
            <= float(settings["canopy_outside_track_maximum_m"])
        ) else 0.0
    cable_relation = 0.0
    if cable.get("available"):
        cross_difference = float(cable["cross_difference_m"])
        top_difference = float(cable["top_elevation_difference_m"])
        if (
            cross_difference <= float(settings["cable_relation_maximum_cross_gap_m"])
            and float(settings["cable_relation_minimum_top_difference_m"])
            <= top_difference
            <= float(settings["cable_relation_maximum_top_difference_m"])
        ):
            cable_relation = _clamp(float(cable["longitudinal_coverage"]) / 0.5)
    periodic = 1.0 if lane["periodic_column_grid_candidate"] else 0.0
    dense_edge = 1.0 if lane["dense_surface_edge_candidate"] else 0.0
    isolated = 1.0 if int(lane["candidate_count"]) <= 2 else 0.0
    sparse = 1.0 - _clamp(
        float(candidate["source_point_count_sum"])
        / float(settings["well_supported_candidate_points"])
    )

    raw_scores = {
        "catenary_support": (
            0.45 * vertical_support
            + 0.25 * catenary_height
            + 0.20 * track_proximity
            + 0.10 * cable_relation
        ),
        "canopy_column": (
            0.45 * vertical_support
            + 0.25 * periodic
            + 0.15 * canopy_height
            + 0.15 * platform_band
        ),
        "building_edge": (
            0.35 * dense_edge
            + 0.30 * low_occupancy
            + 0.20 * _clamp(float(features["linearity"]))
            + 0.15 * (1.0 - isolated)
        ),
        "false_positive": (
            0.45 * low_occupancy
            + 0.25 * isolated
            + 0.15 * sparse
            + 0.15 * (1.0 - dense_edge)
        ),
    }
    total = max(sum(raw_scores.values()), 1e-12)
    scores = {name: float(raw_scores[name] / total) for name in CLASS_NAMES}
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    predicted = ordered[0][0]
    margin = ordered[0][1] - ordered[1][1]
    if ordered[0][1] >= 0.55 and margin >= 0.20:
        confidence = "high"
    elif ordered[0][1] >= 0.40 and margin >= 0.08:
        confidence = "medium"
    else:
        confidence = "low"

    reasons: list[str] = []
    if vertical_support >= 0.75:
        reasons.append("continuous_vertical_geometry")
    if periodic:
        reasons.append("periodic_column_grid")
    if dense_edge:
        reasons.append("dense_along-line_surface_edge_pattern")
    if track_proximity >= 0.75:
        reasons.append("inside_or_close_to_track_envelope")
    if platform_band:
        reasons.append("outside_track_platform_band")
    if cable_relation:
        reasons.append("height_and_cross_position_near_linear_candidate")
    if low_occupancy >= 0.65:
        reasons.append("poor_vertical_height_occupancy")
    if isolated:
        reasons.append("isolated_candidate")
    return predicted, scores, confidence, reasons


def analyze_vertical_candidate_data(
    linear_report: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    settings: dict[str, Any],
    rail_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if linear_report.get("schema_version") != "railway.linear-candidates.v1":
        raise ValueError("Expected railway.linear-candidates.v1")
    raw = [dict(candidate) for candidate in linear_report["vertical_candidates"]]
    tree = cKDTree(np.column_stack((x, y)))
    for candidate in raw:
        candidate["features"] = _candidate_point_features(
            candidate, x, y, z, tree, settings
        )
    groups = _merge_candidate_indexes(raw, settings)
    merged = [
        _aggregate_group(group, raw, linear_report["frame"])
        for group in groups
    ]
    for candidate in merged:
        candidate["features"] = _candidate_point_features(
            candidate, x, y, z, tree, settings
        )
    lanes = _assign_lane_features(merged, settings)
    cable_candidates = list(linear_report.get("cable_candidates", []))
    relationships: list[dict[str, Any]] = []
    for index, candidate in enumerate(merged, start=1):
        candidate["id"] = f"VERTICAL-HYPOTHESIS-{index:04d}"
        candidate["rail_context"] = _rail_context(candidate, rail_report)
        candidate["cable_context"] = _cable_context(candidate, cable_candidates)
        predicted, scores, confidence, reasons = _classify_candidate(candidate, settings)
        candidate["predicted_class"] = predicted
        candidate["class_scores"] = scores
        candidate["confidence"] = confidence
        candidate["evidence_reasons"] = reasons
        candidate["status"] = "automatic_hypothesis_review_required"
        if predicted == "catenary_support" and candidate["cable_context"]["available"]:
            relationships.append(
                {
                    "id": f"RELATIONSHIP-{len(relationships) + 1:04d}",
                    "type": "support_candidate_to_linear_candidate",
                    "source_id": candidate["id"],
                    "target_id": candidate["cable_context"]["nearest_cable_id"],
                    "status": "automatic_hypothesis_review_required",
                }
            )
        if predicted == "canopy_column" and candidate["lane"][
            "periodic_column_grid_candidate"
        ]:
            relationships.append(
                {
                    "id": f"RELATIONSHIP-{len(relationships) + 1:04d}",
                    "type": "member_of_canopy_column_grid_candidate",
                    "source_id": candidate["id"],
                    "target_id": candidate["lane"]["id"],
                    "status": "automatic_hypothesis_review_required",
                }
            )

    class_counts = Counter(str(candidate["predicted_class"]) for candidate in merged)
    confidence_counts = Counter(str(candidate["confidence"]) for candidate in merged)
    rail_lines = _rail_line_summary(rail_report, linear_report["frame"])
    return {
        "schema_version": "railway.vertical-hypotheses.v1",
        "project_id": linear_report.get("project_id"),
        "segment_id": linear_report["segment_id"],
        "source": linear_report["source"],
        "linear_candidate_report": None,
        "rail_candidate_report": None,
        "frame": linear_report["frame"],
        "rail_lines": rail_lines,
        "raw_candidate_count": len(raw),
        "merged_candidate_count": len(merged),
        "merged_source_candidate_count": sum(
            max(0, int(candidate["source_candidate_count"]) - 1) for candidate in merged
        ),
        "class_counts": {name: int(class_counts.get(name, 0)) for name in CLASS_NAMES},
        "confidence_counts": {
            name: int(confidence_counts.get(name, 0)) for name in ("high", "medium", "low")
        },
        "lane_count": len(lanes),
        "lanes": lanes,
        "candidate_count": len(merged),
        "candidates": merged,
        "relationship_count": len(relationships),
        "relationships": relationships,
        "settings": settings,
        "status": "automatic_hypotheses_only_no_asset_registry_write",
        "limitations": [
            "Four-class labels are geometric hypotheses, not reviewed asset truth.",
            "A roof candidate is not available yet, so column-to-beam-to-roof contact is unresolved.",
            "Linear candidates may include canopy ribs; cable proximity alone is not semantic proof.",
        ],
    }


def build_vertical_candidate_graph(report: dict[str, Any]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for rail in report.get("rail_lines", []):
        nodes.append(
            {
                "id": rail["id"],
                "node_type": "rail_line_candidate",
                "cross_position_m": rail["cross_position_m"],
                "status": "geometry_candidate",
            }
        )
    for lane in report["lanes"]:
        nodes.append(
            {
                "id": lane["id"],
                "node_type": "vertical_lane_candidate",
                "cross_position_m": lane["cross_position_m"],
                "periodic_column_grid_candidate": lane[
                    "periodic_column_grid_candidate"
                ],
                "periodic_pattern_confidence": lane[
                    "periodic_pattern_confidence"
                ],
                "dense_surface_edge_candidate": lane[
                    "dense_surface_edge_candidate"
                ],
                "status": "automatic_hypothesis_review_required",
            }
        )
    cable_nodes: set[str] = set()
    for candidate in report["candidates"]:
        nodes.append(
            {
                "id": candidate["id"],
                "node_type": "vertical_asset_hypothesis",
                "predicted_class": candidate["predicted_class"],
                "confidence": candidate["confidence"],
                "longitudinal_position_m": candidate["longitudinal_position_m"],
                "cross_position_m": candidate["cross_position_m"],
                "source_candidate_ids": candidate["source_candidate_ids"],
                "status": candidate["status"],
            }
        )
        edges.append(
            {
                "id": f"EDGE-{len(edges) + 1:04d}",
                "type": "member_of_vertical_lane_candidate",
                "source_id": candidate["id"],
                "target_id": candidate["lane"]["id"],
                "status": "automatic_hypothesis_review_required",
            }
        )
        cable = candidate["cable_context"]
        if candidate["predicted_class"] == "catenary_support" and cable["available"]:
            cable_id = str(cable["nearest_cable_id"])
            if cable_id not in cable_nodes:
                nodes.append(
                    {
                        "id": cable_id,
                        "node_type": "linear_asset_hypothesis",
                        "status": "geometry_candidate",
                    }
                )
                cable_nodes.add(cable_id)
            edges.append(
                {
                    "id": f"EDGE-{len(edges) + 1:04d}",
                    "type": "potential_supports_linear_candidate",
                    "source_id": candidate["id"],
                    "target_id": cable_id,
                    "cross_gap_m": cable["cross_difference_m"],
                    "top_elevation_difference_m": cable[
                        "top_elevation_difference_m"
                    ],
                    "status": "automatic_hypothesis_review_required",
                }
            )
        rail = candidate["rail_context"]
        if candidate["predicted_class"] == "catenary_support" and rail["available"]:
            edges.append(
                {
                    "id": f"EDGE-{len(edges) + 1:04d}",
                    "type": "nearest_rail_line_candidate",
                    "source_id": candidate["id"],
                    "target_id": rail["nearest_rail_id"],
                    "distance_m": rail["nearest_rail_distance_m"],
                    "status": "automatic_hypothesis_review_required",
                }
            )
    return {
        "schema_version": "railway.candidate-asset-evidence-graph.v1",
        "project_id": report.get("project_id"),
        "segment_id": report["segment_id"],
        "source_vertical_hypotheses": report.get("output_report"),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
        "status": "candidate_graph_only_no_asset_registry_write",
    }


def _render_diagnostic(report: dict[str, Any], output: Path) -> None:
    width, height = 1600, 900
    margin_left, margin_right, margin_top, margin_bottom = 120, 80, 100, 100
    image = Image.new("RGB", (width, height), "#06141d")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    candidates = report["candidates"]
    if not candidates:
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
        return
    longitudinal = [float(item["longitudinal_position_m"]) for item in candidates]
    cross = [float(item["cross_position_m"]) for item in candidates]
    s_min, s_max = min(longitudinal), max(longitudinal)
    c_min, c_max = min(cross), max(cross)
    s_pad = max(2.0, (s_max - s_min) * 0.04)
    c_pad = max(2.0, (c_max - c_min) * 0.04)
    s_min -= s_pad
    s_max += s_pad
    c_min -= c_pad
    c_max += c_pad

    def pixel(s_value: float, c_value: float) -> tuple[int, int]:
        px = margin_left + int(
            (s_value - s_min) / max(s_max - s_min, 1e-9)
            * (width - margin_left - margin_right)
        )
        py = height - margin_bottom - int(
            (c_value - c_min) / max(c_max - c_min, 1e-9)
            * (height - margin_top - margin_bottom)
        )
        return px, py

    for lane in report["lanes"]:
        if lane["periodic_column_grid_candidate"]:
            _, py = pixel(s_min, float(lane["cross_position_m"]))
            draw.line((margin_left, py, width - margin_right, py), fill="#376d78", width=2)
    for rail in report.get("rail_lines", []):
        _, py = pixel(s_min, float(rail["cross_position_m"]))
        draw.line((margin_left, py, width - margin_right, py), fill="#56666d", width=2)
    colors = {
        "catenary_support": "#29d3ff",
        "canopy_column": "#b7ff3c",
        "building_edge": "#a98cff",
        "false_positive": "#ff7a35",
    }
    for item in candidates:
        px, py = pixel(
            float(item["longitudinal_position_m"]), float(item["cross_position_m"])
        )
        radius = {"high": 8, "medium": 6, "low": 4}[str(item["confidence"])]
        color = colors[str(item["predicted_class"])]
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)
        if item["confidence"] == "high":
            draw.text((px + 9, py - 8), str(item["id"])[-4:], fill="#d8edf2", font=font)
    title = (
        f"{report['segment_id']} vertical hypotheses | "
        f"raw {report['raw_candidate_count']} -> merged {report['merged_candidate_count']}"
    )
    draw.text((margin_left, 35), title, fill="#eaf9ff", font=font)
    x = margin_left
    for name in CLASS_NAMES:
        draw.rectangle((x, 66, x + 12, 78), fill=colors[name])
        label = f" {name}: {report['class_counts'][name]}"
        draw.text((x + 15, 64), label, fill="#b8d1d8", font=font)
        x += 245
    draw.text(
        (margin_left, height - 45),
        "Automatic hypotheses only; low/medium confidence stays outside the asset registry.",
        fill="#829ca6",
        font=font,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def analyze_vertical_hypotheses(
    project: ProjectConfig,
    segment_id: str,
    overwrite: bool = False,
    linear_report_path: str | Path | None = None,
    rail_report_path: str | Path | None = None,
    settings_path: str | Path | None = None,
) -> dict[str, Any]:
    linear_path = (
        project.resolve(linear_report_path)
        if linear_report_path
        else project.workspace_path("reports") / f"{segment_id}_linear_candidates.json"
    )
    if not linear_path.is_file():
        raise FileNotFoundError(linear_path)
    linear_report = load_json(linear_path)
    source_path = Path(linear_report["source"])
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    if settings_path:
        resolved_settings_path = project.resolve(settings_path)
        settings = load_json(resolved_settings_path)
    else:
        configured = project.value.get("algorithms", {}).get("vertical_hypotheses")
        if configured:
            resolved_settings_path = project.resolve(configured)
            settings = load_json(resolved_settings_path)
        else:
            resolved_settings_path = None
            settings = _resource_settings()

    if rail_report_path:
        resolved_rail_path = project.resolve(rail_report_path)
    else:
        candidate_rail_path = (
            project.workspace_path("reports") / f"{segment_id}_rail_candidates.json"
        )
        resolved_rail_path = candidate_rail_path if candidate_rail_path.is_file() else None
    rail_report = load_json(resolved_rail_path) if resolved_rail_path else None

    output_json = (
        project.workspace_path("reports") / f"{segment_id}_vertical_hypotheses.json"
    )
    output_image = (
        project.workspace_path("reports") / f"{segment_id}_vertical_hypotheses.png"
    )
    output_graph = (
        project.workspace_path("reports")
        / f"{segment_id}_vertical_candidate_asset_graph.json"
    )
    for path in (output_json, output_image, output_graph):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite: {path}")

    cloud = laspy.read(source_path)
    x = np.asarray(cloud.x, dtype=np.float64)
    y = np.asarray(cloud.y, dtype=np.float64)
    z = np.asarray(cloud.z, dtype=np.float64)
    report = analyze_vertical_candidate_data(
        linear_report, x, y, z, settings, rail_report=rail_report
    )
    report["linear_candidate_report"] = str(linear_path)
    report["rail_candidate_report"] = str(resolved_rail_path) if resolved_rail_path else None
    report["settings_source"] = (
        str(resolved_settings_path) if resolved_settings_path else "bundled_default"
    )
    report["output_report"] = str(output_json)
    report["output_diagnostic_image"] = str(output_image)
    report["output_candidate_asset_graph"] = str(output_graph)
    write_json(output_json, report)
    write_json(output_graph, build_vertical_candidate_graph(report))
    _render_diagnostic(report, output_image)
    return report
