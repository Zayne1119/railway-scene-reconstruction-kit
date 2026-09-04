from __future__ import annotations

from pathlib import Path
from typing import Any

import laspy
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.spatial import cKDTree

from .io import load_json, write_json
from .model_point_support import ObjModel, object_vertex_indices, parse_obj_model


def triangulate_model(model: ObjModel) -> np.ndarray:
    triangles: list[np.ndarray] = []
    for faces in model.faces_by_object.values():
        for face in faces:
            anchor = model.vertices[face[0]]
            for index in range(1, len(face) - 1):
                triangles.append(
                    np.asarray(
                        (anchor, model.vertices[face[index]], model.vertices[face[index + 1]]),
                        dtype=np.float64,
                    )
                )
    if not triangles:
        raise ValueError("Model contains no triangulatable faces")
    return np.stack(triangles)


def sample_model_surfaces(
    model: ObjModel,
    *,
    origin_xyz: np.ndarray,
    spacing_m: float = 0.15,
    random_seed: int = 20260902,
) -> tuple[np.ndarray, dict[str, Any]]:
    if spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive")
    triangles = triangulate_model(model)
    edge_ab = triangles[:, 1] - triangles[:, 0]
    edge_ac = triangles[:, 2] - triangles[:, 0]
    areas = np.linalg.norm(np.cross(edge_ab, edge_ac), axis=1) / 2.0
    valid = areas > 1.0e-12
    triangles = triangles[valid]
    areas = areas[valid]
    samples_per_triangle = np.maximum(
        1,
        np.ceil(areas / (spacing_m * spacing_m * 0.5)).astype(np.int64),
    )
    total = int(np.sum(samples_per_triangle))
    sampled = np.empty((total, 3), dtype=np.float64)
    rng = np.random.default_rng(random_seed)
    offset = 0
    for triangle, count in zip(triangles, samples_per_triangle, strict=True):
        count_value = int(count)
        root = np.sqrt(rng.random(count_value))
        second = rng.random(count_value)
        sampled[offset : offset + count_value] = (
            (1.0 - root)[:, None] * triangle[0]
            + (root * (1.0 - second))[:, None] * triangle[1]
            + (root * second)[:, None] * triangle[2]
        )
        offset += count_value
    origin = np.asarray(origin_xyz, dtype=np.float64)
    sampled += origin
    return sampled, {
        "triangle_count": len(triangles),
        "surface_area_m2": float(np.sum(areas)),
        "surface_sample_count": len(sampled),
        "target_spacing_m": spacing_m,
        "random_seed": random_seed,
    }


def load_cloud_sample_rgb(
    path: str | Path,
    *,
    target_point_count: int = 4_000_000,
    chunk_size: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    if target_point_count <= 0:
        raise ValueError("target_point_count must be positive")
    source = Path(path).resolve()
    point_chunks: list[np.ndarray] = []
    color_chunks: list[np.ndarray] = []
    offset = 0
    with laspy.open(source) as reader:
        source_count = int(reader.header.point_count)
        stride = max(1, int(np.ceil(source_count / target_point_count)))
        dimensions = set(reader.header.point_format.dimension_names)
        has_rgb = {"red", "green", "blue"}.issubset(dimensions)
        for chunk in reader.chunk_iterator(chunk_size):
            start = (-offset) % stride
            x = np.asarray(chunk.x, dtype=np.float64)[start::stride]
            y = np.asarray(chunk.y, dtype=np.float64)[start::stride]
            z = np.asarray(chunk.z, dtype=np.float64)[start::stride]
            point_chunks.append(np.column_stack((x, y, z)))
            if has_rgb:
                color = np.column_stack(
                    (
                        np.asarray(chunk.red, dtype=np.uint16)[start::stride],
                        np.asarray(chunk.green, dtype=np.uint16)[start::stride],
                        np.asarray(chunk.blue, dtype=np.uint16)[start::stride],
                    )
                )
                if color.size and int(np.max(color)) > 255:
                    color = color // 256
                color_chunks.append(color.astype(np.uint8))
            else:
                color_chunks.append(np.full((len(x), 3), 180, dtype=np.uint8))
            offset += len(chunk)
    points = np.vstack(point_chunks) if point_chunks else np.empty((0, 3))
    colors = np.vstack(color_chunks) if color_chunks else np.empty((0, 3), dtype=np.uint8)
    return points, colors, {
        "source_point_count": source_count,
        "sample_point_count": len(points),
        "systematic_stride": stride,
    }


def reverse_model_distances(
    cloud_points: np.ndarray,
    model_surface_samples: np.ndarray,
) -> np.ndarray:
    if not len(cloud_points) or not len(model_surface_samples):
        raise ValueError("Cloud and model samples must be non-empty")
    distances, _ = cKDTree(model_surface_samples).query(cloud_points, workers=-1)
    return np.asarray(distances, dtype=np.float64)


def assign_corridor_scope_ownership(
    candidates: list[dict[str, Any]],
    linear_candidates: list[dict[str, Any]],
    *,
    model: ObjModel,
    origin_xyz: np.ndarray,
    frame: dict[str, Any],
    boundary_tolerance_m: float = 0.5,
) -> dict[str, Any]:
    # Scene composition prefixes objects with one or more namespaces.  Accept
    # both original TRACK* nodes and composed ``...--TRACK-*`` /
    # ``...--TRACKGRAPH--TRACK-*`` nodes, while avoiding names such as
    # CATENARY-WIRE-BRIDGE-TRACK-* that are not track geometry.
    track_names = [
        name
        for name in model.faces_by_object
        if name.startswith("TRACK")
        or "--TRACK-" in name
        or "--TRACKGRAPH--TRACK-" in name
    ]
    if not track_names:
        raise ValueError("Cannot derive corridor ownership without TRACK* objects")
    track_indexes = np.unique(
        np.concatenate([object_vertex_indices(model, name) for name in track_names])
    )
    track_xy = (model.vertices[track_indexes] + origin_xyz)[:, :2]
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    track_station = (track_xy - frame_origin) @ along
    minimum = float(np.min(track_station))
    maximum = float(np.max(track_station))
    counts = {
        "within_current_segment": 0,
        "adjacent_previous_segment": 0,
        "adjacent_next_segment": 0,
    }
    for candidate in candidates:
        station = float(candidate["station_m"])
        if station < minimum - boundary_tolerance_m:
            ownership = "adjacent_previous_segment"
        elif station > maximum + boundary_tolerance_m:
            ownership = "adjacent_next_segment"
        else:
            ownership = "within_current_segment"
        candidate["corridor_scope_ownership"] = ownership
        candidate["build_eligible_current_segment"] = ownership == "within_current_segment"
        counts[ownership] += 1
    linear_counts = {
        "within_current_segment": 0,
        "crosses_segment_boundary": 0,
        "outside_current_segment": 0,
    }
    for candidate in linear_candidates:
        start, end = (float(value) for value in candidate["station_range_m"])
        if start >= minimum - boundary_tolerance_m and end <= maximum + boundary_tolerance_m:
            ownership = "within_current_segment"
        elif end < minimum - boundary_tolerance_m or start > maximum + boundary_tolerance_m:
            ownership = "outside_current_segment"
        else:
            ownership = "crosses_segment_boundary"
        candidate["corridor_scope_ownership"] = ownership
        candidate["build_eligible_current_segment"] = ownership != "outside_current_segment"
        linear_counts[ownership] += 1
    return {
        "station_minimum_m": minimum,
        "station_maximum_m": maximum,
        "length_m": maximum - minimum,
        "boundary_tolerance_m": boundary_tolerance_m,
        "vertical_ownership_counts": counts,
        "linear_ownership_counts": linear_counts,
    }


def _column_statistics(
    points: np.ndarray,
    *,
    minimum_xy: np.ndarray,
    cell_size_m: float,
    vertical_bin_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cell_xy = np.floor((points[:, :2] - minimum_xy) / cell_size_m).astype(np.int64)
    width = int(np.max(cell_xy[:, 0])) + 1
    height = int(np.max(cell_xy[:, 1])) + 1
    keys = cell_xy[:, 1] * width + cell_xy[:, 0]
    unique_keys, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    minimum_z = np.full(len(unique_keys), np.inf, dtype=np.float64)
    maximum_z = np.full(len(unique_keys), -np.inf, dtype=np.float64)
    np.minimum.at(minimum_z, inverse, points[:, 2])
    np.maximum.at(maximum_z, inverse, points[:, 2])
    global_min_z = float(np.min(points[:, 2]))
    z_index = np.floor((points[:, 2] - global_min_z) / vertical_bin_m).astype(np.int64)
    voxel_keys = keys.astype(np.int64) * (int(np.max(z_index)) + 2) + z_index
    unique_voxels = np.unique(voxel_keys)
    voxel_cell_keys = unique_voxels // (int(np.max(z_index)) + 2)
    voxel_positions = np.searchsorted(unique_keys, voxel_cell_keys)
    occupied_bins = np.bincount(voxel_positions, minlength=len(unique_keys))
    return unique_keys, counts, minimum_z, maximum_z, occupied_bins, np.asarray(
        (height, width), dtype=np.int64
    )


def detect_vertical_gap_candidates(
    points: np.ndarray,
    distances: np.ndarray,
    *,
    unexplained_distance_m: float = 0.25,
    cell_size_m: float = 0.25,
    vertical_bin_m: float = 0.25,
    minimum_height_m: float = 1.5,
    minimum_sample_points_per_cell: int = 12,
    minimum_vertical_occupancy: float = 0.35,
) -> list[dict[str, Any]]:
    unexplained = distances > unexplained_distance_m
    selected_points = points[unexplained]
    selected_distances = distances[unexplained]
    if not len(selected_points):
        return []
    minimum_xy = np.min(points[:, :2], axis=0)
    (
        keys,
        counts,
        minimum_z,
        maximum_z,
        occupied_bins,
        shape,
    ) = _column_statistics(
        selected_points,
        minimum_xy=minimum_xy,
        cell_size_m=cell_size_m,
        vertical_bin_m=vertical_bin_m,
    )
    spans = maximum_z - minimum_z
    expected_bins = np.maximum(1, np.ceil(spans / vertical_bin_m).astype(np.int64) + 1)
    occupancy = occupied_bins / expected_bins
    candidate_cells = (
        (counts >= minimum_sample_points_per_cell)
        & (spans >= minimum_height_m)
        & (occupancy >= minimum_vertical_occupancy)
    )
    mask = np.zeros(tuple(int(value) for value in shape), dtype=bool)
    width = int(shape[1])
    candidate_keys = keys[candidate_cells]
    mask[candidate_keys // width, candidate_keys % width] = True
    labels, component_count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    point_cell = np.floor((selected_points[:, :2] - minimum_xy) / cell_size_m).astype(
        np.int64
    )
    point_labels = labels[point_cell[:, 1], point_cell[:, 0]]
    candidates: list[dict[str, Any]] = []
    for component in range(1, component_count + 1):
        member = point_labels == component
        if not np.any(member):
            continue
        component_points = selected_points[member]
        component_distances = selected_distances[member]
        minimum = np.min(component_points, axis=0)
        maximum = np.max(component_points, axis=0)
        extent = maximum - minimum
        footprint_max = float(np.max(extent[:2]))
        footprint_min = float(np.min(extent[:2]))
        vertical_bins = np.unique(
            np.floor((component_points[:, 2] - minimum[2]) / vertical_bin_m).astype(np.int64)
        )
        vertical_occupancy = len(vertical_bins) / max(
            1, int(np.ceil(extent[2] / vertical_bin_m)) + 1
        )
        if (
            extent[2] >= 3.0
            and footprint_max <= 1.25
            and vertical_occupancy >= 0.70
            and len(component_points) >= 40
        ):
            classification = "high_confidence_pole_or_mast"
            priority = "P0"
        elif extent[2] >= 2.0 and footprint_max <= 2.0:
            classification = "vertical_asset_candidate"
            priority = "P1"
        elif extent[2] >= 2.0 and footprint_min <= 0.75:
            classification = "facade_edge_fence_or_grouped_vertical"
            priority = "P2"
        else:
            classification = "large_vertical_surface_or_mixed_cluster"
            priority = "P3"
        candidates.append(
            {
                "candidate_id": "",
                "classification": classification,
                "priority": priority,
                "sample_point_count": len(component_points),
                "minimum_xyz_m": [float(value) for value in minimum],
                "maximum_xyz_m": [float(value) for value in maximum],
                "centroid_xyz_m": [
                    float(value) for value in np.median(component_points, axis=0)
                ],
                "extent_xyz_m": [float(value) for value in extent],
                "vertical_occupancy": float(vertical_occupancy),
                "distance_to_model_p50_m": float(np.percentile(component_distances, 50)),
                "distance_to_model_p90_m": float(np.percentile(component_distances, 90)),
            }
        )
    candidates.sort(
        key=lambda item: (
            str(item["priority"]),
            -float(item["extent_xyz_m"][2]),
            -int(item["sample_point_count"]),
        )
    )
    for index, candidate in enumerate(candidates, start=1):
        candidate["candidate_id"] = f"GAP-VERTICAL-{index:04d}"
    return candidates


def match_vertical_candidates_to_assets(
    candidates: list[dict[str, Any]],
    *,
    model: ObjModel,
    origin_xyz: np.ndarray,
    registry: dict[str, Any],
    frame: dict[str, Any] | None = None,
    direct_match_distance_m: float = 0.35,
    nearby_match_distance_m: float = 0.90,
) -> dict[str, int]:
    vertical_types = {
        "canopy_column",
        "catenary_mast",
        "catenary_foundation",
        "station_information_sign",
    }
    references: list[dict[str, Any]] = []
    bounded_surface_references: list[dict[str, Any]] = []
    origin = np.asarray(origin_xyz, dtype=np.float64)
    frame_origin = (
        np.asarray(frame["origin_xy"], dtype=np.float64) if frame is not None else None
    )
    along = np.asarray(frame["along_xy"], dtype=np.float64) if frame is not None else None
    cross = np.asarray(frame["cross_xy"], dtype=np.float64) if frame is not None else None
    for asset in registry.get("assets", []):
        parameters = asset.get("parameters")
        asset_type = str(asset.get("type", ""))
        if asset_type.startswith("station_entry_") and isinstance(parameters, dict):
            station_range = parameters.get("station_range_m")
            cross_range = parameters.get("cross_range_m")
            if (
                isinstance(station_range, list | tuple)
                and len(station_range) == 2
                and isinstance(cross_range, list | tuple)
                and len(cross_range) == 2
            ):
                bounded_surface_references.append(
                    {
                        "asset_id": str(asset.get("id", "")),
                        "asset_type": asset_type,
                        "station_range_m": [float(value) for value in station_range],
                        "cross_range_m": [float(value) for value in cross_range],
                    }
                )
        if asset.get("type") not in vertical_types:
            continue
        geometry = asset.get("geometry")
        object_name = geometry.get("node") if isinstance(geometry, dict) else None
        object_name = str(object_name or asset.get("id", ""))
        if object_name not in model.faces_by_object:
            continue
        indices = object_vertex_indices(model, object_name)
        if not len(indices):
            continue
        points = model.vertices[indices] + origin
        center_xy = np.mean(points[:, :2], axis=0)
        local = center_xy - frame_origin if frame_origin is not None else None
        references.append(
            {
                "asset_id": str(asset.get("id", object_name)),
                "asset_type": str(asset.get("type")),
                "minimum_xy_m": np.min(points[:, :2], axis=0),
                "maximum_xy_m": np.max(points[:, :2], axis=0),
                "station_m": float(local @ along) if local is not None else None,
                "cross_m": float(local @ cross) if local is not None else None,
            }
        )
    relation_counts: dict[str, int] = {}
    for candidate in candidates:
        minimum = np.asarray(candidate["minimum_xyz_m"][:2], dtype=np.float64)
        maximum = np.asarray(candidate["maximum_xyz_m"][:2], dtype=np.float64)
        center = np.asarray(candidate["centroid_xyz_m"][:2], dtype=np.float64)
        candidate_local = center - frame_origin if frame_origin is not None else None
        if candidate_local is not None:
            candidate["station_m"] = float(candidate_local @ along)
            candidate["cross_m"] = float(candidate_local @ cross)
        matches: list[tuple[float, dict[str, Any]]] = []
        for reference in references:
            delta = np.maximum(
                np.maximum(reference["minimum_xy_m"] - maximum, minimum - reference["maximum_xy_m"]),
                0.0,
            )
            matches.append((float(np.linalg.norm(delta)), reference))
        if matches:
            distance, nearest = min(matches, key=lambda item: item[0])
            candidate["nearest_existing_vertical_asset"] = {
                "asset_id": nearest["asset_id"],
                "asset_type": nearest["asset_type"],
                "footprint_distance_m": distance,
            }
        else:
            distance = float("inf")
            candidate["nearest_existing_vertical_asset"] = None
        aligned_reference = None
        nearest_bounded_reference = None
        bounded_distance = float("inf")
        if candidate_local is not None:
            bounded_matches: list[tuple[float, dict[str, Any]]] = []
            for reference in bounded_surface_references:
                station_range = reference["station_range_m"]
                cross_range = reference["cross_range_m"]
                station_delta = max(
                    station_range[0] - candidate["station_m"],
                    candidate["station_m"] - station_range[1],
                    0.0,
                )
                cross_delta = max(
                    cross_range[0] - candidate["cross_m"],
                    candidate["cross_m"] - cross_range[1],
                    0.0,
                )
                bounded_matches.append(
                    (float(np.hypot(station_delta, cross_delta)), reference)
                )
            if bounded_matches:
                bounded_distance, nearest_bounded_reference = min(
                    bounded_matches, key=lambda item: item[0]
                )
                candidate["nearest_existing_bounded_surface_asset"] = {
                    "asset_id": nearest_bounded_reference["asset_id"],
                    "asset_type": nearest_bounded_reference["asset_type"],
                    "station_cross_distance_m": bounded_distance,
                }
            aligned: list[tuple[float, dict[str, Any]]] = []
            for reference in references:
                if reference["asset_type"] != "catenary_mast":
                    continue
                station_delta = abs(candidate["station_m"] - reference["station_m"])
                cross_delta = abs(candidate["cross_m"] - reference["cross_m"])
                if station_delta <= 1.25 and cross_delta <= 2.50:
                    aligned.append((station_delta + cross_delta, reference))
            if aligned:
                _, aligned_reference = min(aligned, key=lambda item: item[0])
        if distance <= direct_match_distance_m or bounded_distance <= direct_match_distance_m:
            relation = "existing_asset_geometry_refinement"
        elif aligned_reference is not None:
            relation = "existing_asset_position_or_detail_refinement"
            candidate["aligned_existing_catenary_asset"] = {
                "asset_id": aligned_reference["asset_id"],
                "station_delta_m": abs(
                    candidate["station_m"] - aligned_reference["station_m"]
                ),
                "cross_delta_m": abs(candidate["cross_m"] - aligned_reference["cross_m"]),
            }
        elif distance <= nearby_match_distance_m or bounded_distance <= 1.50:
            relation = "possible_attachment_or_position_refinement"
        else:
            relation = "new_standalone_vertical_candidate"
        candidate["asset_relation"] = relation
        aligned = candidate.get("aligned_existing_catenary_asset")
        extent = candidate.get("extent_xyz_m") or (float("inf"),) * 3
        station_entry_surface_residual = (
            nearest_bounded_reference is not None
            and str(nearest_bounded_reference["asset_type"]).startswith("station_entry_")
            and (
                (
                    candidate.get("priority") == "P1"
                    and bounded_distance <= 1.50
                )
                or (
                    candidate.get("priority") == "P0"
                    and bounded_distance <= direct_match_distance_m
                )
            )
        )
        if station_entry_surface_residual:
            candidate["priority_before_relation_review"] = candidate["priority"]
            candidate["priority"] = "P2"
            candidate["semantic_review_hint"] = (
                "existing_station_entry_surface_edge_or_residual"
            )
            candidate["automatic_geometry_action"] = (
                "refine_existing_station_entry_group_not_new_standalone_asset"
            )
        if (
            relation == "existing_asset_position_or_detail_refinement"
            and isinstance(aligned, dict)
            and candidate.get("priority") == "P1"
            and float(aligned["station_delta_m"]) <= 0.15
            and 1.50 <= float(aligned["cross_delta_m"]) <= 2.25
            and 3.50 <= float(extent[2]) <= 6.50
            and max(float(extent[0]), float(extent[1])) <= 0.65
            and float(candidate.get("vertical_occupancy", 1.0)) <= 0.60
        ):
            # Repeated discontinuous traces at the same station as a catenary mast
            # are commonly a mix of cantilever hardware, wires and scan companions.
            # Keep them reviewable, but do not promote them as a missing mast.
            candidate["priority_before_relation_review"] = "P1"
            candidate["priority"] = "P2"
            candidate["semantic_review_hint"] = (
                "station_aligned_catenary_companion_trace_or_attachment"
            )
            candidate["automatic_geometry_action"] = "withhold_pending_photo_review"
        relation_counts[relation] = relation_counts.get(relation, 0) + 1
    return relation_counts


def detect_overhead_linear_gap_candidates(
    points: np.ndarray,
    distances: np.ndarray,
    frame: dict[str, Any],
    *,
    unexplained_distance_m: float = 0.25,
    minimum_z_m: float = 25.5,
    voxel_size_m: float = 0.25,
    minimum_station_span_m: float = 5.0,
    minimum_sample_points: int = 30,
) -> list[dict[str, Any]]:
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    selected = (distances > unexplained_distance_m) & (points[:, 2] >= minimum_z_m)
    source_indexes = np.flatnonzero(selected)
    if not len(source_indexes):
        return []
    selected_points = points[source_indexes]
    local_xy = selected_points[:, :2] - frame_origin
    local = np.column_stack(
        (
            local_xy @ along,
            local_xy @ cross,
            selected_points[:, 2],
        )
    )
    minimum = np.min(local, axis=0)
    voxel = np.floor((local - minimum) / voxel_size_m).astype(np.int64)
    shape = np.max(voxel, axis=0) + 1
    occupancy = np.zeros(tuple(int(value) for value in shape), dtype=bool)
    occupancy[voxel[:, 0], voxel[:, 1], voxel[:, 2]] = True
    labels, component_count = ndimage.label(
        occupancy,
        structure=np.ones((3, 3, 3), dtype=np.uint8),
    )
    point_labels = labels[voxel[:, 0], voxel[:, 1], voxel[:, 2]]
    candidates: list[dict[str, Any]] = []
    for component in range(1, component_count + 1):
        member = point_labels == component
        point_count = int(np.sum(member))
        if point_count < minimum_sample_points:
            continue
        component_local = local[member]
        component_global = selected_points[member]
        component_distances = distances[source_indexes[member]]
        local_minimum = np.min(component_local, axis=0)
        local_maximum = np.max(component_local, axis=0)
        extent = local_maximum - local_minimum
        if extent[0] < minimum_station_span_m:
            continue
        if extent[1] <= 1.0 and extent[2] <= 1.5:
            classification = "overhead_wire_or_linear_asset"
            priority = "P0" if extent[0] >= 15.0 and point_count >= 100 else "P1"
        elif extent[1] <= 2.0 and extent[2] <= 2.5:
            classification = "roof_edge_or_grouped_linear_asset"
            priority = "P2"
        else:
            continue
        candidates.append(
            {
                "candidate_id": "",
                "classification": classification,
                "priority": priority,
                "sample_point_count": point_count,
                "station_range_m": [float(local_minimum[0]), float(local_maximum[0])],
                "cross_range_m": [float(local_minimum[1]), float(local_maximum[1])],
                "z_range_m": [float(local_minimum[2]), float(local_maximum[2])],
                "extent_station_cross_z_m": [float(value) for value in extent],
                "centroid_xyz_m": [
                    float(value) for value in np.median(component_global, axis=0)
                ],
                "distance_to_model_p50_m": float(np.percentile(component_distances, 50)),
                "distance_to_model_p90_m": float(np.percentile(component_distances, 90)),
            }
        )
    candidates.sort(
        key=lambda item: (
            str(item["priority"]),
            -float(item["extent_station_cross_z_m"][0]),
            -int(item["sample_point_count"]),
        )
    )
    for index, candidate in enumerate(candidates, start=1):
        candidate["candidate_id"] = f"GAP-LINEAR-{index:04d}"
    return candidates


def match_linear_candidates_to_assets(
    candidates: list[dict[str, Any]],
    *,
    model: ObjModel,
    origin_xyz: np.ndarray,
    registry: dict[str, Any],
    frame: dict[str, Any],
    roof_edge_cross_tolerance_m: float = 0.35,
    roof_vertical_tolerance_m: float = 0.35,
) -> dict[str, int]:
    """Demote canopy edge/fascia traces before treating them as missing wires."""
    origin = np.asarray(origin_xyz, dtype=np.float64)
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    roof_references: list[dict[str, Any]] = []
    for asset in registry.get("assets", []):
        if asset.get("type") != "canopy_roof_surface":
            continue
        geometry = asset.get("geometry")
        object_name = geometry.get("node") if isinstance(geometry, dict) else None
        object_name = str(object_name or asset.get("id", ""))
        if object_name not in model.faces_by_object:
            continue
        indices = object_vertex_indices(model, object_name)
        if not len(indices):
            continue
        points = model.vertices[indices] + origin
        local_xy = points[:, :2] - frame_origin
        stations = local_xy @ along
        laterals = local_xy @ cross
        roof_references.append(
            {
                "asset_id": str(asset.get("id", object_name)),
                "station_range_m": [float(np.min(stations)), float(np.max(stations))],
                "cross_range_m": [float(np.min(laterals)), float(np.max(laterals))],
                "z_range_m": [float(np.min(points[:, 2])), float(np.max(points[:, 2]))],
            }
        )

    relation_counts: dict[str, int] = {}
    for candidate in candidates:
        station_range = [float(value) for value in candidate["station_range_m"]]
        cross_range = [float(value) for value in candidate["cross_range_m"]]
        z_range = [float(value) for value in candidate["z_range_m"]]
        station_span = station_range[1] - station_range[0]
        cross_center = float(np.mean(cross_range))
        matches: list[tuple[float, float, dict[str, Any]]] = []
        for reference in roof_references:
            roof_station = reference["station_range_m"]
            overlap = max(
                0.0,
                min(station_range[1], roof_station[1])
                - max(station_range[0], roof_station[0]),
            )
            if overlap < min(5.0, station_span * 0.50):
                continue
            roof_cross = reference["cross_range_m"]
            edge_distance = min(
                abs(cross_center - roof_cross[0]),
                abs(cross_center - roof_cross[1]),
            )
            roof_z = reference["z_range_m"]
            vertical_distance = max(
                roof_z[0] - z_range[1],
                z_range[0] - roof_z[1],
                0.0,
            )
            if (
                edge_distance <= roof_edge_cross_tolerance_m
                and vertical_distance <= roof_vertical_tolerance_m
            ):
                matches.append((edge_distance, vertical_distance, reference))
        if matches:
            edge_distance, vertical_distance, nearest = min(
                matches, key=lambda item: (item[0], item[1])
            )
            relation = "existing_canopy_roof_edge_or_fascia_refinement"
            candidate["nearest_existing_canopy_roof_edge"] = {
                "asset_id": nearest["asset_id"],
                "cross_edge_distance_m": edge_distance,
                "vertical_interval_distance_m": vertical_distance,
            }
            candidate["classification_before_relation_review"] = candidate[
                "classification"
            ]
            candidate["priority_before_relation_review"] = candidate["priority"]
            candidate["classification"] = "canopy_roof_edge_or_fascia_residual"
            candidate["priority"] = "P2"
            candidate["semantic_review_hint"] = (
                "existing_canopy_roof_edge_or_fascia_not_missing_conductor"
            )
            candidate["automatic_geometry_action"] = (
                "refine_existing_canopy_group_not_new_wire"
            )
        else:
            relation = "unmatched_overhead_linear_candidate"
        candidate["asset_relation"] = relation
        relation_counts[relation] = relation_counts.get(relation, 0) + 1
    return relation_counts


def _distance_summary(distances: np.ndarray) -> dict[str, float | int]:
    return {
        "sample_count": len(distances),
        "p50_m": float(np.percentile(distances, 50)),
        "p90_m": float(np.percentile(distances, 90)),
        "p95_m": float(np.percentile(distances, 95)),
        "coverage_at_0_10m": float(np.mean(distances <= 0.10)),
        "coverage_at_0_20m": float(np.mean(distances <= 0.20)),
        "unexplained_over_0_25m": float(np.mean(distances > 0.25)),
        "unexplained_over_0_50m": float(np.mean(distances > 0.50)),
    }


def render_gap_inventory(
    points: np.ndarray,
    distances: np.ndarray,
    candidates: list[dict[str, Any]],
    frame: dict[str, Any],
    output_path: str | Path,
    *,
    width: int = 1800,
    height: int = 1000,
) -> Path:
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    station = (points[:, :2] - frame_origin) @ along
    lateral = (points[:, :2] - frame_origin) @ cross
    minimum = np.asarray((np.min(station), np.min(lateral)))
    maximum = np.asarray((np.max(station), np.max(lateral)))
    span = np.maximum(maximum - minimum, 1.0e-6)
    margin = 70
    header = 100
    plot_width = width - 2 * margin
    plot_height = height - header - 2 * margin
    scale = min(plot_width / span[0], plot_height / span[1])
    center = (minimum + maximum) / 2.0
    image = Image.new("RGB", (width, height), (4, 18, 26))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=20)
    draw.text((margin, 28), "NEW CLOUD -> CURRENT MODEL / REVERSE GAP AUDIT", fill=(180, 255, 60), font=font)
    draw.text(
        (margin, 58),
        "cyan <= 0.20 m | orange 0.20-0.50 m | red > 0.50 m | rings = vertical candidates",
        fill=(178, 210, 218),
        font=font,
    )
    step = max(1, len(points) // 500_000)
    selected = np.arange(0, len(points), step, dtype=np.int64)
    plot_s = station[selected]
    plot_c = lateral[selected]
    plot_d = distances[selected]
    px = (plot_s - center[0]) * scale + width / 2.0
    py = height / 2.0 + header / 2.0 - (plot_c - center[1]) * scale
    for x_value, y_value, distance in zip(px, py, plot_d, strict=True):
        if distance <= 0.20:
            color = (36, 162, 184)
        elif distance <= 0.50:
            color = (255, 145, 40)
        else:
            color = (245, 57, 73)
        draw.point((int(x_value), int(y_value)), fill=color)
    for candidate in candidates:
        centroid = np.asarray(candidate["centroid_xyz_m"], dtype=np.float64)
        local = centroid[:2] - frame_origin
        candidate_s = float(local @ along)
        candidate_c = float(local @ cross)
        x_value = int((candidate_s - center[0]) * scale + width / 2.0)
        y_value = int(height / 2.0 + header / 2.0 - (candidate_c - center[1]) * scale)
        color = (180, 255, 60) if candidate["priority"] == "P0" else (255, 221, 66)
        radius = 8 if candidate["priority"] == "P0" else 5
        draw.ellipse(
            (x_value - radius, y_value - radius, x_value + radius, y_value + radius),
            outline=color,
            width=2,
        )
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return destination


def render_vertical_candidate_evidence(
    points: np.ndarray,
    distances: np.ndarray,
    candidates: list[dict[str, Any]],
    frame: dict[str, Any],
    output_directory: str | Path,
    *,
    candidates_per_page: int = 8,
) -> list[Path]:
    selected_candidates = [
        candidate
        for candidate in candidates
        if candidate["priority"] == "P0"
        and candidate["asset_relation"] != "existing_asset_geometry_refinement"
    ]
    if not selected_candidates:
        return []
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    station = (points[:, :2] - frame_origin) @ along
    lateral = (points[:, :2] - frame_origin) @ cross
    page_paths: list[Path] = []
    tile_width = 690
    tile_height = 330
    columns = 2
    rows = int(np.ceil(candidates_per_page / columns))
    for page_index, start in enumerate(
        range(0, len(selected_candidates), candidates_per_page), start=1
    ):
        page_candidates = selected_candidates[start : start + candidates_per_page]
        image = Image.new(
            "RGB",
            (columns * tile_width + 40, rows * tile_height + 80),
            (4, 18, 26),
        )
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=17)
        draw.text(
            (24, 20),
            f"P0 VERTICAL GAP EVIDENCE / PAGE {page_index:02d}",
            fill=(180, 255, 60),
            font=font,
        )
        for tile_index, candidate in enumerate(page_candidates):
            row = tile_index // columns
            column = tile_index % columns
            x0 = 20 + column * tile_width
            y0 = 65 + row * tile_height
            x1 = x0 + tile_width - 20
            y1 = y0 + tile_height - 18
            draw.rectangle((x0, y0, x1, y1), outline=(33, 91, 108), width=1)
            center_s = float(candidate["station_m"])
            center_c = float(candidate["cross_m"])
            minimum_z = float(candidate["minimum_xyz_m"][2]) - 0.5
            maximum_z = float(candidate["maximum_xyz_m"][2]) + 0.5
            member = (
                (np.abs(station - center_s) <= 1.5)
                & (np.abs(lateral - center_c) <= 1.5)
                & (points[:, 2] >= minimum_z)
                & (points[:, 2] <= maximum_z)
            )
            indexes = np.flatnonzero(member)
            if len(indexes) > 15_000:
                indexes = indexes[:: int(np.ceil(len(indexes) / 15_000))]
            panel_left = x0 + 14
            panel_top = y0 + 72
            panel_width = x1 - x0 - 28
            panel_height = y1 - panel_top - 14
            if len(indexes):
                plot_x = lateral[indexes]
                plot_z = points[indexes, 2]
                x_min = center_c - 1.5
                x_max = center_c + 1.5
                z_span = max(maximum_z - minimum_z, 1.0e-6)
                px = panel_left + (plot_x - x_min) / (x_max - x_min) * panel_width
                py = panel_top + (maximum_z - plot_z) / z_span * panel_height
                for x_value, y_value, distance in zip(
                    px, py, distances[indexes], strict=True
                ):
                    color = (42, 190, 210) if distance <= 0.25 else (255, 112, 45)
                    draw.point((int(x_value), int(y_value)), fill=color)
                candidate_minimum = np.asarray(candidate["minimum_xyz_m"], dtype=np.float64)
                candidate_maximum = np.asarray(candidate["maximum_xyz_m"], dtype=np.float64)
                within_candidate = (
                    (points[indexes, 0] >= candidate_minimum[0])
                    & (points[indexes, 0] <= candidate_maximum[0])
                    & (points[indexes, 1] >= candidate_minimum[1])
                    & (points[indexes, 1] <= candidate_maximum[1])
                    & (points[indexes, 2] >= candidate_minimum[2])
                    & (points[indexes, 2] <= candidate_maximum[2])
                    & (distances[indexes] > 0.25)
                )
                for x_value, y_value in zip(
                    px[within_candidate], py[within_candidate], strict=True
                ):
                    draw.ellipse(
                        (int(x_value) - 1, int(y_value) - 1, int(x_value) + 1, int(y_value) + 1),
                        fill=(180, 255, 60),
                    )
            relation = str(candidate["asset_relation"])
            draw.text(
                (x0 + 12, y0 + 10),
                (
                    f"{candidate['candidate_id']}  {relation}\n"
                    f"S={center_s:.1f} m  C={center_c:.1f} m  "
                    f"H={candidate['extent_xyz_m'][2]:.2f} m  "
                    f"sample={candidate['sample_point_count']}"
                ),
                fill=(220, 239, 244),
                font=font,
                spacing=5,
            )
            draw.line(
                (panel_left, panel_top + panel_height, panel_left + panel_width, panel_top + panel_height),
                fill=(86, 130, 142),
            )
        page_path = output / f"p0_vertical_evidence_page_{page_index:02d}.png"
        image.save(page_path)
        page_paths.append(page_path)
    return page_paths


def render_linear_candidate_evidence(
    points: np.ndarray,
    distances: np.ndarray,
    candidates: list[dict[str, Any]],
    frame: dict[str, Any],
    output_path: str | Path,
) -> Path | None:
    selected_candidates = [candidate for candidate in candidates if candidate["priority"] == "P0"]
    if not selected_candidates:
        return None
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    local_xy = points[:, :2] - frame_origin
    station = local_xy @ along
    lateral = local_xy @ cross
    width = 1800
    row_height = 430
    image = Image.new(
        "RGB",
        (width, 80 + row_height * len(selected_candidates)),
        (4, 18, 26),
    )
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=18)
    draw.text(
        (24, 22),
        "P0 OVERHEAD LINEAR GAP EVIDENCE / PLAN + ELEVATION",
        fill=(180, 255, 60),
        font=font,
    )
    for index, candidate in enumerate(selected_candidates):
        y0 = 68 + index * row_height
        draw.rectangle((20, y0, width - 20, y0 + row_height - 18), outline=(33, 91, 108))
        station_range = candidate["station_range_m"]
        cross_range = candidate["cross_range_m"]
        z_range = candidate["z_range_m"]
        member = (
            (station >= station_range[0] - 0.5)
            & (station <= station_range[1] + 0.5)
            & (lateral >= cross_range[0] - 0.4)
            & (lateral <= cross_range[1] + 0.4)
            & (points[:, 2] >= z_range[0] - 0.4)
            & (points[:, 2] <= z_range[1] + 0.4)
            & (distances > 0.25)
        )
        indexes = np.flatnonzero(member)
        if len(indexes) > 40_000:
            indexes = indexes[:: int(np.ceil(len(indexes) / 40_000))]
        draw.text(
            (34, y0 + 12),
            (
                f"{candidate['candidate_id']}  {candidate['classification']}  "
                f"span={candidate['extent_station_cross_z_m'][0]:.1f} m  "
                f"sample={candidate['sample_point_count']}"
            ),
            fill=(221, 239, 244),
            font=font,
        )
        panels = (
            ("PLAN / station-cross", lateral, cross_range),
            ("ELEVATION / station-z", points[:, 2], z_range),
        )
        for panel_index, (label, vertical_values, vertical_range) in enumerate(panels):
            panel_x0 = 34 + panel_index * 875
            panel_y0 = y0 + 70
            panel_width = 840
            panel_height = 300
            draw.text((panel_x0, panel_y0 - 26), label, fill=(122, 196, 214), font=font)
            if not len(indexes):
                continue
            s_min = station_range[0] - 0.5
            s_max = station_range[1] + 0.5
            v_min = vertical_range[0] - 0.4
            v_max = vertical_range[1] + 0.4
            px = panel_x0 + (station[indexes] - s_min) / (s_max - s_min) * panel_width
            py = panel_y0 + (v_max - vertical_values[indexes]) / (v_max - v_min) * panel_height
            for x_value, y_value in zip(px, py, strict=True):
                draw.point((int(x_value), int(y_value)), fill=(255, 126, 42))
            draw.rectangle(
                (panel_x0, panel_y0, panel_x0 + panel_width, panel_y0 + panel_height),
                outline=(50, 108, 125),
            )
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return destination


def audit_cloud_model_gaps(
    *,
    cloud_path: str | Path,
    obj_path: str | Path,
    model_origin_path: str | Path,
    frame_report_path: str | Path,
    asset_registry_path: str | Path,
    output_directory: str | Path,
    target_cloud_points: int = 4_000_000,
    model_surface_spacing_m: float = 0.15,
    unexplained_distance_m: float = 0.25,
) -> dict[str, Path]:
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty gap-audit directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model = parse_obj_model(obj_path)
    origin = np.asarray(load_json(Path(model_origin_path))["origin_xyz"], dtype=np.float64)
    model_samples, model_sampling = sample_model_surfaces(
        model,
        origin_xyz=origin,
        spacing_m=model_surface_spacing_m,
    )
    cloud_points, _, cloud_sampling = load_cloud_sample_rgb(
        cloud_path,
        target_point_count=target_cloud_points,
    )
    distances = reverse_model_distances(cloud_points, model_samples)
    frame = load_json(Path(frame_report_path))["frame"]
    candidates = detect_vertical_gap_candidates(
        cloud_points,
        distances,
        unexplained_distance_m=unexplained_distance_m,
    )
    registry = load_json(Path(asset_registry_path))
    relation_counts = match_vertical_candidates_to_assets(
        candidates,
        model=model,
        origin_xyz=origin,
        registry=registry,
        frame=frame,
    )
    linear_candidates = detect_overhead_linear_gap_candidates(
        cloud_points,
        distances,
        frame,
        unexplained_distance_m=unexplained_distance_m,
    )
    linear_relation_counts = match_linear_candidates_to_assets(
        linear_candidates,
        model=model,
        origin_xyz=origin,
        registry=registry,
        frame=frame,
    )
    corridor_scope = assign_corridor_scope_ownership(
        candidates,
        linear_candidates,
        model=model,
        origin_xyz=origin,
        frame=frame,
    )
    bands: list[dict[str, Any]] = []
    for name, minimum_z, maximum_z in (
        ("track_and_ground", 19.0, 22.0),
        ("platform_and_facade", 22.0, 26.0),
        ("canopy_and_overhead", 26.0, 31.0),
    ):
        selected = (cloud_points[:, 2] >= minimum_z) & (cloud_points[:, 2] < maximum_z)
        if np.any(selected):
            bands.append(
                {
                    "name": name,
                    "minimum_z_m": minimum_z,
                    "maximum_z_m": maximum_z,
                    **_distance_summary(distances[selected]),
                }
            )
    priority_counts = {
        priority: sum(candidate["priority"] == priority for candidate in candidates)
        for priority in ("P0", "P1", "P2", "P3")
    }
    report_path = output / "cloud_to_model_gap_inventory.json"
    figure_path = output / "cloud_to_model_gap_inventory.png"
    report = {
        "schema_version": "railway.cloud-model-gap-audit.v1",
        "inputs": {
            "cloud": str(Path(cloud_path).resolve()),
            "obj": str(Path(obj_path).resolve()),
            "model_origin": str(Path(model_origin_path).resolve()),
            "asset_registry": str(Path(asset_registry_path).resolve()),
        },
        "method": {
            "direction": "new_point_cloud_to_current_model",
            "model_surface_sampling": model_sampling,
            "cloud_sampling": cloud_sampling,
            "unexplained_distance_m": unexplained_distance_m,
            "warning": (
                "Unexplained points can be missing assets, finer detail, vegetation, moving objects, "
                "or residual registration error. Candidate classification is a build-review trigger."
            ),
        },
        "overall": _distance_summary(distances),
        "height_bands": bands,
        "corridor_scope": corridor_scope,
        "vertical_candidate_count": len(candidates),
        "priority_counts": priority_counts,
        "asset_relation_counts": relation_counts,
        "vertical_candidates": candidates,
        "overhead_linear_candidate_count": len(linear_candidates),
        "overhead_linear_priority_counts": {
            priority: sum(candidate["priority"] == priority for candidate in linear_candidates)
            for priority in ("P0", "P1", "P2")
        },
        "overhead_linear_asset_relation_counts": linear_relation_counts,
        "overhead_linear_candidates": linear_candidates,
        "decision": "new_asset_discovery_required",
    }
    render_gap_inventory(cloud_points, distances, candidates, frame, figure_path)
    evidence_paths = render_vertical_candidate_evidence(
        cloud_points,
        distances,
        candidates,
        frame,
        output / "candidate_evidence",
    )
    linear_evidence_path = render_linear_candidate_evidence(
        cloud_points,
        distances,
        linear_candidates,
        frame,
        output / "candidate_evidence" / "p0_overhead_linear_evidence.png",
    )
    report["candidate_evidence_figures"] = [str(path) for path in evidence_paths]
    report["linear_evidence_figure"] = (
        str(linear_evidence_path) if linear_evidence_path is not None else None
    )
    write_json(report_path, report)
    return {
        "report": report_path,
        "figure": figure_path,
        **{f"evidence_{index + 1}": path for index, path in enumerate(evidence_paths)},
        **({"linear_evidence": linear_evidence_path} if linear_evidence_path else {}),
    }
