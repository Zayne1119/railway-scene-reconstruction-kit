from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import laspy
import numpy as np

from .io import load_json, write_json
from .model_point_support import object_vertex_indices, parse_obj_model


def contiguous_true_ranges(values: np.ndarray) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(np.asarray(values, dtype=bool)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            ranges.append((start, index))
            start = None
    if start is not None:
        ranges.append((start, len(values)))
    return ranges


def audit_canopy_support_roof_contact(
    *,
    cloud_path: str | Path,
    candidate_report_path: str | Path,
    candidate_obj_path: str | Path,
    candidate_origin_path: str | Path,
    frame_report_path: str | Path,
    canopy_plane_audit_path: str | Path,
    roof_object_name: str,
    output_path: str | Path,
    cross_bin_m: float = 0.25,
    station_bin_m: float = 1.0,
    plane_tolerance_m: float = 0.06,
) -> dict[str, Any]:
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite roof-contact audit: {output}")
    candidate_report = load_json(Path(candidate_report_path))
    frame = load_json(Path(frame_report_path))["frame"]
    plane_audit = load_json(Path(canopy_plane_audit_path))
    plane = next(
        item for item in plane_audit["records"] if item["object_name"] == roof_object_name
    )["fitted_top_plane"]
    model = parse_obj_model(candidate_obj_path)
    origin = np.asarray(load_json(Path(candidate_origin_path))["origin_xyz"], dtype=np.float64)
    roof_indexes = object_vertex_indices(model, roof_object_name)
    if not len(roof_indexes):
        raise ValueError(f"Roof object is absent: {roof_object_name}")
    roof_vertices = model.vertices[roof_indexes] + origin
    support_root = str(candidate_report["asset_root"])
    support_indexes = object_vertex_indices(model, support_root)
    if not len(support_indexes):
        raise ValueError(f"Candidate support is absent: {support_root}")
    support_vertices = model.vertices[support_indexes] + origin

    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)

    def bounds(vertices: np.ndarray) -> dict[str, list[float]]:
        local = vertices[:, :2] - frame_origin
        station = local @ along
        lateral = local @ cross
        return {
            "station_range_m": [float(np.min(station)), float(np.max(station))],
            "cross_range_m": [float(np.min(lateral)), float(np.max(lateral))],
            "z_range_m": [float(np.min(vertices[:, 2])), float(np.max(vertices[:, 2]))],
        }

    roof_bounds = bounds(roof_vertices)
    support_bounds = bounds(support_vertices)
    target_cross = float(candidate_report["fit"]["cross_m"])
    station_minimum, station_maximum = roof_bounds["station_range_m"]
    cross_minimum = min(roof_bounds["cross_range_m"][0], target_cross) - 1.0
    cross_maximum = max(roof_bounds["cross_range_m"][1], target_cross) + 1.0
    cross_edges = np.arange(cross_minimum, cross_maximum + cross_bin_m, cross_bin_m)
    station_edges = np.arange(
        station_minimum, station_maximum + station_bin_m, station_bin_m
    )
    counts = np.zeros(len(cross_edges) - 1, dtype=np.int64)
    station_cells: list[set[int]] = [set() for _ in counts]
    plane_center = np.asarray(plane["center_xy"], dtype=np.float64)
    plane_a, plane_b, plane_c = plane["z_equals_a_dx_plus_b_dy_plus_c"]
    with laspy.open(Path(cloud_path)) as reader:
        for chunk in reader.chunk_iterator(2_000_000):
            x = np.asarray(chunk.x, dtype=np.float64)
            y = np.asarray(chunk.y, dtype=np.float64)
            z = np.asarray(chunk.z, dtype=np.float64)
            local = np.column_stack((x, y)) - frame_origin
            station = local @ along
            lateral = local @ cross
            predicted = (
                plane_a * (x - plane_center[0])
                + plane_b * (y - plane_center[1])
                + plane_c
            )
            selected = (
                (station >= station_minimum)
                & (station <= station_maximum)
                & (lateral >= cross_minimum)
                & (lateral <= cross_maximum)
                & (np.abs(z - predicted) <= plane_tolerance_m)
            )
            if not np.any(selected):
                continue
            cross_index = np.clip(
                np.searchsorted(cross_edges, lateral[selected], side="right") - 1,
                0,
                len(counts) - 1,
            )
            station_index = np.clip(
                np.searchsorted(station_edges, station[selected], side="right") - 1,
                0,
                len(station_edges) - 2,
            )
            counts += np.bincount(cross_index, minlength=len(counts))
            for cross_cell, station_cell in zip(cross_index, station_index, strict=True):
                station_cells[int(cross_cell)].add(int(station_cell))
    station_coverage = np.asarray([len(value) for value in station_cells], dtype=np.int64)
    required_station_cells = max(1, int(np.ceil(0.80 * (len(station_edges) - 1))))
    continuous = (counts >= 500) & (station_coverage >= required_station_cells)
    ranges = contiguous_true_ranges(continuous)
    evidence_ranges = [
        [float(cross_edges[start]), float(cross_edges[end])] for start, end in ranges
    ]
    if evidence_ranges:
        distance_to_evidence = min(
            max(0.0, lower - target_cross, target_cross - upper)
            for lower, upper in evidence_ranges
        )
    else:
        distance_to_evidence = float("inf")
    mesh_cross_overlap = max(
        0.0,
        min(support_bounds["cross_range_m"][1], roof_bounds["cross_range_m"][1])
        - max(support_bounds["cross_range_m"][0], roof_bounds["cross_range_m"][0]),
    )
    gates = {
        "continuous_roof_evidence_exists": bool(evidence_ranges),
        "support_cross_within_0_50m_of_continuous_roof_evidence": (
            distance_to_evidence <= 0.50
        ),
        "support_plan_overlaps_roof_mesh": mesh_cross_overlap > 0.0,
    }
    passed = all(gates.values())
    result = {
        "schema_version": "railway.canopy-support-roof-contact-audit.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "candidate_id": candidate_report["candidate_id"],
        "asset_root": support_root,
        "roof_object_name": roof_object_name,
        "roof_mesh_bounds": roof_bounds,
        "support_mesh_bounds": support_bounds,
        "target_cross_m": target_cross,
        "settings": {
            "cross_bin_m": cross_bin_m,
            "station_bin_m": station_bin_m,
            "plane_tolerance_m": plane_tolerance_m,
            "required_station_cell_coverage": required_station_cells,
        },
        "continuous_roof_cross_ranges_m": evidence_ranges,
        "distance_to_continuous_roof_evidence_m": distance_to_evidence,
        "support_roof_mesh_cross_overlap_m": mesh_cross_overlap,
        "cross_bins": [
            {
                "cross_range_m": [float(cross_edges[index]), float(cross_edges[index + 1])],
                "point_count": int(counts[index]),
                "station_cell_coverage": int(station_coverage[index]),
                "continuous_roof_evidence": bool(continuous[index]),
            }
            for index in range(len(counts))
        ],
        "gates": gates,
        "passed": passed,
        "status": (
            "support_roof_contact_accepted"
            if passed
            else "support_candidate_rejected_no_continuous_roof_contact"
        ),
        "decision": (
            "retain_candidate"
            if passed
            else "do_not_add_standalone_support; classify sparse bands as coincident roof/platform fragments"
        ),
    }
    write_json(output, result)
    return result
