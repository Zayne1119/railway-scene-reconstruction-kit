from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .io import load_json, write_json
from .model_point_support import object_vertex_indices, parse_obj_model


def _sample_polyline(points: np.ndarray, spacing_m: float = 0.02) -> np.ndarray:
    ordered = np.asarray(points, dtype=np.float64)
    if len(ordered) < 2:
        return ordered
    samples = []
    for first, second in pairwise(ordered):
        count = max(2, int(np.ceil(float(np.linalg.norm(second - first)) / spacing_m)) + 1)
        amount = np.linspace(0.0, 1.0, count)[:, None]
        samples.append(first * (1.0 - amount) + second * amount)
    return np.vstack(samples)


def sample_slab_cross_section(points_cross_z: np.ndarray) -> np.ndarray:
    """Densify top, bottom and side edges independently of face triangulation."""
    points = np.asarray(points_cross_z, dtype=np.float64)
    cross_values = np.unique(np.round(points[:, 0], 5))
    top = []
    bottom = []
    for cross in cross_values:
        selected = points[np.isclose(points[:, 0], cross, atol=1.0e-5)]
        top.append([cross, float(np.max(selected[:, 1]))])
        bottom.append([cross, float(np.min(selected[:, 1]))])
    top_points = np.asarray(top, dtype=np.float64)
    bottom_points = np.asarray(bottom, dtype=np.float64)
    pieces = [_sample_polyline(top_points), _sample_polyline(bottom_points)]
    for index in (0, -1):
        low = bottom_points[index]
        high = top_points[index]
        pieces.append(_sample_polyline(np.vstack((low, high))))
    return np.unique(np.round(np.vstack(pieces), 5), axis=0)


def _subsystem_points(
    *,
    obj_path: Path,
    origin_path: Path,
    frame: dict[str, Any],
    prefixes: tuple[str, ...],
) -> np.ndarray:
    model = parse_obj_model(obj_path)
    origin = np.asarray(load_json(origin_path)["origin_xyz"], dtype=np.float64)
    indexes = sorted(
        {
            int(index)
            for name in model.faces_by_object
            if name.startswith(prefixes)
            for index in object_vertex_indices(model, name)
        }
    )
    if not indexes:
        return np.empty((0, 3), dtype=np.float64)
    world = model.vertices[np.asarray(indexes, dtype=np.int64)] + origin
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    local_xy = world[:, :2] - frame_origin
    return np.column_stack((local_xy @ along, local_xy @ cross, world[:, 2]))


def audit_subsystem_handoff(
    current_points: np.ndarray,
    adjacent_points: np.ndarray,
    *,
    seam_station_m: float,
    edge_depth_m: float,
    station_gap_max_m: float,
    cross_z_p90_max_m: float,
    boundary_search_radius_m: float = 1.0,
    sample_slab_section: bool = False,
) -> dict[str, Any]:
    current = np.asarray(current_points, dtype=np.float64)
    adjacent = np.asarray(adjacent_points, dtype=np.float64)
    if not len(current) or not len(adjacent):
        return {
            "passed": False,
            "status": "missing_subsystem_geometry",
            "current_point_count": len(current),
            "adjacent_point_count": len(adjacent),
        }
    if boundary_search_radius_m <= 0:
        raise ValueError("Boundary search radius must be positive")
    # A refined subsystem may intentionally extend a few centimetres beyond the
    # nominal corridor seam, while an ownership-clipped adjacent subsystem can
    # begin a few centimetres before it. Search a bounded neighbourhood first,
    # then measure the *actual* opposing edges. Using station_gap_max_m as the
    # search radius hid overlaps larger than the acceptance tolerance.
    current_near = current[
        (current[:, 0] >= seam_station_m - boundary_search_radius_m)
        & (current[:, 0] <= seam_station_m + boundary_search_radius_m)
    ]
    adjacent_near = adjacent[
        (adjacent[:, 0] >= seam_station_m - boundary_search_radius_m)
        & (adjacent[:, 0] <= seam_station_m + boundary_search_radius_m)
    ]
    if not len(current_near) or not len(adjacent_near):
        return {
            "passed": False,
            "status": "subsystem_does_not_reach_boundary",
            "current_point_count": len(current),
            "adjacent_point_count": len(adjacent),
        }
    current_edge_station = float(np.max(current_near[:, 0]))
    adjacent_edge_station = float(np.min(adjacent_near[:, 0]))
    current_edge = current_near[
        current_near[:, 0] >= current_edge_station - edge_depth_m
    ]
    adjacent_edge = adjacent_near[
        adjacent_near[:, 0] <= adjacent_edge_station + edge_depth_m
    ]
    current_cross_z = np.unique(np.round(current_edge[:, 1:], 5), axis=0)
    adjacent_cross_z = np.unique(np.round(adjacent_edge[:, 1:], 5), axis=0)
    if sample_slab_section:
        current_cross_z = sample_slab_cross_section(current_cross_z)
        adjacent_cross_z = sample_slab_cross_section(adjacent_cross_z)
    current_to_adjacent = cKDTree(adjacent_cross_z).query(current_cross_z, workers=-1)[0]
    adjacent_to_current = cKDTree(current_cross_z).query(adjacent_cross_z, workers=-1)[0]
    distances = np.concatenate((current_to_adjacent, adjacent_to_current))
    station_gap = adjacent_edge_station - current_edge_station
    p90 = float(np.percentile(distances, 90))
    gates = {
        "absolute_station_gap_within_limit": abs(station_gap) <= station_gap_max_m,
        "symmetric_cross_z_p90_within_limit": p90 <= cross_z_p90_max_m,
    }
    return {
        "current_edge_station_m": current_edge_station,
        "adjacent_edge_station_m": adjacent_edge_station,
        "signed_station_gap_m": station_gap,
        "current_edge_sample_count": len(current_cross_z),
        "adjacent_edge_sample_count": len(adjacent_cross_z),
        "symmetric_cross_z_distance": {
            "p50_m": float(np.percentile(distances, 50)),
            "p90_m": p90,
            "p95_m": float(np.percentile(distances, 95)),
            "maximum_m": float(np.max(distances)),
        },
        "station_gap_max_m": station_gap_max_m,
        "boundary_search_radius_m": boundary_search_radius_m,
        "cross_section_sampling": (
            "densified_slab_edges" if sample_slab_section else "unique_vertices"
        ),
        "cross_z_p90_max_m": cross_z_p90_max_m,
        "gates": gates,
        "passed": all(gates.values()),
        "status": "pass" if all(gates.values()) else "handoff_reconciliation_required",
    }


def audit_adjacent_context_seams(
    *,
    current_obj: str | Path,
    current_origin: str | Path,
    adjacent_obj: str | Path,
    adjacent_origin: str | Path,
    frame_report: str | Path,
    seam_station_m: float,
    output_path: str | Path,
) -> dict[str, Any]:
    current_model = Path(current_obj).resolve()
    current_origin_path = Path(current_origin).resolve()
    adjacent_model = Path(adjacent_obj).resolve()
    adjacent_origin_path = Path(adjacent_origin).resolve()
    frame_path = Path(frame_report).resolve()
    output = Path(output_path).resolve()
    for path in (
        current_model,
        current_origin_path,
        adjacent_model,
        adjacent_origin_path,
        frame_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(output)
    frame = load_json(frame_path)["frame"]
    settings = {
        "track": {
            "current_prefixes": ("TRACKGRAPH--TRACK-",),
            "adjacent_prefixes": ("TRACK--", "TRACK-", "NEXTTRACK--TRACK-"),
            "edge_depth_m": 0.08,
            "station_gap_max_m": 0.05,
            "cross_z_p90_max_m": 0.15,
        },
        "platform": {
            "current_prefixes": (
                "PLATFORM-",
                "SEG2050-PLATFORM--",
                "SEG2100-PLATFORM--",
                "SEG2150-PLATFORM--",
            ),
            "adjacent_prefixes": (
                "STATION--PLATFORM--",
                "NEXTPLATFORM--STATION--PLATFORM--",
                "NEXTPLATFORM--S2200_2250M-PLATFORM-",
                "S2200_2250M-PLATFORM-",
                "ADJACENT-NEXT-RIGHT-PLATFORM-TRANSITION",
            ),
            "edge_depth_m": 0.08,
            "station_gap_max_m": 0.05,
            "cross_z_p90_max_m": 0.20,
            "sample_slab_section": True,
        },
        "opposite_platform": {
            "current_prefixes": (
                "S2050-OPPOSITE-PLATFORM-",
                "S2100-OPPOSITE-PLATFORM-",
                "S2150-OPPOSITE-PLATFORM-",
            ),
            "adjacent_prefixes": (
                "STATION--OPPOSITE-PLATFORM-",
                "ADJACENT-NEXT-OPPOSITE-PLATFORM-",
            ),
            "edge_depth_m": 0.08,
            "station_gap_max_m": 0.05,
            "cross_z_p90_max_m": 0.20,
            "sample_slab_section": True,
        },
        "canopy_roof_surface_01": {
            "current_prefixes": (
                "SEG2150-CANOPY--RIGHT-CANOPY-ROOF-CANDIDATE-001-SURFACE-01-",
            ),
            "adjacent_prefixes": (
                "ADJACENT-NEXT-RIGHT-CANOPY-ROOF-SURFACE-01",
            ),
            "edge_depth_m": 0.08,
            "station_gap_max_m": 0.05,
            "cross_z_p90_max_m": 0.03,
            "sample_slab_section": True,
        },
        "canopy_roof_surface_02": {
            "current_prefixes": (
                "SEG2150-CANOPY--RIGHT-CANOPY-ROOF-CANDIDATE-001-SURFACE-02-RUN-03",
            ),
            "adjacent_prefixes": (
                "ADJACENT-NEXT-RIGHT-CANOPY-ROOF-SURFACE-02",
            ),
            "edge_depth_m": 0.08,
            "station_gap_max_m": 0.05,
            "cross_z_p90_max_m": 0.03,
            "sample_slab_section": True,
        },
        "canopy_roof_surface_03": {
            "current_prefixes": (
                "SEG2150-CANOPY--RIGHT-CANOPY-ROOF-CANDIDATE-001-SURFACE-03-",
            ),
            "adjacent_prefixes": (
                "ADJACENT-NEXT-RIGHT-CANOPY-ROOF-SURFACE-03",
            ),
            "edge_depth_m": 0.08,
            "station_gap_max_m": 0.05,
            "cross_z_p90_max_m": 0.03,
            "sample_slab_section": True,
        },
        "opposite_canopy_roof": {
            "current_prefixes": (
                "S2050-OPPOSITE-CANOPY-ROOF-",
                "S2100-OPPOSITE-CANOPY-ROOF-",
                "S2150-OPPOSITE-CANOPY-ROOF-",
            ),
            "adjacent_prefixes": ("ADJACENT-NEXT-OPPOSITE-CANOPY-ROOF-",),
            "edge_depth_m": 0.08,
            "station_gap_max_m": 0.05,
            "cross_z_p90_max_m": 0.05,
            "sample_slab_section": True,
        },
    }
    subsystems: dict[str, Any] = {}
    for name, values in settings.items():
        current_points = _subsystem_points(
            obj_path=current_model,
            origin_path=current_origin_path,
            frame=frame,
            prefixes=values["current_prefixes"],
        )
        adjacent_points = _subsystem_points(
            obj_path=adjacent_model,
            origin_path=adjacent_origin_path,
            frame=frame,
            prefixes=values["adjacent_prefixes"],
        )
        subsystems[name] = audit_subsystem_handoff(
            current_points,
            adjacent_points,
            seam_station_m=seam_station_m,
            edge_depth_m=float(values["edge_depth_m"]),
            station_gap_max_m=float(values["station_gap_max_m"]),
            cross_z_p90_max_m=float(values["cross_z_p90_max_m"]),
            sample_slab_section=bool(values.get("sample_slab_section", False)),
        )
    passed = all(item["passed"] for item in subsystems.values())
    result = {
        "schema_version": "railway.adjacent-context-seam-audit.v1",
        "current_obj": str(current_model),
        "adjacent_obj": str(adjacent_model),
        "frame_report": str(frame_path),
        "seam_station_m": seam_station_m,
        "subsystems": subsystems,
        "passed": passed,
        "status": "pass" if passed else "candidate_blocked_reconciliation_required",
        "limitations": [
            "This audit compares boundary cross-section vertices, not full surface-to-cloud fit.",
            "Conductors are audited separately because the adjacent automatic scene has incomplete wire classes.",
        ],
    }
    write_json(output, result)
    return result
