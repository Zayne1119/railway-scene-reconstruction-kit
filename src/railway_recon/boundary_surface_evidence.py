from __future__ import annotations

from pathlib import Path
from typing import Any

import laspy
import numpy as np

from .io import load_json, write_json


def summarize_station_bins(
    station: np.ndarray,
    *,
    station_minimum_m: float,
    station_maximum_m: float,
    station_bin_m: float,
    minimum_points_per_bin: int,
) -> dict[str, Any]:
    edges = np.arange(
        station_minimum_m,
        station_maximum_m + station_bin_m * 0.5,
        station_bin_m,
    )
    if edges[-1] < station_maximum_m:
        edges = np.append(edges, station_maximum_m)
    counts, _ = np.histogram(station, bins=edges)
    occupied = counts >= minimum_points_per_bin
    bins = [
        {
            "station_range_m": [float(edges[index]), float(edges[index + 1])],
            "point_count": int(counts[index]),
            "occupied": bool(occupied[index]),
        }
        for index in range(len(counts))
    ]
    runs: list[list[float]] = []
    start: int | None = None
    for index, value in enumerate([*occupied.tolist(), False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append([float(edges[start]), float(edges[index])])
            start = None
    longest = max((end - begin for begin, end in runs), default=0.0)
    return {
        "station_bin_m": station_bin_m,
        "minimum_points_per_bin": minimum_points_per_bin,
        "bin_count": len(counts),
        "occupied_bin_count": int(np.sum(occupied)),
        "occupied_fraction": float(np.mean(occupied)) if len(occupied) else 0.0,
        "occupied_station_runs_m": runs,
        "longest_occupied_run_m": longest,
        "bins": bins,
    }


def analyze_boundary_surface_evidence(
    *,
    cloud_path: str | Path,
    frame_report_path: str | Path,
    output_path: str | Path,
    station_minimum_m: float,
    station_maximum_m: float,
    windows: list[dict[str, Any]],
    station_bin_m: float = 0.50,
    minimum_points_per_bin: int = 20,
    chunk_size: int = 2_000_000,
) -> dict[str, Any]:
    cloud = Path(cloud_path).resolve()
    frame_path = Path(frame_report_path).resolve()
    output = Path(output_path).resolve()
    for path in (cloud, frame_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(output)
    if station_maximum_m <= station_minimum_m:
        raise ValueError("Station evidence interval must be increasing")
    if not windows:
        raise ValueError("At least one surface window is required")
    frame = load_json(frame_path)["frame"]
    origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    collected: dict[str, list[np.ndarray]] = {
        str(window["name"]): [] for window in windows
    }
    source_point_count = 0
    corridor_point_count = 0
    with laspy.open(cloud) as reader:
        source_point_count = int(reader.header.point_count)
        for chunk in reader.chunk_iterator(chunk_size):
            x = np.asarray(chunk.x, dtype=np.float64)
            y = np.asarray(chunk.y, dtype=np.float64)
            z = np.asarray(chunk.z, dtype=np.float64)
            local = np.column_stack((x, y)) - origin
            station = local @ along
            lateral = local @ cross
            corridor = (station >= station_minimum_m) & (station <= station_maximum_m)
            corridor_point_count += int(np.sum(corridor))
            for window in windows:
                cross_range = [float(value) for value in window["cross_range_m"]]
                z_range = [float(value) for value in window["z_range_m"]]
                selected = (
                    corridor
                    & (lateral >= cross_range[0])
                    & (lateral <= cross_range[1])
                    & (z >= z_range[0])
                    & (z <= z_range[1])
                )
                if np.any(selected):
                    collected[str(window["name"])].append(
                        np.column_stack((station[selected], lateral[selected], z[selected]))
                    )
    results: list[dict[str, Any]] = []
    for window in windows:
        name = str(window["name"])
        points = (
            np.vstack(collected[name])
            if collected[name]
            else np.empty((0, 3), dtype=np.float64)
        )
        summary = summarize_station_bins(
            points[:, 0],
            station_minimum_m=station_minimum_m,
            station_maximum_m=station_maximum_m,
            station_bin_m=station_bin_m,
            minimum_points_per_bin=minimum_points_per_bin,
        )
        results.append(
            {
                "name": name,
                "cross_range_m": [float(value) for value in window["cross_range_m"]],
                "z_range_m": [float(value) for value in window["z_range_m"]],
                "point_count": len(points),
                "cross_p05_p50_p95_m": (
                    np.percentile(points[:, 1], (5, 50, 95)).tolist()
                    if len(points)
                    else None
                ),
                "z_p05_p50_p95_m": (
                    np.percentile(points[:, 2], (5, 50, 95)).tolist()
                    if len(points)
                    else None
                ),
                **summary,
            }
        )
    result = {
        "schema_version": "railway.boundary-surface-evidence.v1",
        "cloud": str(cloud),
        "frame_report": str(frame_path),
        "source_point_count": source_point_count,
        "corridor_station_range_m": [station_minimum_m, station_maximum_m],
        "corridor_point_count": corridor_point_count,
        "windows": results,
        "status": "point_evidence_profile_written_no_geometry_action",
    }
    write_json(output, result)
    return result
