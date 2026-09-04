from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

import laspy
import numpy as np

from .io import load_json, write_json


def _binned_vertical_profile(
    points: np.ndarray,
    *,
    minimum_station_m: float,
    maximum_station_m: float,
    bin_size_m: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    edges = np.arange(minimum_station_m, maximum_station_m + bin_size_m, bin_size_m)
    for low, high in pairwise(edges):
        z = points[(points[:, 0] >= low) & (points[:, 0] < high), 2]
        if len(z) < 50:
            continue
        records.append(
            {
                "station_range_m": [float(low), float(high)],
                "point_count": len(z),
                "z_percentiles_m": {
                    key: float(value)
                    for key, value in zip(
                        ("p02", "p10", "p50", "p90", "p98"),
                        np.percentile(z, (2.0, 10.0, 50.0, 90.0, 98.0)),
                        strict=True,
                    )
                },
            }
        )
    return records


def analyse_station_entry_surfaces(
    *,
    cloud_path: str | Path,
    frame_report_path: str | Path,
    output_path: str | Path,
    station_bounds_m: tuple[float, float] = (98.0, 126.0),
    cross_bounds_m: tuple[float, float] = (10.0, 30.0),
    z_bounds_m: tuple[float, float] = (20.0, 28.5),
    plane_bands_m: tuple[tuple[str, float, float], ...] = (
        ("platform_side_facade", 15.35, 15.85),
        ("outer_facade", 19.75, 20.30),
    ),
    station_bin_size_m: float = 1.0,
    cross_peak_station_segments_m: tuple[tuple[float, float], ...] = (
        (98.0, 103.5),
        (103.5, 112.5),
        (115.0, 124.0),
    ),
    chunk_size: int = 2_000_000,
) -> Path:
    cloud = Path(cloud_path).resolve()
    frame = load_json(Path(frame_report_path))["frame"]
    origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    point_parts: list[np.ndarray] = []
    with laspy.open(cloud) as reader:
        for chunk in reader.chunk_iterator(chunk_size):
            x = np.asarray(chunk.x, dtype=np.float64)
            y = np.asarray(chunk.y, dtype=np.float64)
            z = np.asarray(chunk.z, dtype=np.float64)
            local = np.column_stack((x, y)) - origin
            station = local @ along
            lateral = local @ cross
            selected = (
                (station >= station_bounds_m[0])
                & (station <= station_bounds_m[1])
                & (lateral >= cross_bounds_m[0])
                & (lateral <= cross_bounds_m[1])
                & (z >= z_bounds_m[0])
                & (z <= z_bounds_m[1])
            )
            if np.any(selected):
                point_parts.append(np.column_stack((station[selected], lateral[selected], z[selected])))
    if not point_parts:
        raise ValueError("Station-entry crop contains no points")
    points = np.vstack(point_parts)

    cross_edges = np.arange(cross_bounds_m[0], cross_bounds_m[1] + 0.10, 0.10)
    cross_counts, _ = np.histogram(points[:, 1], bins=cross_edges)
    peak_indexes = np.argsort(cross_counts)[-12:][::-1]
    cross_peaks = [
        {
            "cross_range_m": [float(cross_edges[index]), float(cross_edges[index + 1])],
            "point_count": int(cross_counts[index]),
        }
        for index in peak_indexes
    ]
    segmented_cross_peaks: list[dict[str, Any]] = []
    fine_cross_edges = np.arange(cross_bounds_m[0], cross_bounds_m[1] + 0.025, 0.025)
    for station_low, station_high in cross_peak_station_segments_m:
        segment_points = points[
            (points[:, 0] >= station_low) & (points[:, 0] < station_high)
        ]
        fine_counts, _ = np.histogram(segment_points[:, 1], bins=fine_cross_edges)
        segment_peak_indexes = np.argsort(fine_counts)[-10:][::-1]
        segmented_cross_peaks.append(
            {
                "station_range_m": [station_low, station_high],
                "point_count": len(segment_points),
                "cross_peaks": [
                    {
                        "cross_range_m": [
                            float(fine_cross_edges[index]),
                            float(fine_cross_edges[index + 1]),
                        ],
                        "point_count": int(fine_counts[index]),
                    }
                    for index in segment_peak_indexes
                ],
            }
        )
    planes: list[dict[str, Any]] = []
    for name, low, high in plane_bands_m:
        plane_points = points[(points[:, 1] >= low) & (points[:, 1] <= high)]
        planes.append(
            {
                "name": name,
                "cross_band_m": [low, high],
                "point_count": len(plane_points),
                "cross_percentiles_m": {
                    key: float(value)
                    for key, value in zip(
                        ("p05", "p50", "p95"),
                        np.percentile(plane_points[:, 1], (5.0, 50.0, 95.0)),
                        strict=True,
                    )
                },
                "station_profile": _binned_vertical_profile(
                    plane_points,
                    minimum_station_m=station_bounds_m[0],
                    maximum_station_m=station_bounds_m[1],
                    bin_size_m=station_bin_size_m,
                ),
            }
        )

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        output,
        {
            "schema_version": "railway.station-entry-surface-analysis.v1",
            "cloud": str(cloud),
            "crop_bounds_local": {
                "station_m": list(station_bounds_m),
                "cross_m": list(cross_bounds_m),
                "z_m": list(z_bounds_m),
            },
            "crop_point_count": len(points),
            "cross_histogram_peaks": cross_peaks,
            "segmented_cross_histogram_peaks": segmented_cross_peaks,
            "plane_bands": planes,
            "status": "surface_fit_evidence_only_geometry_unchanged",
        },
    )
    return output
