from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from scipy.ndimage import binary_dilation, label

from ..config import ProjectConfig
from ..geometry import fit_segment_corridor_frame
from ..io import load_json, write_json


def _write_subset(cloud: laspy.LasData, mask: np.ndarray, output: Path) -> None:
    result = laspy.LasData(copy.deepcopy(cloud.header))
    result.points = cloud.points[mask]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
    result.write(temporary)
    os.replace(temporary, output)


def _vertical_candidates(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, config: dict[str, Any]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    size = float(config["xy_bin_m"])
    x_min = math.floor(float(x.min()) / size) * size
    y_min = math.floor(float(y.min()) / size) * size
    nx = math.ceil((float(x.max()) - x_min) / size) + 1
    ny = math.ceil((float(y.max()) - y_min) / size) + 1
    xi = np.clip(((x - x_min) / size).astype(np.int64), 0, nx - 1)
    yi = np.clip(((y - y_min) / size).astype(np.int64), 0, ny - 1)
    keys = xi + nx * yi
    count = np.bincount(keys, minlength=nx * ny)
    minimum = np.full(nx * ny, np.inf, dtype=np.float32)
    maximum = np.full(nx * ny, -np.inf, dtype=np.float32)
    np.minimum.at(minimum, keys, z.astype(np.float32))
    np.maximum.at(maximum, keys, z.astype(np.float32))
    cells = (
        ((maximum - minimum) >= float(config["minimum_z_span_m"]))
        & (count >= int(config["minimum_cell_points"]))
    ).reshape(ny, nx)
    connected, component_count = label(binary_dilation(cells, iterations=1))
    point_components = connected[yi, xi]
    accepted_labels: list[int] = []
    records: list[dict[str, Any]] = []
    for component_id in range(1, component_count + 1):
        mask = point_components == component_id
        point_count = int(np.count_nonzero(mask))
        if point_count < int(config["minimum_component_points"]):
            continue
        footprint = max(float(np.ptp(x[mask])), float(np.ptp(y[mask])))
        height = float(np.ptp(z[mask]))
        if footprint > float(config["maximum_footprint_m"]) or height < float(
            config["minimum_z_span_m"]
        ):
            continue
        accepted_labels.append(component_id)
        records.append(
            {
                "id": f"VERTICAL-CANDIDATE-{len(records) + 1:04d}",
                "center_x": float(np.median(x[mask])),
                "center_y": float(np.median(y[mask])),
                "minimum_z": float(z[mask].min()),
                "maximum_z": float(z[mask].max()),
                "height_m": height,
                "footprint_m": footprint,
                "point_count": point_count,
                "status": "unclassified_requires_photo_evidence",
            }
        )
    return np.isin(point_components, accepted_labels), records


def _cable_candidates(
    longitudinal: np.ndarray,
    cross: np.ndarray,
    z: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    z_min, z_max = np.percentile(
        z, [float(config["z_percentile_min"]), float(config["z_percentile_max"])]
    )
    valid = (
        (cross >= float(config["minimum_cross_m"]))
        & (cross <= float(config["maximum_cross_m"]))
        & (z >= z_min)
        & (z <= z_max)
    )
    if not np.any(valid):
        return np.zeros(len(z), dtype=bool), []
    cross_bin = float(config["cross_bin_m"])
    z_bin = float(config["z_bin_m"])
    long_bin = float(config["longitudinal_bin_m"])
    cross_min = float(config["minimum_cross_m"])
    long_min = math.floor(float(longitudinal[valid].min()) / long_bin) * long_bin
    nx = math.ceil((float(config["maximum_cross_m"]) - cross_min) / cross_bin) + 1
    nz = math.ceil((float(z_max) - float(z_min)) / z_bin) + 1
    ny = math.ceil((float(longitudinal[valid].max()) - long_min) / long_bin) + 1
    xi = np.clip(((cross[valid] - cross_min) / cross_bin).astype(np.int64), 0, nx - 1)
    zi = np.clip(((z[valid] - z_min) / z_bin).astype(np.int64), 0, nz - 1)
    yi = np.clip(((longitudinal[valid] - long_min) / long_bin).astype(np.int64), 0, ny - 1)
    triplets = np.unique(xi + nx * (zi + nz * yi))
    xz_keys = triplets % (nx * nz)
    coverage = (np.bincount(xz_keys, minlength=nx * nz) / max(1, ny)).reshape(nz, nx)
    connected, component_count = label(
        binary_dilation(coverage >= float(config["minimum_coverage"]), iterations=1)
    )
    lines: list[dict[str, Any]] = []
    for component_id in range(1, component_count + 1):
        z_indexes, x_indexes = np.nonzero(connected == component_id)
        if not len(x_indexes):
            continue
        span = max(
            (x_indexes.max() - x_indexes.min() + 1) * cross_bin,
            (z_indexes.max() - z_indexes.min() + 1) * z_bin,
        )
        if span > float(config["maximum_cross_section_span_m"]):
            continue
        component_coverage = coverage[z_indexes, x_indexes]
        best = int(np.argmax(component_coverage))
        lines.append(
            {
                "id": f"CABLE-CANDIDATE-{len(lines) + 1:04d}",
                "cross_position_m": cross_min + (x_indexes[best] + 0.5) * cross_bin,
                "elevation_z": float(z_min) + (z_indexes[best] + 0.5) * z_bin,
                "longitudinal_coverage": float(component_coverage[best]),
                "status": "unclassified_requires_photo_evidence",
            }
        )
    accepted = np.zeros(len(z), dtype=bool)
    for item in lines:
        accepted |= (
            (np.abs(cross - item["cross_position_m"]) <= float(config["candidate_half_width_m"]))
            & (np.abs(z - item["elevation_z"]) <= float(config["candidate_half_height_m"]))
        )
    return accepted & valid, lines


def detect_linear_candidates(
    project: ProjectConfig,
    segment_id: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    settings = load_json(project.resolve(project.value["algorithms"]["linear_detection"]))
    source = project.workspace_path("segments") / f"{segment_id}.laz"
    if not source.is_file():
        raise FileNotFoundError(source)
    vertical_output = project.workspace_path("derived") / f"{segment_id}_vertical_candidates.laz"
    cable_output = project.workspace_path("derived") / f"{segment_id}_cable_candidates.laz"
    report_path = project.workspace_path("reports") / f"{segment_id}_linear_candidates.json"
    for path in (vertical_output, cable_output, report_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite: {path}")

    cloud = laspy.read(source)
    x = np.asarray(cloud.x, dtype=np.float64)
    y = np.asarray(cloud.y, dtype=np.float64)
    z = np.asarray(cloud.z, dtype=np.float64)
    frame, segment_cameras = fit_segment_corridor_frame(project, segment_id)
    longitudinal, cross = frame.project(x, y)
    vertical_mask, verticals = _vertical_candidates(x, y, z, settings["vertical"])
    cable_mask, cables = _cable_candidates(
        longitudinal, cross, z, settings["cable"]
    )
    _write_subset(cloud, vertical_mask, vertical_output)
    _write_subset(cloud, cable_mask, cable_output)
    report = {
        "schema_version": "railway.linear-candidates.v1",
        "project_id": project.project_id,
        "segment_id": segment_id,
        "source": str(source),
        "frame": frame.to_json(),
        "frame_camera_count": len(segment_cameras),
        "frame_camera_index_range": [
            segment_cameras[0]["index"], segment_cameras[-1]["index"]
        ],
        "vertical_candidate_count": len(verticals),
        "vertical_candidates": verticals,
        "vertical_output": str(vertical_output),
        "cable_candidate_count": len(cables),
        "cable_candidates": cables,
        "cable_output": str(cable_output),
        "status": "geometry_candidates_only_semantic_review_required",
    }
    write_json(report_path, report)
    return report
