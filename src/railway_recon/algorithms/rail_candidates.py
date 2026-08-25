from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks

from ..config import ProjectConfig
from ..geometry import fit_segment_corridor_frame
from ..io import load_json, write_json


def _height_mask(z: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, tuple[float, float]]:
    mode = config.get("z_mode", "percentile")
    if mode == "absolute":
        minimum = float(config["minimum_z"])
        maximum = float(config["maximum_z"])
    elif mode == "percentile":
        minimum, maximum = np.percentile(
            z,
            [float(config.get("z_percentile_min", 1.0)), float(config.get("z_percentile_max", 35.0))],
        )
        minimum += float(config.get("z_min_offset_m", 0.0))
        maximum += float(config.get("z_max_offset_m", 0.0))
    else:
        raise ValueError("rail detection z_mode must be 'absolute' or 'percentile'")
    if maximum <= minimum:
        raise ValueError("Invalid rail search height range")
    return (z >= minimum) & (z <= maximum), (float(minimum), float(maximum))


def _save_diagnostic(
    occupancy: np.ndarray,
    peaks: np.ndarray,
    pairs: list[dict[str, Any]],
    output: Path,
) -> None:
    density = np.log1p(occupancy.astype(np.float32))
    if density.max() > 0:
        density /= density.max()
    image = np.zeros((occupancy.shape[0], occupancy.shape[1], 3), dtype=np.uint8)
    image[..., 0] = (density * 24).astype(np.uint8)
    image[..., 1] = (density * 150).astype(np.uint8)
    image[..., 2] = (35 + density * 210).astype(np.uint8)
    image = np.flipud(image)
    scale = max(1, 1400 // max(1, image.shape[1]))
    image = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
    canvas = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(canvas)
    for peak in peaks:
        x = int(peak * scale)
        draw.line((x, 0, x, canvas.height - 1), fill=(255, 100, 45), width=max(1, scale))
    paired = {int(index) for pair in pairs for index in pair["peak_indexes"]}
    for peak in paired:
        x = int(peak * scale)
        draw.line((x, 0, x, canvas.height - 1), fill=(190, 255, 40), width=max(2, scale))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def detect_rail_candidates(
    project: ProjectConfig,
    segment_id: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    settings_path = project.resolve(project.value["algorithms"]["rail_detection"])
    config = load_json(settings_path)
    source = project.workspace_path("segments") / f"{segment_id}.laz"
    if not source.is_file():
        raise FileNotFoundError(source)
    output = project.workspace_path("derived") / f"{segment_id}_rail_candidates.laz"
    report_path = project.workspace_path("reports") / f"{segment_id}_rail_candidates.json"
    diagnostic = project.workspace_path("reports") / f"{segment_id}_rail_candidates.png"
    for path in (output, report_path, diagnostic):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite: {path}")

    cloud = laspy.read(source)
    x = np.asarray(cloud.x, dtype=np.float64)
    y = np.asarray(cloud.y, dtype=np.float64)
    z = np.asarray(cloud.z, dtype=np.float64)
    if not len(x):
        raise ValueError(f"Point cloud is empty: {source}")

    frame, segment_cameras = fit_segment_corridor_frame(project, segment_id)
    longitudinal, cross = frame.project(x, y)
    search, z_range = _height_mask(z, config)
    if not np.any(search):
        raise ValueError("No points remain in the rail search height range")

    cross_bin = float(config["cross_bin_m"])
    longitudinal_bin = float(config["longitudinal_bin_m"])
    # Pad the histogram so a real rail at the search envelope edge can still be a peak.
    cross_min = math.floor(float(cross[search].min()) / cross_bin) * cross_bin - 2 * cross_bin
    cross_max = math.ceil(float(cross[search].max()) / cross_bin) * cross_bin + 2 * cross_bin
    long_min = math.floor(float(longitudinal[search].min()) / longitudinal_bin) * longitudinal_bin
    long_max = math.ceil(float(longitudinal[search].max()) / longitudinal_bin) * longitudinal_bin
    nx = max(1, math.ceil((cross_max - cross_min) / cross_bin) + 1)
    ny = max(1, math.ceil((long_max - long_min) / longitudinal_bin) + 1)
    xi = np.clip(((cross[search] - cross_min) / cross_bin).astype(np.int64), 0, nx - 1)
    yi = np.clip(((longitudinal[search] - long_min) / longitudinal_bin).astype(np.int64), 0, ny - 1)
    occupancy = np.zeros((ny, nx), dtype=np.uint32)
    np.add.at(occupancy, (yi, xi), 1)
    coverage = np.count_nonzero(occupancy, axis=0) / max(1, ny)
    smooth = gaussian_filter1d(
        median_filter(coverage, size=int(config.get("median_filter_bins", 3))),
        sigma=float(config.get("gaussian_sigma_bins", 1.0)),
    )
    minimum_peak_distance = max(
        1, round(float(config["minimum_peak_spacing_m"]) / cross_bin)
    )
    peaks, _ = find_peaks(
        smooth,
        height=float(config["minimum_line_coverage"]),
        prominence=float(config["minimum_peak_prominence"]),
        distance=minimum_peak_distance,
    )
    positions = cross_min + (peaks + 0.5) * cross_bin
    scores = smooth[peaks]

    candidates: list[dict[str, Any]] = []
    for left in range(len(peaks)):
        for right in range(left + 1, len(peaks)):
            separation = float(positions[right] - positions[left])
            if float(config["rail_pair_minimum_m"]) <= separation <= float(
                config["rail_pair_maximum_m"]
            ):
                candidates.append(
                    {
                        "peak_indexes": [int(peaks[left]), int(peaks[right])],
                        "cross_positions_m": [float(positions[left]), float(positions[right])],
                        "separation_m": separation,
                        "score": float(scores[left] + scores[right]),
                    }
                )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    pairs: list[dict[str, Any]] = []
    used: set[int] = set()
    for pair in candidates:
        indexes = set(pair["peak_indexes"])
        if indexes & used:
            continue
        pairs.append({**pair, "track_id": f"TRACK-{len(pairs) + 1:04d}"})
        used.update(indexes)

    selected_lines = sorted({position for pair in pairs for position in pair["cross_positions_m"]})
    half_width = float(config["candidate_half_width_m"])
    candidate_mask = np.zeros(len(x), dtype=bool)
    line_records: list[dict[str, Any]] = []
    for index, position in enumerate(selected_lines, start=1):
        line_mask = search & (np.abs(cross - position) <= half_width)
        candidate_mask |= line_mask
        line_records.append(
            {
                "id": f"RAIL-{index:04d}",
                "cross_position_m": float(position),
                "median_z_m": float(np.median(z[line_mask])) if np.any(line_mask) else None,
                "point_count": int(np.count_nonzero(line_mask)),
            }
        )

    result_cloud = laspy.LasData(copy.deepcopy(cloud.header))
    result_cloud.points = cloud.points[candidate_mask]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
    result_cloud.write(temporary)
    os.replace(temporary, output)
    _save_diagnostic(occupancy, peaks, pairs, diagnostic)

    report = {
        "schema_version": "railway.rail-candidates.v1",
        "project_id": project.project_id,
        "segment_id": segment_id,
        "source": str(source),
        "frame": frame.to_json(),
        "frame_camera_count": len(segment_cameras),
        "frame_camera_index_range": [
            segment_cameras[0]["index"], segment_cameras[-1]["index"]
        ],
        "longitudinal_range_m": [float(long_min), float(long_max)],
        "search_z_range_m": list(z_range),
        "search_point_count": int(np.count_nonzero(search)),
        "peak_count": len(peaks),
        "peaks": [
            {"grid_index": int(i), "cross_position_m": float(p), "coverage": float(s)}
            for i, p, s in zip(peaks, positions, scores)
        ],
        "rail_pair_count": len(pairs),
        "rail_pairs": pairs,
        "rail_lines": line_records,
        "candidate_point_count": int(np.count_nonzero(candidate_mask)),
        "candidate_output": str(output),
        "diagnostic_image": str(diagnostic),
        "status": "geometry_candidates_only_manual_review_required",
    }
    write_json(report_path, report)
    return report
