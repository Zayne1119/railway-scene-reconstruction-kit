from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from PIL import Image, ImageDraw

from .camera import camera_trajectory, load_camera_rows
from .config import ProjectConfig
from .geometry import fit_corridor_frame_dominant_axis_regression
from .io import load_json, write_json

OUTPUT_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{2,63}$")


def _find_interval(
    items: list[dict[str, Any]], chainage_m: float, start_key: str, end_key: str
) -> dict[str, Any]:
    for index, item in enumerate(items):
        start = float(item[start_key])
        end = float(item[end_key])
        if start <= chainage_m < end or (
            index == len(items) - 1 and math.isclose(chainage_m, end)
        ):
            return item
    raise ValueError(f"No interval contains chainage {chainage_m:.3f} m")


def _density_rgb(histogram: np.ndarray) -> Image.Image:
    density = np.log1p(histogram.astype(np.float32))
    positive = density[density > 0]
    scale = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    density = np.clip(density / max(scale, 1e-6), 0.0, 1.0)
    red = (8.0 + 145.0 * density).astype(np.uint8)
    green = (20.0 + 205.0 * density).astype(np.uint8)
    blue = (30.0 + 225.0 * density).astype(np.uint8)
    image = np.stack((red, green, blue), axis=-1)
    image[histogram == 0] = (4, 18, 27)
    return Image.fromarray(image)


def _render_panel(
    histogram: np.ndarray,
    output: Path,
    title: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    x_label: str,
    y_label: str,
) -> None:
    plot_width, plot_height = 1000, 420
    left, top, right, bottom = 76, 54, 26, 58
    canvas = Image.new(
        "RGB", (left + plot_width + right, top + plot_height + bottom), (4, 18, 27)
    )
    density = _density_rgb(np.flipud(histogram)).resize(
        (plot_width, plot_height), Image.Resampling.NEAREST
    )
    canvas.paste(density, (left, top))
    draw = ImageDraw.Draw(canvas)
    grid = (42, 77, 91)
    axis = (126, 185, 198)
    text = (207, 231, 235)
    for fraction in np.linspace(0.0, 1.0, 9):
        x = left + round(float(fraction) * plot_width)
        draw.line((x, top, x, top + plot_height), fill=grid, width=1)
        value = x_range[0] + float(fraction) * (x_range[1] - x_range[0])
        draw.text((x - 16, top + plot_height + 8), f"{value:.1f}", fill=axis)
    for fraction in np.linspace(0.0, 1.0, 6):
        y = top + plot_height - round(float(fraction) * plot_height)
        draw.line((left, y, left + plot_width, y), fill=grid, width=1)
        value = y_range[0] + float(fraction) * (y_range[1] - y_range[0])
        draw.text((8, y - 6), f"{value:.1f}", fill=axis)
    draw.rectangle(
        (left, top, left + plot_width, top + plot_height), outline=axis, width=2
    )
    draw.text((left, 18), title, fill=text)
    draw.text((left + plot_width // 2 - 35, top + plot_height + 34), x_label, fill=text)
    draw.text((8, 18), y_label, fill=text)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _source_hash_record(benchmark_root: Path, source: Path) -> dict[str, Any]:
    dataset = load_json(benchmark_root / "manifests" / "dataset-manifest.json")
    for scene in dataset.get("scenes", []):
        for item in scene.get("inputs", []):
            if item.get("kind") != "point_cloud":
                continue
            reference = Path(str(item["path"]))
            resolved = (
                reference.resolve()
                if reference.is_absolute()
                else (benchmark_root / reference).resolve()
            )
            if resolved == source:
                return {
                    "path": str(source),
                    "bytes": source.stat().st_size,
                    "sha256": item.get("sha256"),
                    "hash_source": "dataset_manifest_declared",
                }
    return {
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": None,
        "hash_source": "not_declared",
    }


def render_neutral_rail_evidence(
    project: ProjectConfig,
    benchmark_root: str | Path,
    scene_id: str,
    start_m: float,
    end_m: float,
    spacing_m: float = 2.0,
    slice_half_width_m: float = 0.30,
    cross_min_m: float = -20.0,
    cross_max_m: float = 20.0,
    z_min_m: float | None = None,
    z_max_m: float | None = None,
    output_name: str = "rail_neutral_evidence_v1",
    chunk_size_points: int = 2_000_000,
) -> Path:
    """Render model-free rail annotation evidence directly from the source cloud."""

    if end_m <= start_m:
        raise ValueError("end_m must exceed start_m")
    if spacing_m <= 0 or slice_half_width_m <= 0:
        raise ValueError("spacing_m and slice_half_width_m must be positive")
    if cross_max_m <= cross_min_m:
        raise ValueError("cross_max_m must exceed cross_min_m")
    if not OUTPUT_NAME_PATTERN.fullmatch(output_name):
        raise ValueError("Invalid output_name")

    benchmark = Path(benchmark_root).resolve()
    output_root = benchmark / "annotations" / output_name
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite neutral evidence: {output_root}")
    output_root.mkdir(parents=True)

    split = load_json(benchmark / "manifests" / "split-manifest.json")
    blocks = [item for item in split["blocks"] if item["scene_id"] == scene_id]
    if not blocks:
        raise ValueError(f"No benchmark blocks found for scene {scene_id}")
    segment_manifest = load_json(project.workspace_path("segment_manifest"))
    segments = list(segment_manifest.get("segments", []))
    if not segments:
        raise ValueError("Project segment manifest contains no segments")

    camera_path = project.input_path("camera_csv")
    source = project.input_path("point_cloud")
    if camera_path is None or not camera_path.is_file():
        raise FileNotFoundError(camera_path)
    if source is None or not source.is_file():
        raise FileNotFoundError(source)
    trajectory = camera_trajectory(load_camera_rows(camera_path))
    context_start = max(float(trajectory[0]["distance_m"]), start_m - 25.0)
    context_end = min(float(trajectory[-1]["distance_m"]), end_m + 25.0)
    cameras = [
        item
        for item in trajectory
        if context_start <= float(item["distance_m"]) <= context_end
    ]
    if len(cameras) < 2:
        raise ValueError("Fewer than two cameras cover the requested chainage range")
    frame = fit_corridor_frame_dominant_axis_regression(cameras)
    camera_x = np.asarray([item["x"] for item in cameras], dtype=np.float64)
    camera_y = np.asarray([item["y"] for item in cameras], dtype=np.float64)
    camera_chainage = np.asarray(
        [item["distance_m"] for item in cameras], dtype=np.float64
    )
    camera_longitudinal, _ = frame.project(camera_x, camera_y)
    along_order = np.argsort(camera_longitudinal)
    along_sorted = camera_longitudinal[along_order]
    chainage_by_along = camera_chainage[along_order]
    chainage_order = np.argsort(camera_chainage)
    local_start, local_end = np.interp(
        [start_m, end_m],
        camera_chainage[chainage_order],
        camera_longitudinal[chainage_order],
    )
    local_min, local_max = sorted((float(local_start), float(local_end)))

    camera_z = np.asarray([item["z"] for item in cameras], dtype=np.float64)
    derived_z_min = float(np.median(camera_z) - 4.5)
    derived_z_max = float(np.median(camera_z) - 1.5)
    z_min = derived_z_min if z_min_m is None else float(z_min_m)
    z_max = derived_z_max if z_max_m is None else float(z_max_m)
    if z_max <= z_min:
        raise ValueError("z_max_m must exceed z_min_m")

    station_count = max(1, math.ceil((end_m - start_m) / spacing_m))
    stations = start_m + (np.arange(station_count, dtype=np.float64) + 0.5) * spacing_m
    stations = stations[stations < end_m + 1e-9]
    station_count = len(stations)
    cross_bins, z_bins = 800, 300
    plan_long_bin_m = 0.10
    plan_long_bins = max(1, math.ceil((end_m - start_m) / plan_long_bin_m))
    cross_hist = np.zeros((station_count, z_bins, cross_bins), dtype=np.uint16)
    plan_hist = np.zeros((plan_long_bins, cross_bins), dtype=np.uint16)
    station_point_count = np.zeros(station_count, dtype=np.int64)
    selected_point_count = 0

    with laspy.open(source) as reader:
        for points in reader.chunk_iterator(chunk_size_points):
            x = np.asarray(points.x, dtype=np.float64)
            y = np.asarray(points.y, dtype=np.float64)
            z = np.asarray(points.z, dtype=np.float64)
            longitudinal, cross = frame.project(x, y)
            broad = (
                (longitudinal >= local_min)
                & (longitudinal <= local_max)
                & (cross >= cross_min_m)
                & (cross <= cross_max_m)
                & (z >= z_min)
                & (z <= z_max)
            )
            if not np.any(broad):
                continue
            longitudinal = longitudinal[broad]
            cross = cross[broad]
            z = z[broad]
            chainage = np.interp(longitudinal, along_sorted, chainage_by_along)
            inside = (chainage >= start_m) & (chainage < end_m)
            if not np.any(inside):
                continue
            chainage = chainage[inside]
            cross = cross[inside]
            z = z[inside]
            selected_point_count += len(chainage)
            cross_index = np.clip(
                ((cross - cross_min_m) / (cross_max_m - cross_min_m) * cross_bins).astype(
                    np.int64
                ),
                0,
                cross_bins - 1,
            )
            long_index = np.clip(
                ((chainage - start_m) / plan_long_bin_m).astype(np.int64),
                0,
                plan_long_bins - 1,
            )
            np.add.at(plan_hist, (long_index, cross_index), 1)

            station_index = np.floor((chainage - start_m) / spacing_m).astype(
                np.int64
            )
            valid_station = (station_index >= 0) & (station_index < station_count)
            station_index = station_index[valid_station]
            task_distance = np.abs(chainage[valid_station] - stations[station_index])
            sliced = task_distance <= slice_half_width_m
            if not np.any(sliced):
                continue
            station_index = station_index[sliced]
            sliced_cross = cross[valid_station][sliced]
            sliced_z = z[valid_station][sliced]
            sliced_cross_index = np.clip(
                (
                    (sliced_cross - cross_min_m)
                    / (cross_max_m - cross_min_m)
                    * cross_bins
                ).astype(np.int64),
                0,
                cross_bins - 1,
            )
            z_index = np.clip(
                ((sliced_z - z_min) / (z_max - z_min) * z_bins).astype(np.int64),
                0,
                z_bins - 1,
            )
            np.add.at(cross_hist, (station_index, z_index, sliced_cross_index), 1)
            np.add.at(station_point_count, station_index, 1)

    evidence_root = output_root / "evidence"
    station_records: list[dict[str, Any]] = []
    for index, station in enumerate(stations):
        task_id = f"RAIL-TRUTH-{round(station * 1000):06d}"
        cross_path = evidence_root / f"{task_id}-cross-section.png"
        plan_path = evidence_root / f"{task_id}-plan-strip.png"
        _render_panel(
            cross_hist[index],
            cross_path,
            f"CROSS SECTION | chainage {station:.2f} m | raw point cloud",
            (cross_min_m, cross_max_m),
            (z_min, z_max),
            "cross-track m",
            "Z m",
        )
        center_bin = int((station - start_m) / plan_long_bin_m)
        radius = max(1, round(5.0 / plan_long_bin_m))
        left_bin = max(0, center_bin - radius)
        right_bin = min(plan_long_bins, center_bin + radius)
        plan_crop = plan_hist[left_bin:right_bin].T
        _render_panel(
            plan_crop,
            plan_path,
            f"PLAN STRIP | chainage {station:.2f} m | raw point cloud",
            (
                start_m + left_bin * plan_long_bin_m,
                start_m + right_bin * plan_long_bin_m,
            ),
            (cross_min_m, cross_max_m),
            "chainage m",
            "cross m",
        )
        segment = _find_interval(
            segments, float(station), "chainage_start_m", "chainage_end_m"
        )
        block = _find_interval(blocks, float(station), "core_start_m", "core_end_m")
        station_records.append(
            {
                "task_id": task_id,
                "segment_id": segment["id"],
                "block_id": block["block_id"],
                "chainage_m": float(station),
                "cross_section_image": str(cross_path),
                "plan_strip_image": str(plan_path),
                "point_count": int(station_point_count[index]),
                "cross_min_m": cross_min_m,
                "cross_max_m": cross_max_m,
                "z_min_m": z_min,
                "z_max_m": z_max,
                "slice_half_width_m": slice_half_width_m,
            }
        )

    manifest = {
        "schema_version": "railway.rail-neutral-evidence.v1",
        "scene_id": scene_id,
        "created_at": datetime.now(UTC).isoformat(),
        "evidence_source": "raw_point_cloud",
        "model_overlay": False,
        "candidate_overlay": False,
        "source_point_cloud": _source_hash_record(benchmark, source),
        "camera_pose_path": str(camera_path),
        "frame": frame.to_json(),
        "frame_source": "camera_trajectory_only",
        "chainage_range_m": [start_m, end_m],
        "spacing_m": spacing_m,
        "cross_range_m": [cross_min_m, cross_max_m],
        "z_range_m": [z_min, z_max],
        "z_range_source": (
            "camera_median_offsets" if z_min_m is None and z_max_m is None else "explicit"
        ),
        "selected_point_count": selected_point_count,
        "stations": station_records,
        "limitations": [
            "This package is annotation evidence, not an algorithm prediction.",
            "The chainage frame is derived from camera trajectory only.",
            "No model, TrackGraph, rail candidate or fit heatmap is rendered.",
        ],
    }
    manifest_path = output_root / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path
