from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .config import ProjectConfig
from .io import load_json, write_json


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _component_type(spans: np.ndarray, bottom_delta: float, dark_fraction: float) -> str:
    longitudinal, cross, height = (float(value) for value in spans)
    if (
        0.7 <= longitudinal <= 3.5
        and cross <= 0.9
        and 1.2 <= height <= 3.0
        and bottom_delta <= 0.35
        and dark_fraction >= 0.25
    ):
        return "freestanding_sign_candidate"
    if longitudinal >= 3.0 and cross <= 0.8 and 0.6 <= height <= 2.0:
        return "railing_or_fence_candidate"
    if (
        longitudinal <= 2.5
        and cross <= 2.0
        and 0.3 <= height <= 2.5
        and bottom_delta <= 0.45
    ):
        return "equipment_box_or_small_asset_candidate"
    return "unresolved_above_platform_component"


def _extract_side(
    side_id: str,
    s: np.ndarray,
    c: np.ndarray,
    z: np.ndarray,
    rgb: np.ndarray,
    *,
    s_range: tuple[float, float],
    c_range: tuple[float, float],
    platform_z: float,
    voxel_m: float,
    minimum_voxel_points: int,
    minimum_component_points: int,
    minimum_component_voxels: int,
) -> dict[str, Any]:
    mask = (
        (s >= s_range[0])
        & (s <= s_range[1])
        & (c >= c_range[0])
        & (c <= c_range[1])
        & (z >= platform_z + 0.12)
        & (z <= platform_z + 3.2)
    )
    selected = np.flatnonzero(mask)
    ss, cc, zz, colours = s[selected], c[selected], z[selected], rgb[selected]
    shape = tuple(
        int(np.ceil((high - low) / voxel_m)) + 1
        for low, high in (s_range, c_range, (platform_z + 0.12, platform_z + 3.2))
    )
    si = np.clip(((ss - s_range[0]) / voxel_m).astype(np.int32), 0, shape[0] - 1)
    ci = np.clip(((cc - c_range[0]) / voxel_m).astype(np.int32), 0, shape[1] - 1)
    zi = np.clip(
        ((zz - (platform_z + 0.12)) / voxel_m).astype(np.int32),
        0,
        shape[2] - 1,
    )
    linear = np.ravel_multi_index((si, ci, zi), shape)
    unique, counts = np.unique(linear, return_counts=True)
    occupied = np.zeros(int(np.prod(shape)), dtype=bool)
    occupied[unique[counts >= minimum_voxel_points]] = True
    occupied = occupied.reshape(shape)
    labels, label_count = ndimage.label(
        occupied, structure=np.ones((3, 3, 3), dtype=bool)
    )
    point_labels = labels[si, ci, zi]
    records = []
    for label_id in range(1, label_count + 1):
        voxel_count = int(np.count_nonzero(labels == label_id))
        indexes = np.flatnonzero(point_labels == label_id)
        if voxel_count < minimum_component_voxels or indexes.size < minimum_component_points:
            continue
        values = np.column_stack((ss[indexes], cc[indexes], zz[indexes]))
        minimum, maximum = values.min(axis=0), values.max(axis=0)
        spans = maximum - minimum
        if spans[2] < 0.25:
            continue
        component_colours = colours[indexes].astype(np.float64)
        dark_fraction = float(np.mean(np.max(component_colours, axis=1) <= 85.0))
        bottom_delta = float(minimum[2] - platform_z)
        records.append(
            {
                "temporary_label": int(label_id),
                "point_count": int(indexes.size),
                "occupied_voxel_count": voxel_count,
                "minimum_s_c_z_m": minimum.tolist(),
                "maximum_s_c_z_m": maximum.tolist(),
                "centroid_s_c_z_m": np.median(values, axis=0).tolist(),
                "span_s_c_z_m": spans.tolist(),
                "bottom_above_platform_m": bottom_delta,
                "median_rgb": [
                    round(value) for value in np.median(component_colours, axis=0)
                ],
                "dark_point_fraction": dark_fraction,
                "candidate_class": _component_type(spans, bottom_delta, dark_fraction),
            }
        )
    records.sort(key=lambda item: item["centroid_s_c_z_m"][0])
    for index, record in enumerate(records, start=1):
        record["id"] = f"{side_id.upper()}-SMALL-ASSET-CANDIDATE-{index:03d}"
        record["status"] = "point_cloud_candidate_photo_review_required"
    return {
        "side": side_id,
        "search_envelope": {
            "longitudinal_range_m": list(s_range),
            "cross_range_m": list(c_range),
            "z_range_m": [platform_z + 0.12, platform_z + 3.2],
            "platform_reference_z_m": platform_z,
        },
        "selected_point_count": int(selected.size),
        "voxel_shape": list(shape),
        "raw_connected_component_count": int(label_count),
        "retained_candidate_count": len(records),
        "candidates": records,
    }


def _render(report: dict[str, Any], output: Path) -> None:
    sides = report["sides"]
    panel_width, panel_height = 760, 780
    width = 80 + panel_width * len(sides) + 40 * max(len(sides) - 1, 0)
    height = 930
    image = Image.new("RGB", (width, height), "#06141d")
    draw = ImageDraw.Draw(image)
    draw.text((28, 20), "SMALL ABOVE-PLATFORM ASSET CANDIDATES", fill="#eaf9ff", font=_font(24))
    draw.text(
        (28, 58),
        "Connected point components only; geometry remains blocked until photo semantics pass.",
        fill="#9bb1b9",
        font=_font(16),
    )
    palette = {
        "freestanding_sign_candidate": "#b7ff3c",
        "railing_or_fence_candidate": "#24d7ff",
        "equipment_box_or_small_asset_candidate": "#ff9f31",
        "unresolved_above_platform_component": "#aa8cff",
    }
    for panel_index, side in enumerate(sides):
        x0 = 40 + panel_index * (panel_width + 40)
        y0, x1, y1 = 110, x0 + panel_width, 890
        draw.rectangle((x0, y0, x1, y1), outline="#28505e", width=2)
        draw.text((x0 + 14, y0 + 12), side["side"].upper(), fill="#b7ff3c", font=_font(18))
        envelope = side["search_envelope"]
        s_low, s_high = envelope["longitudinal_range_m"]
        c_low, c_high = envelope["cross_range_m"]

        def px(
            value: float,
            panel_x0: float = x0,
            minimum_s: float = s_low,
            maximum_s: float = s_high,
        ) -> float:
            return panel_x0 + 35 + (value - minimum_s) / (maximum_s - minimum_s) * (panel_width - 70)

        def py(
            value: float,
            panel_y1: float = y1,
            minimum_c: float = c_low,
            maximum_c: float = c_high,
        ) -> float:
            return panel_y1 - 40 - (value - minimum_c) / (maximum_c - minimum_c) * (panel_height - 105)

        for candidate in side["candidates"]:
            minimum, maximum = candidate["minimum_s_c_z_m"], candidate["maximum_s_c_z_m"]
            box = (px(minimum[0]), py(maximum[1]), px(maximum[0]), py(minimum[1]))
            colour = palette[candidate["candidate_class"]]
            draw.rectangle(box, outline=colour, width=3)
            draw.text((box[0], max(y0 + 42, box[1] - 16)), candidate["id"].split("-")[-1], fill=colour, font=_font(13))
        draw.text(
            (x0 + 14, y1 - 26),
            f"points={side['selected_point_count']:,} candidates={side['retained_candidate_count']}",
            fill="#9bb1b9",
            font=_font(14),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def analyze_small_platform_asset_candidates(
    project: ProjectConfig,
    segment_id: str,
    platform_report_path: str | Path,
    ownership_plan_path: str | Path,
    *,
    overwrite: bool = False,
    voxel_m: float = 0.15,
    minimum_voxel_points: int = 3,
    minimum_component_points: int = 30,
    minimum_component_voxels: int = 6,
) -> dict[str, Any]:
    platform_path = project.resolve(platform_report_path)
    ownership_path = project.resolve(ownership_plan_path)
    source = project.workspace_path("segments") / f"{segment_id}.laz"
    for path in (platform_path, ownership_path, source):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = project.workspace_path("reports") / f"{segment_id}_small_platform_asset_candidates.json"
    diagnostic = project.workspace_path("reports") / f"{segment_id}_small_platform_asset_candidates.png"
    if not overwrite and (output.exists() or diagnostic.exists()):
        raise FileExistsError("Refusing to overwrite small-asset candidate outputs")
    platform = load_json(platform_path)
    ownership = load_json(ownership_path)
    segment = next(
        item for item in ownership["segments"] if item["segment_id"] == segment_id
    )
    shared_frame = ownership["frame"]
    shared_origin = np.asarray(shared_frame["origin_xy"], dtype=np.float64)
    shared_along = np.asarray(shared_frame["along_xy"], dtype=np.float64)
    local_frame = platform["frame"]
    local_origin = np.asarray(local_frame["origin_xy"], dtype=np.float64)
    local_along = np.asarray(local_frame["along_xy"], dtype=np.float64)
    local_cross = np.asarray(local_frame["cross_xy"], dtype=np.float64)
    boundary_points = [
        shared_origin + shared_along * float(value) for value in segment["owned_interval_m"]
    ]
    local_s_range = tuple(
        sorted(float(np.dot(point - local_origin, local_along)) for point in boundary_points)
    )
    las = laspy.read(source)
    x, y, z = (
        np.asarray(las.x, dtype=np.float64),
        np.asarray(las.y, dtype=np.float64),
        np.asarray(las.z, dtype=np.float64),
    )
    rgb = np.column_stack((las.red, las.green, las.blue)).astype(np.uint8)
    delta = np.column_stack((x - local_origin[0], y - local_origin[1]))
    s, c = delta @ local_along, delta @ local_cross
    sides = []
    for index, component in enumerate(platform["platform_components"], start=1):
        side_name = str(component.get("side", f"platform-{index}"))
        if any(item["side"] == side_name for item in sides):
            side_name = f"{side_name}-{index}"
        sides.append(
            _extract_side(
                side_name,
                s,
                c,
                z,
                rgb,
                s_range=local_s_range,
                c_range=tuple(float(value) for value in component["cross_range_m"]),
                platform_z=float(np.median(component["z_range_m"])),
                voxel_m=voxel_m,
                minimum_voxel_points=minimum_voxel_points,
                minimum_component_points=minimum_component_points,
                minimum_component_voxels=minimum_component_voxels,
            )
        )
    report = {
        "schema_version": "railway.small-platform-asset-candidates.v1",
        "project_id": project.project_id,
        "segment_id": segment_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "source": str(source),
        "source_policy": "read_only",
        "platform_report": str(platform_path),
        "ownership_plan": str(ownership_path),
        "owned_interval_global_station_m": segment["owned_interval_m"],
        "voxel_settings": {
            "voxel_m": voxel_m,
            "minimum_voxel_points": minimum_voxel_points,
            "minimum_component_points": minimum_component_points,
            "minimum_component_voxels": minimum_component_voxels,
        },
        "sides": sides,
        "candidate_count": sum(item["retained_candidate_count"] for item in sides),
        "geometry_write": False,
        "asset_registry_write": False,
        "status": "point_cloud_candidates_generated_photo_review_required",
        "diagnostic": str(diagnostic),
        "limitations": [
            "Connected components are geometric candidates, not asset semantics.",
            "Canopy columns, signs, machinery and facade fragments can occupy the same size range.",
            "No mesh may be generated until multi-view photo and platform-contact gates pass.",
        ],
    }
    write_json(output, report)
    _render(report, diagnostic)
    return report
