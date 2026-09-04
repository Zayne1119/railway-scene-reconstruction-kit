from __future__ import annotations

from pathlib import Path
from typing import Any

import laspy
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .io import load_json, write_json


def _rgb8(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    rgb = np.column_stack((red, green, blue)).astype(np.float64)
    if not np.any(rgb):
        return np.full((len(rgb), 3), 174, dtype=np.uint8)
    scale = 257.0 if float(np.percentile(rgb, 99)) > 255.0 else 1.0
    return np.clip(rgb / scale, 0.0, 255.0).astype(np.uint8)


def _draw_projection(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    points: np.ndarray,
    colours: np.ndarray,
    candidate_mask: np.ndarray,
    axes: tuple[int, int],
    labels: tuple[str, str],
    title: str,
    preserve_candidate_colour: bool,
) -> None:
    left, top, right, bottom = box
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=15)
    draw.rectangle(box, fill=(4, 18, 26), outline=(50, 111, 128), width=2)
    draw.text((left + 18, top + 12), title, fill=(222, 241, 245), font=font)
    plot = (left + 55, top + 52, right - 22, bottom - 42)
    x = points[:, axes[0]]
    y = points[:, axes[1]]
    minimum = np.asarray((np.min(x), np.min(y)), dtype=np.float64)
    maximum = np.asarray((np.max(x), np.max(y)), dtype=np.float64)
    extent = np.maximum(maximum - minimum, 1.0e-6)
    width = plot[2] - plot[0]
    height = plot[3] - plot[1]
    scale = min(width / extent[0], height / extent[1])
    px = np.rint(plot[0] + (x - minimum[0]) * scale).astype(np.int32)
    py = np.rint(plot[3] - (y - minimum[1]) * scale).astype(np.int32)
    context_indexes = np.flatnonzero(~candidate_mask)
    for index in context_indexes:
        colour = tuple(int(value * 0.48) for value in colours[index])
        draw.point((int(px[index]), int(py[index])), fill=colour)
    for index in np.flatnonzero(candidate_mask):
        position = (int(px[index]), int(py[index]))
        fill = (
            tuple(int(value) for value in colours[index])
            if preserve_candidate_colour
            else (174, 255, 81)
        )
        draw.ellipse(
            (position[0] - 1, position[1] - 1, position[0] + 1, position[1] + 1),
            fill=fill,
        )
    draw.text((left + 18, bottom - 30), labels[0], fill=(121, 205, 221), font=small)
    draw.text((right - 70, top + 42), labels[1], fill=(121, 205, 221), font=small)


def render_vertical_candidate_detail_evidence(
    *,
    cloud_path: str | Path,
    gap_report_path: str | Path,
    frame_report_path: str | Path,
    output_directory: str | Path,
    crop_half_width_m: float = 2.0,
    maximum_points_per_candidate: int = 180_000,
    chunk_size: int = 2_000_000,
    priorities: tuple[str, ...] = ("P0",),
    candidate_ids: tuple[str, ...] = (),
    asset_relations: tuple[str, ...] = ("new_standalone_vertical_candidate",),
    current_segment_only: bool = False,
    preserve_candidate_colour: bool = False,
) -> dict[str, Path]:
    cloud = Path(cloud_path).resolve()
    gap_report = load_json(Path(gap_report_path))
    frame = load_json(Path(frame_report_path))["frame"]
    requested_ids = set(candidate_ids)
    allowed_asset_relations = set(asset_relations)
    candidates = [
        item
        for item in gap_report["vertical_candidates"]
        if item.get("priority") in priorities
        and item.get("asset_relation") in allowed_asset_relations
        and (not requested_ids or str(item.get("candidate_id")) in requested_ids)
        and (
            not current_segment_only
            or item.get("corridor_scope_ownership") == "within_current_segment"
        )
    ]
    if not candidates:
        raise ValueError("No matching vertical candidates found")
    origin_xy = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    samples: dict[str, list[np.ndarray]] = {item["candidate_id"]: [] for item in candidates}
    colours: dict[str, list[np.ndarray]] = {item["candidate_id"]: [] for item in candidates}
    masks: dict[str, list[np.ndarray]] = {item["candidate_id"]: [] for item in candidates}
    with laspy.open(cloud) as reader:
        for chunk in reader.chunk_iterator(chunk_size):
            x = np.asarray(chunk.x, dtype=np.float64)
            y = np.asarray(chunk.y, dtype=np.float64)
            z = np.asarray(chunk.z, dtype=np.float64)
            local_xy = np.column_stack((x, y)) - origin_xy
            station = local_xy @ along
            lateral = local_xy @ cross
            rgb = _rgb8(
                np.asarray(chunk.red), np.asarray(chunk.green), np.asarray(chunk.blue)
            )
            for candidate in candidates:
                station_center = float(candidate["station_m"])
                cross_center = float(candidate["cross_m"])
                z_min = float(candidate["minimum_xyz_m"][2])
                z_max = float(candidate["maximum_xyz_m"][2])
                selected = (
                    (np.abs(station - station_center) <= crop_half_width_m)
                    & (np.abs(lateral - cross_center) <= crop_half_width_m)
                    & (z >= z_min - 1.0)
                    & (z <= z_max + 1.0)
                )
                if not np.any(selected):
                    continue
                selected_points = np.column_stack(
                    (station[selected], lateral[selected], z[selected])
                )
                world_selected = np.column_stack((x[selected], y[selected], z[selected]))
                minimum = np.asarray(candidate["minimum_xyz_m"], dtype=np.float64)
                maximum = np.asarray(candidate["maximum_xyz_m"], dtype=np.float64)
                candidate_selected = np.all(
                    (world_selected >= minimum[None, :])
                    & (world_selected <= maximum[None, :]),
                    axis=1,
                )
                candidate_id = str(candidate["candidate_id"])
                samples[candidate_id].append(selected_points)
                colours[candidate_id].append(rgb[selected])
                masks[candidate_id].append(candidate_selected)
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        points = np.vstack(samples[candidate_id])
        colour = np.vstack(colours[candidate_id])
        candidate_mask = np.concatenate(masks[candidate_id])
        if len(points) > maximum_points_per_candidate:
            stride = int(np.ceil(len(points) / maximum_points_per_candidate))
            points = points[::stride]
            colour = colour[::stride]
            candidate_mask = candidate_mask[::stride]
        canvas = Image.new("RGB", (1800, 1120), (3, 13, 20))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default(size=23)
        small = ImageFont.load_default(size=17)
        draw.text(
            (28, 24),
            f"{candidate_id} / DENSE COLOURED POINT-CLOUD THREE-VIEW REVIEW",
            fill=(174, 255, 81),
            font=font,
        )
        draw.text(
            (28, 60),
            (
                f"station {candidate['station_m']:.2f} m | cross "
                f"{candidate['cross_m']:.2f} m | height {candidate['extent_xyz_m'][2]:.2f} m | "
                f"crop points {len(points):,} | "
                + (
                    "candidate = source RGB"
                    if preserve_candidate_colour
                    else "green = detector candidate"
                )
            ),
            fill=(169, 204, 212),
            font=small,
        )
        _draw_projection(
            canvas,
            (28, 105, 1772, 420),
            points,
            colour,
            candidate_mask,
            (0, 1),
            ("station", "cross"),
            "PLAN / station-cross",
            preserve_candidate_colour,
        )
        _draw_projection(
            canvas,
            (28, 440, 890, 1090),
            points,
            colour,
            candidate_mask,
            (0, 2),
            ("station", "Z"),
            "LONGITUDINAL ELEVATION / station-Z",
            preserve_candidate_colour,
        )
        _draw_projection(
            canvas,
            (910, 440, 1772, 1090),
            points,
            colour,
            candidate_mask,
            (1, 2),
            ("cross", "Z"),
            "CROSS SECTION / cross-Z",
            preserve_candidate_colour,
        )
        path = output / f"{candidate_id.lower()}_three_view.png"
        canvas.save(path)
        outputs[candidate_id] = path
        records.append(
            {
                "candidate_id": candidate_id,
                "station_m": candidate["station_m"],
                "cross_m": candidate["cross_m"],
                "rendered_crop_point_count": len(points),
                "rendered_candidate_point_count": int(np.sum(candidate_mask)),
                "evidence_figure": str(path),
                "decision": "manual_semantic_review_required",
            }
        )
    manifest = output / "vertical_candidate_detail_evidence.json"
    write_json(
        manifest,
        {
            "schema_version": "railway.vertical-candidate-detail-evidence.v1",
            "cloud": str(cloud),
            "selection": {
                "priorities": list(priorities),
                "candidate_ids": list(candidate_ids),
                "asset_relations": list(asset_relations),
                "current_segment_only": current_segment_only,
                "preserve_candidate_colour": preserve_candidate_colour,
            },
            "candidate_count": len(records),
            "candidates": records,
        },
    )
    outputs["manifest"] = manifest
    return outputs
