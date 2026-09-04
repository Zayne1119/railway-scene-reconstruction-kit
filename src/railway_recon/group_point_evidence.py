from __future__ import annotations

from pathlib import Path
from typing import Any

import laspy
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .io import load_json, write_json
from .vertical_candidate_evidence import _rgb8


def _draw_group_projection(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    points: np.ndarray,
    colours: np.ndarray,
    candidate_boxes: list[dict[str, Any]],
    axes: tuple[int, int],
    labels: tuple[str, str],
    title: str,
) -> None:
    left, top, right, bottom = box
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=14)
    draw.rectangle(box, fill=(4, 18, 26), outline=(50, 111, 128), width=2)
    draw.text((left + 18, top + 12), title, fill=(222, 241, 245), font=font)
    plot = (left + 58, top + 52, right - 24, bottom - 42)
    x = points[:, axes[0]]
    y = points[:, axes[1]]
    minimum = np.asarray((np.min(x), np.min(y)), dtype=np.float64)
    maximum = np.asarray((np.max(x), np.max(y)), dtype=np.float64)
    extent = np.maximum(maximum - minimum, 1.0e-6)
    width = plot[2] - plot[0]
    height = plot[3] - plot[1]
    scale = min(width / extent[0], height / extent[1])

    def project(values: np.ndarray) -> tuple[int, int]:
        px = round(plot[0] + (values[0] - minimum[0]) * scale)
        py = round(plot[3] - (values[1] - minimum[1]) * scale)
        return px, py

    px = np.rint(plot[0] + (x - minimum[0]) * scale).astype(np.int32)
    py = np.rint(plot[3] - (y - minimum[1]) * scale).astype(np.int32)
    for index in range(len(points)):
        colour = tuple(int(value * 0.72) for value in colours[index])
        draw.point((int(px[index]), int(py[index])), fill=colour)

    palette = ((174, 255, 81), (255, 158, 58), (69, 212, 255), (216, 108, 255))
    for index, candidate in enumerate(candidate_boxes):
        local_minimum = np.asarray(candidate["local_minimum"], dtype=np.float64)[list(axes)]
        local_maximum = np.asarray(candidate["local_maximum"], dtype=np.float64)[list(axes)]
        p0 = project(np.asarray((local_minimum[0], local_maximum[1])))
        p1 = project(np.asarray((local_maximum[0], local_minimum[1])))
        colour = palette[index % len(palette)]
        draw.rectangle((p0[0], p0[1], p1[0], p1[1]), outline=colour, width=2)
        draw.text((p0[0] + 3, p0[1] + 2), candidate["candidate_id"].split("-")[-1], fill=colour, font=small)

    draw.text((left + 18, bottom - 30), labels[0], fill=(121, 205, 221), font=small)
    draw.text((right - 78, top + 42), labels[1], fill=(121, 205, 221), font=small)


def render_group_point_evidence(
    *,
    cloud_path: str | Path,
    gap_report_path: str | Path,
    frame_report_path: str | Path,
    output_directory: str | Path,
    candidate_ids: tuple[str, ...],
    station_margin_m: float = 4.0,
    cross_margin_m: float = 4.0,
    z_margin_m: float = 1.0,
    maximum_points: int = 500_000,
    chunk_size: int = 2_000_000,
) -> dict[str, Path]:
    if not candidate_ids:
        raise ValueError("candidate_ids cannot be empty")
    cloud = Path(cloud_path).resolve()
    gap_report = load_json(Path(gap_report_path))
    frame = load_json(Path(frame_report_path))["frame"]
    requested = set(candidate_ids)
    candidates = [
        item for item in gap_report["vertical_candidates"] if item["candidate_id"] in requested
    ]
    found = {item["candidate_id"] for item in candidates}
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"Unknown candidate IDs: {', '.join(missing)}")

    origin_xy = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    station_bounds = (
        min(float(item["station_m"]) for item in candidates) - station_margin_m,
        max(float(item["station_m"]) for item in candidates) + station_margin_m,
    )
    cross_bounds = (
        min(float(item["cross_m"]) for item in candidates) - cross_margin_m,
        max(float(item["cross_m"]) for item in candidates) + cross_margin_m,
    )
    z_bounds = (
        min(float(item["minimum_xyz_m"][2]) for item in candidates) - z_margin_m,
        max(float(item["maximum_xyz_m"][2]) for item in candidates) + z_margin_m,
    )
    point_parts: list[np.ndarray] = []
    colour_parts: list[np.ndarray] = []
    with laspy.open(cloud) as reader:
        for chunk in reader.chunk_iterator(chunk_size):
            x = np.asarray(chunk.x, dtype=np.float64)
            y = np.asarray(chunk.y, dtype=np.float64)
            z = np.asarray(chunk.z, dtype=np.float64)
            local_xy = np.column_stack((x, y)) - origin_xy
            station = local_xy @ along
            lateral = local_xy @ cross
            selected = (
                (station >= station_bounds[0])
                & (station <= station_bounds[1])
                & (lateral >= cross_bounds[0])
                & (lateral <= cross_bounds[1])
                & (z >= z_bounds[0])
                & (z <= z_bounds[1])
            )
            if np.any(selected):
                point_parts.append(np.column_stack((station[selected], lateral[selected], z[selected])))
                colour_parts.append(
                    _rgb8(
                        np.asarray(chunk.red)[selected],
                        np.asarray(chunk.green)[selected],
                        np.asarray(chunk.blue)[selected],
                    )
                )
    if not point_parts:
        raise ValueError("No points were found inside the candidate group crop")
    points = np.vstack(point_parts)
    colours = np.vstack(colour_parts)
    source_point_count = len(points)
    if len(points) > maximum_points:
        stride = int(np.ceil(len(points) / maximum_points))
        points = points[::stride]
        colours = colours[::stride]

    candidate_boxes: list[dict[str, Any]] = []
    for item in candidates:
        world_minimum = np.asarray(item["minimum_xyz_m"], dtype=np.float64)
        world_maximum = np.asarray(item["maximum_xyz_m"], dtype=np.float64)
        world_corners_xy = np.asarray(
            (
                (world_minimum[0], world_minimum[1]),
                (world_minimum[0], world_maximum[1]),
                (world_maximum[0], world_minimum[1]),
                (world_maximum[0], world_maximum[1]),
            ),
            dtype=np.float64,
        )
        local_xy = world_corners_xy - origin_xy
        local_station = local_xy @ along
        local_cross = local_xy @ cross
        candidate_boxes.append(
            {
                "candidate_id": item["candidate_id"],
                "local_minimum": [
                    float(np.min(local_station)),
                    float(np.min(local_cross)),
                    float(world_minimum[2]),
                ],
                "local_maximum": [
                    float(np.max(local_station)),
                    float(np.max(local_cross)),
                    float(world_maximum[2]),
                ],
            }
        )

    canvas = Image.new("RGB", (1900, 1250), (3, 13, 20))
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default(size=24)
    small = ImageFont.load_default(size=17)
    draw.text(
        (28, 22),
        "GROUPED DENSE POINT-CLOUD EVIDENCE / CANDIDATE BOXES ARE REVIEW AIDS ONLY",
        fill=(174, 255, 81),
        font=title_font,
    )
    draw.text(
        (28, 59),
        (
            f"candidates {len(candidates)} | source crop {source_point_count:,} points | "
            f"rendered {len(points):,} | S {station_bounds[0]:.2f}..{station_bounds[1]:.2f} m | "
            f"C {cross_bounds[0]:.2f}..{cross_bounds[1]:.2f} m | Z {z_bounds[0]:.2f}..{z_bounds[1]:.2f} m"
        ),
        fill=(169, 204, 212),
        font=small,
    )
    _draw_group_projection(
        canvas,
        (28, 98, 1872, 445),
        points,
        colours,
        candidate_boxes,
        (0, 1),
        ("station", "cross"),
        "PLAN / station-cross",
    )
    _draw_group_projection(
        canvas,
        (28, 465, 940, 1220),
        points,
        colours,
        candidate_boxes,
        (0, 2),
        ("station", "Z"),
        "LONGITUDINAL ELEVATION / station-Z",
    )
    _draw_group_projection(
        canvas,
        (960, 465, 1872, 1220),
        points,
        colours,
        candidate_boxes,
        (1, 2),
        ("cross", "Z"),
        "CROSS SECTION / cross-Z",
    )

    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    figure = output / "group_point_evidence.png"
    canvas.save(figure)
    manifest = output / "group_point_evidence.json"
    write_json(
        manifest,
        {
            "schema_version": "railway.group-point-evidence.v1",
            "cloud": str(cloud),
            "candidate_ids": list(candidate_ids),
            "crop_bounds_local": {
                "station_m": list(station_bounds),
                "cross_m": list(cross_bounds),
                "z_m": list(z_bounds),
            },
            "source_crop_point_count": source_point_count,
            "rendered_point_count": len(points),
            "candidate_boxes": candidate_boxes,
            "figure": str(figure),
            "status": "review_evidence_only_geometry_unchanged",
        },
    )
    return {"figure": figure, "manifest": manifest}
