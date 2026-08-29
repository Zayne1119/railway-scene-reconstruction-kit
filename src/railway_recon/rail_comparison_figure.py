from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .camera import camera_trajectory, load_camera_rows
from .geometry import CorridorFrame
from .io import load_json, sha256_file, write_json

PLOT_LEFT = 76
PLOT_TOP = 54
PLOT_WIDTH = 1000
PLOT_HEIGHT = 420


def _prediction_points(
    report: dict[str, Any],
    route_xy: np.ndarray,
    evidence_frame: CorridorFrame,
) -> list[tuple[float, float]]:
    frame = CorridorFrame.from_json(report["frame"])
    local_longitudinal, _ = frame.project(
        np.asarray([route_xy[0]]), np.asarray([route_xy[1]])
    )
    longitudinal = float(local_longitudinal[0])
    points: list[tuple[float, float]] = []
    for line in report.get("rail_lines", []):
        cross = float(line["cross_fit_slope_m_per_m"]) * longitudinal + float(
            line["cross_fit_intercept_m"]
        )
        z = float(line["z_fit_slope_m_per_m"]) * longitudinal + float(
            line["z_fit_intercept_m"]
        )
        world = frame.world_xy(np.asarray([longitudinal]), np.asarray([cross]))[0]
        _, evidence_cross = evidence_frame.project(
            np.asarray([world[0]]), np.asarray([world[1]])
        )
        points.append((float(evidence_cross[0]), z))
    return points


def _pixel(
    cross: float,
    z: float,
    cross_range: tuple[float, float],
    z_range: tuple[float, float],
) -> tuple[int, int]:
    x_fraction = (cross - cross_range[0]) / (cross_range[1] - cross_range[0])
    y_fraction = (z - z_range[0]) / (z_range[1] - z_range[0])
    return (
        PLOT_LEFT + round(x_fraction * PLOT_WIDTH),
        PLOT_TOP + PLOT_HEIGHT - round(y_fraction * PLOT_HEIGHT),
    )


def _draw_marker(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    color: tuple[int, int, int],
    shape: str,
) -> None:
    x, y = xy
    radius = 7
    draw.line((x, y - 22, x, y + 22), fill=color, width=3)
    if shape == "circle":
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
    else:
        draw.polygon(
            ((x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)),
            outline=color,
        )


def render_rail_method_comparison(
    evidence_manifest_path: str | Path,
    baseline_reports: list[tuple[str, str | Path]],
    method_reports: list[tuple[str, str | Path]],
    output_dir: str | Path,
) -> Path:
    evidence_source = Path(evidence_manifest_path).resolve()
    evidence = load_json(evidence_source)
    if evidence.get("schema_version") != "railway.rail-neutral-evidence.v1":
        raise ValueError("Unsupported neutral rail evidence manifest")
    if evidence.get("model_overlay") or evidence.get("candidate_overlay"):
        raise ValueError("Comparison background must be neutral raw-point evidence")
    baseline_paths = {segment: Path(path).resolve() for segment, path in baseline_reports}
    method_paths = {segment: Path(path).resolve() for segment, path in method_reports}
    if not baseline_paths or set(baseline_paths) != set(method_paths):
        raise ValueError("Baseline and method report segments must match")
    baseline = {segment: load_json(path) for segment, path in baseline_paths.items()}
    method = {segment: load_json(path) for segment, path in method_paths.items()}
    for segment in baseline_paths:
        if baseline[segment].get("segment_id") != segment or method[segment].get(
            "segment_id"
        ) != segment:
            raise ValueError(f"Prediction report segment mismatch: {segment}")

    output_root = Path(output_dir).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite comparison figures: {output_root}")
    output_root.mkdir(parents=True)
    trajectory = camera_trajectory(load_camera_rows(Path(evidence["camera_pose_path"])))
    route_chainage = np.asarray([item["distance_m"] for item in trajectory], dtype=np.float64)
    route_x = np.asarray([item["x"] for item in trajectory], dtype=np.float64)
    route_y = np.asarray([item["y"] for item in trajectory], dtype=np.float64)
    evidence_frame = CorridorFrame.from_json(evidence["frame"])

    selected_stations: list[dict[str, Any]] = []
    for segment in sorted(baseline_paths):
        stations = [item for item in evidence["stations"] if item["segment_id"] == segment]
        if not stations:
            raise ValueError(f"Neutral evidence has no station for {segment}")
        station = min(
            stations,
            key=lambda item: abs(
                float(item["chainage_m"])
                - float(np.median([value["chainage_m"] for value in stations]))
            ),
        )
        chainage = float(station["chainage_m"])
        route_xy = np.asarray(
            [
                np.interp(chainage, route_chainage, route_x),
                np.interp(chainage, route_chainage, route_y),
            ],
            dtype=np.float64,
        )
        cross_range = float(station["cross_min_m"]), float(station["cross_max_m"])
        z_range = float(station["z_min_m"]), float(station["z_max_m"])
        canvas = Image.open(station["cross_section_image"]).convert("RGB")
        draw = ImageDraw.Draw(canvas)
        baseline_points = _prediction_points(
            baseline[segment], route_xy, evidence_frame
        )
        method_points = _prediction_points(method[segment], route_xy, evidence_frame)
        for cross, z in baseline_points:
            if cross_range[0] <= cross <= cross_range[1] and z_range[0] <= z <= z_range[1]:
                _draw_marker(draw, _pixel(cross, z, cross_range, z_range), (224, 86, 255), "diamond")
        for cross, z in method_points:
            if cross_range[0] <= cross <= cross_range[1] and z_range[0] <= z <= z_range[1]:
                _draw_marker(draw, _pixel(cross, z, cross_range, z_range), (170, 255, 55), "circle")
        draw.rectangle((680, 10, 1090, 45), fill=(4, 18, 27))
        draw.text((692, 18), "Open3D baseline", fill=(224, 86, 255))
        draw.text((842, 18), "Rail-specific method", fill=(170, 255, 55))
        draw.text(
            (365, 36),
            "QUALITATIVE PREDICTIONS - NOT GROUND TRUTH",
            fill=(255, 151, 65),
        )
        image_path = output_root / f"{segment}-cross-section-comparison.png"
        canvas.save(image_path)
        selected_stations.append(
            {
                "segment_id": segment,
                "chainage_m": chainage,
                "neutral_source_image": station["cross_section_image"],
                "output_image": str(image_path),
                "baseline_line_count": len(baseline_points),
                "method_line_count": len(method_points),
            }
        )

    images = [Image.open(item["output_image"]).convert("RGB") for item in selected_stations]
    sheet = Image.new("RGB", (images[0].width * 2, images[0].height * 2), (4, 18, 27))
    for index, image in enumerate(images):
        sheet.paste(image, ((index % 2) * image.width, (index // 2) * image.height))
    sheet_path = output_root / "rail-method-comparison-contact-sheet.png"
    sheet.save(sheet_path)
    manifest = {
        "schema_version": "railway.rail-method-comparison-figure.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "visualization_role": "prediction_overlay_not_ground_truth",
        "evidence_manifest": str(evidence_source),
        "evidence_manifest_sha256": sha256_file(evidence_source),
        "baseline_reports": [
            {"segment_id": key, "path": str(path), "sha256": sha256_file(path)}
            for key, path in sorted(baseline_paths.items())
        ],
        "method_reports": [
            {"segment_id": key, "path": str(path), "sha256": sha256_file(path)}
            for key, path in sorted(method_paths.items())
        ],
        "stations": selected_stations,
        "contact_sheet": str(sheet_path),
        "limitations": [
            "Colored markers are predictions, not annotations or survey truth.",
            "The background image is neutral raw-point evidence.",
            "This figure is qualitative and does not replace frozen holdout metrics.",
        ],
    }
    manifest_path = output_root / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path
