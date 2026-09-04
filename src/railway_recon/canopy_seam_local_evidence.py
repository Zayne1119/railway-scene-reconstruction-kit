from __future__ import annotations

from pathlib import Path
from typing import Any

import laspy
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .io import load_json, write_json


def _predict_top(section: dict[str, Any], cross_m: np.ndarray) -> np.ndarray:
    low = float(section["cross_min_m"])
    high = float(section["cross_max_m"])
    fraction = (cross_m - low) / (high - low)
    return float(section["top_z_at_min_cross_m"]) + fraction * (
        float(section["top_z_at_max_cross_m"])
        - float(section["top_z_at_min_cross_m"])
    )


def estimate_surface_support(
    points: np.ndarray,
    section: dict[str, Any],
    *,
    station_min_m: float,
    station_max_m: float,
    lateral_expansion_m: float = 0.55,
    vertical_tolerance_m: float = 0.07,
    minimum_point_count: int = 100,
) -> dict[str, Any]:
    """Estimate an observed roof strip extent near one model top plane."""
    station = points[:, 0]
    lateral = points[:, 1]
    cross_min = float(section["cross_min_m"])
    cross_max = float(section["cross_max_m"])
    within = (
        (station >= station_min_m)
        & (station <= station_max_m)
        & (lateral >= cross_min - lateral_expansion_m)
        & (lateral <= cross_max + lateral_expansion_m)
    )
    candidate = points[within]
    if not len(candidate):
        return {"status": "data_insufficient", "point_count": 0}
    residual = candidate[:, 2] - _predict_top(section, candidate[:, 1])
    supported = candidate[np.abs(residual) <= vertical_tolerance_m]
    if len(supported) < minimum_point_count:
        return {
            "status": "data_insufficient",
            "point_count": len(supported),
            "minimum_point_count": minimum_point_count,
        }
    observed_min, observed_max = np.percentile(supported[:, 1], [1.0, 99.0])
    return {
        "status": "supported",
        "point_count": len(supported),
        "observed_cross_min_m": float(observed_min),
        "observed_cross_max_m": float(observed_max),
        "model_cross_min_m": cross_min,
        "model_cross_max_m": cross_max,
        "minimum_edge_residual_m": float(observed_min - cross_min),
        "maximum_edge_residual_m": float(observed_max - cross_max),
        "vertical_residual_abs_p90_m": float(
            np.percentile(
                np.abs(
                    supported[:, 2] - _predict_top(section, supported[:, 1])
                ),
                90,
            )
        ),
    }


def classify_edge_change(
    *,
    model_change_m: float,
    observed_change_m: float,
    stable_observed_limit_m: float = 0.10,
    meaningful_model_change_m: float = 0.20,
    matching_change_tolerance_m: float = 0.15,
) -> str:
    if (
        abs(model_change_m) >= meaningful_model_change_m
        and abs(observed_change_m) <= stable_observed_limit_m
    ):
        return "model_segmentation_mismatch"
    if (
        abs(model_change_m) >= meaningful_model_change_m
        and abs(observed_change_m - model_change_m) <= matching_change_tolerance_m
    ):
        return "point_supported_physical_transition"
    if (
        abs(model_change_m) < meaningful_model_change_m
        and abs(observed_change_m) <= stable_observed_limit_m
    ):
        return "stable_boundary"
    return "ambiguous_requires_local_geometry_review"


def _near_roof_mask(
    points: np.ndarray,
    left_sections: list[dict[str, Any]],
    right_sections: list[dict[str, Any]],
) -> np.ndarray:
    minimum_residual = np.full(len(points), np.inf, dtype=np.float64)
    for side_mask, sections in (
        (points[:, 0] <= 0.0, left_sections),
        (points[:, 0] > 0.0, right_sections),
    ):
        for section in sections:
            low = float(section["cross_min_m"]) - 0.35
            high = float(section["cross_max_m"]) + 0.35
            valid = side_mask & (points[:, 1] >= low) & (points[:, 1] <= high)
            if not np.any(valid):
                continue
            residual = np.abs(
                points[valid, 2] - _predict_top(section, points[valid, 1])
            )
            minimum_residual[valid] = np.minimum(minimum_residual[valid], residual)
    return minimum_residual <= 0.08


def _plot_seam(
    points: np.ndarray,
    left_sections: list[dict[str, Any]],
    right_sections: list[dict[str, Any]],
    output: Path,
    title: str,
) -> None:
    width, height = 1600, 1000
    image = Image.new("RGB", (width, height), (5, 18, 26))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=22)
    small = ImageFont.load_default(size=17)
    draw.rectangle((20, 20, width - 20, 85), fill=(3, 14, 21), outline=(72, 180, 199))
    draw.text((42, 38), title, fill=(221, 240, 244), font=font)

    panels = {
        "plan": (60, 125, 760, 940),
        "cross": (840, 125, 1540, 940),
    }
    for name, box in panels.items():
        draw.rectangle(box, outline=(45, 100, 116), width=2)
        draw.text((box[0] + 15, box[1] + 12), name.upper(), fill=(174, 255, 81), font=small)

    near = points[_near_roof_mask(points, left_sections, right_sections)]
    if len(near) > 60_000:
        indexes = np.linspace(0, len(near) - 1, 60_000, dtype=np.int64)
        near = near[indexes]
    plan = panels["plan"]
    cross_box = panels["cross"]

    def plan_xy(station: float, lateral: float) -> tuple[int, int]:
        x = plan[0] + 35 + int((station + 1.25) / 2.5 * (plan[2] - plan[0] - 70))
        y = plan[3] - 35 - int((lateral - 0.3) / 9.8 * (plan[3] - plan[1] - 70))
        return x, y

    def section_xy(lateral: float, z: float) -> tuple[int, int]:
        x = cross_box[0] + 35 + int((lateral - 0.3) / 9.8 * (cross_box[2] - cross_box[0] - 70))
        y = cross_box[3] - 35 - int((z - 24.7) / 2.5 * (cross_box[3] - cross_box[1] - 70))
        return x, y

    for station, lateral, z, red, green, blue in near:
        colour = (int(red), int(green), int(blue))
        x, y = plan_xy(float(station), float(lateral))
        if plan[0] < x < plan[2] and plan[1] < y < plan[3]:
            draw.point((x, y), fill=colour)
        x, y = section_xy(float(lateral), float(z))
        if cross_box[0] < x < cross_box[2] and cross_box[1] < y < cross_box[3]:
            draw.point((x, y), fill=colour)

    seam_x, _ = plan_xy(0.0, 0.3)
    draw.line((seam_x, plan[1] + 32, seam_x, plan[3] - 25), fill=(255, 150, 35), width=3)
    for side, sections, station_span, colour in (
        ("L", left_sections, (-1.2, 0.0), (84, 221, 238)),
        ("R", right_sections, (0.0, 1.2), (255, 184, 67)),
    ):
        for section in sections:
            for key in ("cross_min_m", "cross_max_m"):
                lateral = float(section[key])
                start = plan_xy(station_span[0], lateral)
                end = plan_xy(station_span[1], lateral)
                draw.line((start, end), fill=colour, width=3)
            samples = np.linspace(
                float(section["cross_min_m"]),
                float(section["cross_max_m"]),
                100,
            )
            line = [
                section_xy(float(value), float(_predict_top(section, np.asarray([value]))[0]))
                for value in samples
            ]
            draw.line(line, fill=colour, width=3)
        draw.text(
            (cross_box[0] + (25 if side == "L" else 125), cross_box[1] + 42),
            f"{side} model top",
            fill=colour,
            font=small,
        )
    draw.text((plan[0] + 20, plan[3] - 25), "station offset -1.25 ... +1.25 m", fill=(143, 194, 204), font=small)
    draw.text((cross_box[0] + 20, cross_box[3] - 25), "cross / z roof-near points", fill=(143, 194, 204), font=small)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def audit_canopy_seam_local_evidence(
    *,
    cloud_path: str | Path,
    frame_report_path: str | Path,
    union_seam_path: str | Path,
    physical_seam_path: str | Path,
    output_directory: str | Path,
    station_half_width_m: float = 1.25,
    cross_min_m: float = 0.3,
    cross_max_m: float = 10.1,
    z_min_m: float = 24.7,
    z_max_m: float = 27.2,
) -> dict[str, Path]:
    frame = load_json(Path(frame_report_path))["frame"]
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    union_audit = load_json(Path(union_seam_path))
    physical_audit = load_json(Path(physical_seam_path))
    seam_stations = [float(item["station_m"]) for item in union_audit["seams"]]
    windows: list[list[np.ndarray]] = [[] for _ in seam_stations]
    source = Path(cloud_path).resolve()
    with laspy.open(source) as reader:
        source_count = int(reader.header.point_count)
        for chunk in reader.chunk_iterator(2_000_000):
            xyz = np.column_stack((np.asarray(chunk.x), np.asarray(chunk.y), np.asarray(chunk.z)))
            delta = xyz[:, :2] - frame_origin
            station = delta @ along
            lateral = delta @ cross
            red = np.asarray(chunk.red, dtype=np.float64)
            green = np.asarray(chunk.green, dtype=np.float64)
            blue = np.asarray(chunk.blue, dtype=np.float64)
            colour_scale = 257.0 if max(red.max(), green.max(), blue.max()) > 255 else 1.0
            for index, seam_station in enumerate(seam_stations):
                offset = station - seam_station
                mask = (
                    (np.abs(offset) <= station_half_width_m)
                    & (lateral >= cross_min_m)
                    & (lateral <= cross_max_m)
                    & (xyz[:, 2] >= z_min_m)
                    & (xyz[:, 2] <= z_max_m)
                )
                if np.any(mask):
                    windows[index].append(
                        np.column_stack(
                            (
                                offset[mask],
                                lateral[mask],
                                xyz[mask, 2],
                                red[mask] / colour_scale,
                                green[mask] / colour_scale,
                                blue[mask] / colour_scale,
                            )
                        )
                    )
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    seam_reports: list[dict[str, Any]] = []
    image_paths: list[Path] = []
    for index, (chunks, physical) in enumerate(
        zip(windows, physical_audit["seams"], strict=True), start=1
    ):
        points = np.vstack(chunks) if chunks else np.empty((0, 6), dtype=np.float64)
        left_sections = physical["left_sections"]
        right_sections = physical["right_sections"]
        surfaces: list[dict[str, Any]] = []
        left_by_token = {
            item["object_name"].split("-SURFACE-", 1)[1].split("-", 1)[0]: item
            for item in left_sections
        }
        right_by_token = {
            item["object_name"].split("-SURFACE-", 1)[1].split("-", 1)[0]: item
            for item in right_sections
        }
        for token in sorted(left_by_token):
            left = estimate_surface_support(
                points,
                left_by_token[token],
                station_min_m=-1.1,
                station_max_m=-0.15,
            )
            right = estimate_surface_support(
                points,
                right_by_token[token],
                station_min_m=0.15,
                station_max_m=1.1,
            )
            edge_classification: dict[str, str] = {}
            if left["status"] == "supported" and right["status"] == "supported":
                for edge in ("min", "max"):
                    model_change = float(right_by_token[token][f"cross_{edge}_m"]) - float(
                        left_by_token[token][f"cross_{edge}_m"]
                    )
                    observed_change = float(right[f"observed_cross_{edge}_m"]) - float(
                        left[f"observed_cross_{edge}_m"]
                    )
                    edge_classification[edge] = classify_edge_change(
                        model_change_m=model_change,
                        observed_change_m=observed_change,
                    )
            surfaces.append(
                {
                    "surface_token": token,
                    "left": left,
                    "right": right,
                    "edge_classification": edge_classification,
                }
            )
        classifications = [
            value
            for surface in surfaces
            for value in surface["edge_classification"].values()
        ]
        if not classifications:
            conclusion = "data_insufficient"
        elif "model_segmentation_mismatch" in classifications:
            conclusion = "model_segmentation_mismatch_present"
        elif "point_supported_physical_transition" in classifications:
            conclusion = "point_supported_physical_transition_present"
        elif all(value == "stable_boundary" for value in classifications):
            conclusion = "stable_physical_boundary"
        else:
            conclusion = "mixed_or_ambiguous"
        image_path = output / f"seam_{index:02d}_local_evidence.png"
        _plot_seam(
            points,
            left_sections,
            right_sections,
            image_path,
            f"Seam {index:02d} / {physical['left_segment']} - {physical['right_segment']}",
        )
        image_paths.append(image_path)
        seam_reports.append(
            {
                "seam_index": index,
                "left_segment": physical["left_segment"],
                "right_segment": physical["right_segment"],
                "station_m": seam_stations[index - 1],
                "local_point_count": len(points),
                "surfaces": surfaces,
                "conclusion": conclusion,
                "image": str(image_path),
            }
        )
    report_path = output / "canopy_seam_local_evidence.json"
    write_json(
        report_path,
        {
            "schema_version": "railway.canopy-seam-local-evidence.v1",
            "cloud": str(source),
            "source_point_count": source_count,
            "window": {
                "station_half_width_m": station_half_width_m,
                "cross_min_m": cross_min_m,
                "cross_max_m": cross_max_m,
                "z_min_m": z_min_m,
                "z_max_m": z_max_m,
            },
            "seams": seam_reports,
            "policy": (
                "This report diagnoses local evidence only; geometry generation requires "
                "a separate candidate and all downstream gates."
            ),
        },
    )
    return {"report": report_path, **{f"image_{i+1}": p for i, p in enumerate(image_paths)}}
