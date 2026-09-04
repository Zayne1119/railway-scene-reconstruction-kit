from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .io import load_json, write_json
from .model_point_support import object_vertex_indices, parse_obj_model


def _z_overlap_ratio(first: dict[str, Any], second: dict[str, Any]) -> float:
    low = max(float(first["minimum_z"]), float(second["minimum_z"]))
    high = min(float(first["maximum_z"]), float(second["maximum_z"]))
    overlap = max(0.0, high - low)
    first_span = max(1e-9, float(first["maximum_z"]) - float(first["minimum_z"]))
    second_span = max(1e-9, float(second["maximum_z"]) - float(second["minimum_z"]))
    return overlap / min(first_span, second_span)


def deduplicate_vertical_candidates(
    candidates: list[dict[str, Any]],
    *,
    horizontal_tolerance_m: float = 0.45,
    minimum_z_overlap_ratio: float = 0.6,
) -> list[dict[str, Any]]:
    groups: list[list[dict[str, Any]]] = []
    for candidate in sorted(candidates, key=lambda item: int(item["point_count"]), reverse=True):
        group = next(
            (
                items
                for items in groups
                if math.hypot(
                    float(candidate["center_x"]) - float(items[0]["center_x"]),
                    float(candidate["center_y"]) - float(items[0]["center_y"]),
                )
                <= horizontal_tolerance_m
                and _z_overlap_ratio(candidate, items[0]) >= minimum_z_overlap_ratio
            ),
            None,
        )
        if group is None:
            groups.append([candidate])
        else:
            group.append(candidate)

    records: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        weights = np.asarray([int(item["point_count"]) for item in group], dtype=np.float64)
        records.append(
            {
                "id": f"SUPPLEMENTAL-VERTICAL-{index:03d}",
                "center_x": float(
                    np.average([float(item["center_x"]) for item in group], weights=weights)
                ),
                "center_y": float(
                    np.average([float(item["center_y"]) for item in group], weights=weights)
                ),
                "minimum_z": min(float(item["minimum_z"]) for item in group),
                "maximum_z": max(float(item["maximum_z"]) for item in group),
                "footprint_m": max(float(item.get("footprint_m", 0.1)) for item in group),
                "point_count_sum": int(sum(int(item["point_count"]) for item in group)),
                "source_occurrence_count": len(group),
                "source_candidates": [
                    {"segment_id": item["segment_id"], "id": item["id"]} for item in group
                ],
            }
        )
    return records


def _point_to_box_xy(point: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> float:
    delta = np.maximum(np.maximum(minimum[:2] - point[:2], point[:2] - maximum[:2]), 0.0)
    return float(np.linalg.norm(delta))


def compare_vertical_candidates_to_assets(
    candidates: list[dict[str, Any]],
    assets: list[dict[str, Any]],
    *,
    match_tolerance_m: float = 0.55,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for candidate in candidates:
        point = np.asarray(
            [candidate["center_x"], candidate["center_y"], candidate["minimum_z"]],
            dtype=np.float64,
        )
        ranked = sorted(
            (
                (
                    _point_to_box_xy(point, asset["minimum_xyz"], asset["maximum_xyz"]),
                    asset,
                )
                for asset in assets
            ),
            key=lambda item: item[0],
        )
        distance, asset = ranked[0]
        candidate_z = {
            "minimum_z": candidate["minimum_z"],
            "maximum_z": candidate["maximum_z"],
        }
        asset_z = {
            "minimum_z": float(asset["minimum_xyz"][2]),
            "maximum_z": float(asset["maximum_xyz"][2]),
        }
        overlap = _z_overlap_ratio(candidate_z, asset_z)
        explained = distance <= match_tolerance_m and overlap >= 0.35
        record = dict(candidate)
        record.update(
            {
                "nearest_asset_id": asset["id"],
                "nearest_asset_type": asset["type"],
                "nearest_asset_horizontal_box_distance_m": distance,
                "nearest_asset_z_overlap_ratio": overlap,
                "decision": (
                    "explained_by_existing_vertical_asset"
                    if explained
                    else "unmatched_requires_photo_or_point_review"
                ),
            }
        )
        result.append(record)
    return result


def _render_plan(records: list[dict[str, Any]], assets: list[dict[str, Any]], path: Path) -> None:
    width, height = 1800, 1000
    margin = 90
    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)
    coordinates = [
        ((asset["minimum_xyz"] + asset["maximum_xyz"]) / 2.0)[:2] for asset in assets
    ] + [np.asarray([record["center_x"], record["center_y"]]) for record in records]
    values = np.vstack(coordinates)
    minimum = np.min(values, axis=0)
    maximum = np.max(values, axis=0)
    span = np.maximum(maximum - minimum, 1e-6)

    def project(point: np.ndarray) -> tuple[float, float]:
        scale = min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])
        x = margin + (point[0] - minimum[0]) * scale
        y = height - margin - (point[1] - minimum[1]) * scale
        return float(x), float(y)

    for asset in assets:
        center = (asset["minimum_xyz"] + asset["maximum_xyz"]) / 2.0
        x, y = project(center[:2])
        draw.rectangle((x - 5, y - 5, x + 5, y + 5), fill="#7f8f98")
    for record in records:
        explained = record["decision"] == "explained_by_existing_vertical_asset"
        colour = "#66d9ef" if explained else "#ff9f31"
        x, y = project(np.asarray([record["center_x"], record["center_y"]]))
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=colour, outline="#14242d")
        if not explained:
            draw.text((x + 9, y - 9), record["id"].rsplit("-", 1)[-1], fill=colour)
    draw.text((30, 25), "SUPPLEMENTAL VERTICAL CANDIDATES / EXISTING ASSETS", fill="#14242d")
    draw.text((30, 48), "grey=model | cyan=explained | orange=unmatched review", fill="#52646e")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def build_supplemental_vertical_comparison(
    report_paths: list[str | Path],
    obj_path: str | Path,
    origin_path: str | Path,
    registry_path: str | Path,
    output_path: str | Path,
    diagnostic_path: str | Path,
) -> Path:
    raw_candidates: list[dict[str, Any]] = []
    source_counts: list[dict[str, Any]] = []
    for report_path in report_paths:
        report = load_json(Path(report_path))
        source_counts.append(
            {
                "segment_id": report["segment_id"],
                "vertical_candidate_count": report["vertical_candidate_count"],
                "cable_candidate_count": report["cable_candidate_count"],
            }
        )
        raw_candidates.extend(
            {**candidate, "segment_id": report["segment_id"]}
            for candidate in report["vertical_candidates"]
        )

    model = parse_obj_model(obj_path)
    origin = np.asarray(load_json(Path(origin_path))["origin_xyz"], dtype=np.float64)
    registry = load_json(Path(registry_path))
    structural_types = {"canopy_column", "catenary_mast"}
    assets: list[dict[str, Any]] = []
    for asset in registry.get("assets", []):
        name = str(asset.get("id", ""))
        if asset.get("type") not in structural_types or name not in model.faces_by_object:
            continue
        vertices = model.vertices[object_vertex_indices(model, name)] + origin
        assets.append(
            {
                "id": name,
                "type": str(asset["type"]),
                "minimum_xyz": np.min(vertices, axis=0),
                "maximum_xyz": np.max(vertices, axis=0),
            }
        )

    deduplicated = deduplicate_vertical_candidates(raw_candidates)
    compared = compare_vertical_candidates_to_assets(deduplicated, assets)
    output = Path(output_path).resolve()
    diagnostic = Path(diagnostic_path).resolve()
    payload = {
        "schema_version": "railway.supplemental-vertical-comparison.v1",
        "inputs": {
            "linear_reports": [str(Path(path).resolve()) for path in report_paths],
            "obj": str(Path(obj_path).resolve()),
            "origin": str(Path(origin_path).resolve()),
            "registry": str(Path(registry_path).resolve()),
        },
        "source_counts": source_counts,
        "raw_candidate_count": len(raw_candidates),
        "deduplicated_candidate_count": len(deduplicated),
        "existing_vertical_asset_count": len(assets),
        "explained_candidate_count": sum(
            item["decision"] == "explained_by_existing_vertical_asset" for item in compared
        ),
        "unmatched_candidate_count": sum(
            item["decision"] == "unmatched_requires_photo_or_point_review" for item in compared
        ),
        "records": compared,
        "diagnostic": str(diagnostic),
        "policy": (
            "Unmatched geometry is a review candidate only. It cannot create an asset until "
            "point continuity and photo semantics pass."
        ),
    }
    write_json(output, payload)
    _render_plan(compared, assets, diagnostic)
    return output


def build_photo_review_vertical_report(
    comparison_path: str | Path,
    frame_report_path: str | Path,
    output_path: str | Path,
) -> Path:
    comparison = load_json(Path(comparison_path))
    frame_report = load_json(Path(frame_report_path))
    frame = frame_report["frame"]
    origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    candidates: list[dict[str, Any]] = []
    for record in comparison["records"]:
        if record["decision"] != "unmatched_requires_photo_or_point_review":
            continue
        delta = np.asarray([record["center_x"], record["center_y"]]) - origin
        candidates.append(
            {
                "id": record["id"],
                "longitudinal_position_m": float(delta @ along),
                "cross_position_m": float(delta @ cross),
                "minimum_z": float(record["minimum_z"]),
                "maximum_z": float(record["maximum_z"]),
                "footprint_m": float(record.get("footprint_m", 0.1)),
                "predicted_class": "supplemental_unmatched_vertical",
                "confidence": "review_required",
            }
        )
    destination = Path(output_path).resolve()
    write_json(
        destination,
        {
            "schema_version": "railway.vertical-hypotheses.photo-review-adapter.v1",
            "frame": frame,
            "candidates": candidates,
            "source_comparison": str(Path(comparison_path).resolve()),
            "geometry_write": False,
            "status": "photo_review_adapter_only",
        },
    )
    return destination
