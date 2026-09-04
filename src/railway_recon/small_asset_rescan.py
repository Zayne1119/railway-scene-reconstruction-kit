from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .cloud_model_gap_audit import (
    load_cloud_sample_rgb,
    reverse_model_distances,
    sample_model_surfaces,
)
from .io import load_json, sha256_file, write_json
from .model_point_support import object_vertex_indices, parse_obj_model


def classify_small_asset_component(
    spans_s_c_z_m: np.ndarray,
    *,
    bottom_above_platform_m: float,
    top_below_canopy_m: float,
) -> str:
    station, cross, height = (float(value) for value in spans_s_c_z_m)
    if station >= 2.0 and cross <= 1.1 and 0.55 <= height <= 2.2:
        return "railing_or_fence_gap_candidate"
    if (
        0.45 <= station <= 4.0
        and cross <= 1.25
        and 0.55 <= height <= 3.0
        and (bottom_above_platform_m <= 0.55 or top_below_canopy_m <= 0.75)
    ):
        return "freestanding_or_hanging_sign_candidate"
    if (
        station <= 2.5
        and cross <= 2.0
        and 0.30 <= height <= 2.5
        and bottom_above_platform_m <= 0.65
    ):
        return "equipment_box_or_cabinet_candidate"
    return "unresolved_small_surface_or_scan_residual"


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _platform_envelopes(
    model: Any,
    origin_xyz: np.ndarray,
    registry: dict[str, Any],
    frame_origin: np.ndarray,
    along: np.ndarray,
    cross: np.ndarray,
) -> list[dict[str, Any]]:
    groups: dict[str, list[np.ndarray]] = {"right": [], "opposite": []}
    for asset in registry.get("assets", []):
        if asset.get("type") != "platform_surface":
            continue
        node = str(asset.get("geometry", {}).get("node", ""))
        indexes = object_vertex_indices(model, node)
        if not len(indexes):
            continue
        world = model.vertices[indexes] + origin_xyz
        delta = world[:, :2] - frame_origin
        local = np.column_stack((delta @ along, delta @ cross, world[:, 2]))
        side = "right" if float(np.median(local[:, 1])) > -10.0 else "opposite"
        groups[side].append(local)
    envelopes: list[dict[str, Any]] = []
    for side, values in groups.items():
        if not values:
            continue
        points = np.vstack(values)
        minimum, maximum = points.min(axis=0), points.max(axis=0)
        envelopes.append(
            {
                "side": side,
                "station_range_m": [float(minimum[0] - 0.35), float(maximum[0] + 0.35)],
                "cross_range_m": [float(minimum[1] - 0.55), float(maximum[1] + 0.55)],
                "platform_z_m": float(np.median(points[:, 2])),
            }
        )
    return envelopes


def _existing_small_asset_bounds(
    model: Any,
    origin_xyz: np.ndarray,
    registry: dict[str, Any],
    frame_origin: np.ndarray,
    along: np.ndarray,
    cross: np.ndarray,
) -> list[dict[str, Any]]:
    allowed = {
        "platform_fence",
        "station_information_sign",
        "canopy_column",
        "station_entry_door",
        "station_entry_glazing_frame",
    }
    records: list[dict[str, Any]] = []
    for asset in registry.get("assets", []):
        if asset.get("type") not in allowed:
            continue
        node = str(asset.get("geometry", {}).get("node", ""))
        indexes = object_vertex_indices(model, node)
        if not len(indexes):
            continue
        world = model.vertices[indexes] + origin_xyz
        delta = world[:, :2] - frame_origin
        local = np.column_stack((delta @ along, delta @ cross, world[:, 2]))
        records.append(
            {
                "asset_id": str(asset["id"]),
                "asset_type": str(asset["type"]),
                "minimum_s_c_z_m": local.min(axis=0),
                "maximum_s_c_z_m": local.max(axis=0),
            }
        )
    return records


def _bbox_gap(
    first_minimum: np.ndarray,
    first_maximum: np.ndarray,
    second_minimum: np.ndarray,
    second_maximum: np.ndarray,
) -> float:
    delta = np.maximum(
        np.maximum(second_minimum - first_maximum, first_minimum - second_maximum), 0.0
    )
    return float(np.linalg.norm(delta))


def _scan_envelope(
    points: np.ndarray,
    colours: np.ndarray,
    distances: np.ndarray,
    *,
    envelope: dict[str, Any],
    existing: list[dict[str, Any]],
    voxel_m: float,
    unexplained_distance_m: float,
) -> dict[str, Any]:
    s_low, s_high = envelope["station_range_m"]
    c_low, c_high = envelope["cross_range_m"]
    platform_z = float(envelope["platform_z_m"])
    canopy_z = platform_z + 5.5
    selected = (
        (points[:, 0] >= s_low)
        & (points[:, 0] <= s_high)
        & (points[:, 1] >= c_low)
        & (points[:, 1] <= c_high)
        & (points[:, 2] >= platform_z + 0.12)
        & (points[:, 2] <= platform_z + 3.2)
        & (distances > unexplained_distance_m)
    )
    values = points[selected]
    selected_colours = colours[selected]
    if not len(values):
        return {**envelope, "selected_point_count": 0, "candidates": []}
    minimum = np.asarray([s_low, c_low, platform_z + 0.12])
    shape = np.ceil(
        (np.asarray([s_high, c_high, platform_z + 3.2]) - minimum) / voxel_m
    ).astype(np.int32) + 1
    voxel = np.floor((values - minimum) / voxel_m).astype(np.int32)
    voxel = np.minimum(np.maximum(voxel, 0), shape - 1)
    linear = np.ravel_multi_index(voxel.T, tuple(int(value) for value in shape))
    unique, counts = np.unique(linear, return_counts=True)
    occupied = np.zeros(int(np.prod(shape)), dtype=bool)
    occupied[unique[counts >= 2]] = True
    occupied = occupied.reshape(tuple(int(value) for value in shape))
    labels, label_count = ndimage.label(
        occupied, structure=np.ones((3, 3, 3), dtype=bool)
    )
    point_labels = labels[voxel[:, 0], voxel[:, 1], voxel[:, 2]]
    candidates: list[dict[str, Any]] = []
    for label in range(1, label_count + 1):
        member = point_labels == label
        if np.count_nonzero(member) < 24:
            continue
        component = values[member]
        component_colours = selected_colours[member]
        component_distances = distances[selected][member]
        component_minimum, component_maximum = component.min(axis=0), component.max(axis=0)
        spans = component_maximum - component_minimum
        if spans[2] < 0.25 or max(spans[0], spans[1]) > 8.0:
            continue
        bottom = float(component_minimum[2] - platform_z)
        top_below = float(canopy_z - component_maximum[2])
        classification = classify_small_asset_component(
            spans,
            bottom_above_platform_m=bottom,
            top_below_canopy_m=top_below,
        )
        matches = [
            (
                _bbox_gap(
                    component_minimum,
                    component_maximum,
                    item["minimum_s_c_z_m"],
                    item["maximum_s_c_z_m"],
                ),
                item,
            )
            for item in existing
        ]
        nearest_distance, nearest = min(matches, key=lambda item: item[0])
        relation = (
            "existing_asset_detail_or_alignment_residual"
            if nearest_distance <= 0.70
            else "new_small_asset_candidate"
        )
        median_colour = np.median(component_colours, axis=0)
        green_excess = float(
            median_colour[1] - max(float(median_colour[0]), float(median_colour[2]))
        )
        likely_vegetation = green_excess >= 12.0 and classification.startswith("unresolved")
        if likely_vegetation:
            disposition = "reject_likely_vegetation_or_landscape"
        elif relation.startswith("existing"):
            disposition = "close_as_existing_asset_refinement_residual"
        else:
            disposition = "withhold_new_geometry_pending_semantic_photo_evidence"
        candidates.append(
            {
                "candidate_id": "",
                "side": envelope["side"],
                "classification": classification,
                "sample_point_count": int(np.count_nonzero(member)),
                "minimum_s_c_z_m": component_minimum.tolist(),
                "maximum_s_c_z_m": component_maximum.tolist(),
                "span_s_c_z_m": spans.tolist(),
                "bottom_above_platform_m": bottom,
                "top_below_nominal_canopy_m": top_below,
                "distance_to_model_p50_m": float(np.percentile(component_distances, 50)),
                "distance_to_model_p90_m": float(np.percentile(component_distances, 90)),
                "median_rgb": [round(float(value)) for value in median_colour],
                "nearest_existing_asset": {
                    "asset_id": nearest["asset_id"],
                    "asset_type": nearest["asset_type"],
                    "bbox_gap_m": nearest_distance,
                },
                "asset_relation": relation,
                "disposition": disposition,
                "geometry_write": False,
            }
        )
    candidates.sort(
        key=lambda item: (
            item["disposition"] != "withhold_new_geometry_pending_semantic_photo_evidence",
            item["classification"],
            item["minimum_s_c_z_m"][0],
        )
    )
    for index, candidate in enumerate(candidates, start=1):
        candidate["candidate_id"] = f"SMALL-GAP-{envelope['side'].upper()}-{index:03d}"
    return {
        **envelope,
        "selected_point_count": int(np.count_nonzero(selected)),
        "raw_component_count": int(label_count),
        "retained_candidate_count": len(candidates),
        "candidates": candidates,
    }


def _render(report: dict[str, Any], path: Path) -> None:
    width, panel_height = 1600, 570
    canvas = Image.new("RGB", (width, 100 + panel_height * len(report["sides"])), "#06141d")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 20), "SUPPLEMENTAL CLOUD / SMALL-ASSET GAP RESCAN", fill="#eaf9ff", font=_font(24))
    draw.text((30, 54), "Only residual connected components; no candidate writes geometry without semantic evidence.", fill="#91aab3", font=_font(15))
    palette = {
        "railing_or_fence_gap_candidate": "#24d7ff",
        "freestanding_or_hanging_sign_candidate": "#b7ff3c",
        "equipment_box_or_cabinet_candidate": "#ff9f31",
        "unresolved_small_surface_or_scan_residual": "#a98cff",
    }
    for panel_index, side in enumerate(report["sides"]):
        x0, y0 = 30, 95 + panel_index * panel_height
        x1, y1 = width - 30, y0 + panel_height - 20
        draw.rectangle((x0, y0, x1, y1), outline="#28505e", width=2)
        draw.text((x0 + 14, y0 + 12), side["side"].upper(), fill="#eaf9ff", font=_font(18))
        s_low, s_high = side["station_range_m"]
        c_low, c_high = side["cross_range_m"]

        def px(
            value: float,
            panel_x0: float = x0,
            panel_x1: float = x1,
            minimum_s: float = s_low,
            maximum_s: float = s_high,
        ) -> float:
            return panel_x0 + 45 + (value - minimum_s) / max(
                maximum_s - minimum_s, 1.0e-9
            ) * (panel_x1 - panel_x0 - 80)

        def py(
            value: float,
            panel_y0: float = y0,
            panel_y1: float = y1,
            minimum_c: float = c_low,
            maximum_c: float = c_high,
        ) -> float:
            return panel_y1 - 45 - (value - minimum_c) / max(
                maximum_c - minimum_c, 1.0e-9
            ) * (panel_y1 - panel_y0 - 95)

        for candidate in side["candidates"]:
            low, high = candidate["minimum_s_c_z_m"], candidate["maximum_s_c_z_m"]
            box = (px(low[0]), py(high[1]), px(high[0]), py(low[1]))
            colour = palette[candidate["classification"]]
            draw.rectangle(box, outline=colour, width=3)
            draw.text((box[0], max(y0 + 42, box[1] - 15)), candidate["candidate_id"].split("-")[-1], fill=colour, font=_font(12))
        draw.text((x0 + 14, y1 - 27), f"residual points={side['selected_point_count']:,} retained={side['retained_candidate_count']}", fill="#91aab3", font=_font(14))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def rescan_small_asset_gaps(
    *,
    cloud_path: str | Path,
    obj_path: str | Path,
    origin_path: str | Path,
    registry_path: str | Path,
    frame_report_path: str | Path,
    output_directory: str | Path,
    target_cloud_points: int = 6_000_000,
    voxel_m: float = 0.15,
    unexplained_distance_m: float = 0.25,
) -> dict[str, Path]:
    cloud = Path(cloud_path).resolve()
    obj = Path(obj_path).resolve()
    origin_file = Path(origin_path).resolve()
    registry_file = Path(registry_path).resolve()
    frame_file = Path(frame_report_path).resolve()
    for source in (cloud, obj, origin_file, registry_file, frame_file):
        if not source.is_file():
            raise FileNotFoundError(source)
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty rescan directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    model = parse_obj_model(obj)
    origin_xyz = np.asarray(load_json(origin_file)["origin_xyz"], dtype=np.float64)
    registry = load_json(registry_file)
    frame = load_json(frame_file)["frame"]
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    model_samples, model_sampling = sample_model_surfaces(
        model, origin_xyz=origin_xyz, spacing_m=0.15
    )
    cloud_points, colours, cloud_sampling = load_cloud_sample_rgb(
        cloud, target_point_count=target_cloud_points
    )
    distances = reverse_model_distances(cloud_points, model_samples)
    delta = cloud_points[:, :2] - frame_origin
    local_points = np.column_stack((delta @ along, delta @ cross, cloud_points[:, 2]))
    envelopes = _platform_envelopes(
        model, origin_xyz, registry, frame_origin, along, cross
    )
    existing = _existing_small_asset_bounds(
        model, origin_xyz, registry, frame_origin, along, cross
    )
    sides = [
        _scan_envelope(
            local_points,
            colours,
            distances,
            envelope=envelope,
            existing=existing,
            voxel_m=voxel_m,
            unexplained_distance_m=unexplained_distance_m,
        )
        for envelope in envelopes
    ]
    candidates = [candidate for side in sides for candidate in side["candidates"]]
    unresolved = [
        item
        for item in candidates
        if item["disposition"] == "withhold_new_geometry_pending_semantic_photo_evidence"
    ]
    report_path = output / "small_asset_gap_rescan.json"
    figure_path = output / "small_asset_gap_rescan.png"
    report = {
        "schema_version": "railway.small-asset-gap-rescan.v1",
        "inputs": {
            "cloud": str(cloud),
            "cloud_sha256": sha256_file(cloud),
            "obj": str(obj),
            "obj_sha256": sha256_file(obj),
            "registry": str(registry_file),
        },
        "method": {
            "model_surface_sampling": model_sampling,
            "cloud_sampling": cloud_sampling,
            "unexplained_distance_m": unexplained_distance_m,
            "voxel_m": voxel_m,
            "vertical_search": "platform +0.12 m to +3.20 m",
        },
        "sides": sides,
        "summary": {
            "retained_candidate_count": len(candidates),
            "new_semantically_unresolved_candidate_count": len(unresolved),
            "closed_existing_asset_residual_count": sum(
                item["disposition"] == "close_as_existing_asset_refinement_residual"
                for item in candidates
            ),
            "rejected_vegetation_count": sum(
                item["disposition"] == "reject_likely_vegetation_or_landscape"
                for item in candidates
            ),
            "by_class": {
                value: sum(item["classification"] == value for item in candidates)
                for value in sorted({item["classification"] for item in candidates})
            },
        },
        "geometry_write": False,
        "registry_write": False,
        "status": (
            "rescan_complete_new_candidates_withheld_pending_semantics"
            if unresolved
            else "rescan_complete_no_unresolved_small_asset_gap"
        ),
        "limitations": [
            "The scan is restricted to modelled platform envelopes and does not classify off-platform vegetation.",
            "Geometry-only dimensions cannot distinguish every cabinet, sign and facade fragment.",
            "Unresolved candidates remain evidence records and are not added to the final mesh or asset table.",
        ],
    }
    write_json(report_path, report)
    _render(report, figure_path)
    return {"report": report_path, "figure": figure_path}
