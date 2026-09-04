from __future__ import annotations

from pathlib import Path
from typing import Any

import laspy
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .io import load_json, sha256_file, write_json
from .model_point_support import object_vertex_indices, parse_obj_model


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _fit_linear_traces(points: np.ndarray, *, seed: int = 20260902) -> list[dict[str, Any]]:
    """Extract dominant line traces from a transverse/elevation point slice."""
    if len(points) < 20:
        return []
    # Collapse dense scanned surfaces to a deterministic 5 cm cross-section grid.
    grid = np.round(points[:, 1:3] / 0.05).astype(np.int32)
    _, unique_index = np.unique(grid, axis=0, return_index=True)
    remaining = points[np.sort(unique_index), 1:3]
    rng = np.random.default_rng(seed)
    traces: list[dict[str, Any]] = []
    for trace_index in range(8):
        if len(remaining) < 12:
            break
        best: tuple[float, np.ndarray, float, float] | None = None
        pair_count = min(8_000, max(500, len(remaining) * 18))
        first = rng.integers(0, len(remaining), pair_count)
        second = rng.integers(0, len(remaining), pair_count)
        for a, b in zip(first, second, strict=True):
            if a == b:
                continue
            du = float(remaining[b, 0] - remaining[a, 0])
            if abs(du) < 0.55:
                continue
            slope = float((remaining[b, 1] - remaining[a, 1]) / du)
            if abs(slope) > 1.8:
                continue
            intercept = float(remaining[a, 1] - slope * remaining[a, 0])
            residual = np.abs(remaining[:, 1] - (slope * remaining[:, 0] + intercept))
            inlier = residual <= 0.065
            if np.count_nonzero(inlier) < 10:
                continue
            values = remaining[inlier]
            u05, u95 = np.percentile(values[:, 0], (5.0, 95.0))
            span = float(u95 - u05)
            if span < 0.55:
                continue
            occupied = len(np.unique(np.floor(values[:, 0] / 0.12).astype(np.int32)))
            expected = max(1, int(np.ceil(span / 0.12)) + 1)
            coverage = float(occupied / expected)
            score = span * coverage * float(np.sqrt(len(values)))
            if best is None or score > best[0]:
                best = score, inlier, slope, intercept
        if best is None:
            break
        _, inlier, _, _ = best
        values = remaining[inlier]
        slope, intercept = np.polyfit(values[:, 0], values[:, 1], 1)
        residual = values[:, 1] - (slope * values[:, 0] + intercept)
        u05, u95 = np.percentile(values[:, 0], (5.0, 95.0))
        span = float(u95 - u05)
        occupied = len(np.unique(np.floor(values[:, 0] / 0.12).astype(np.int32)))
        expected = max(1, int(np.ceil(span / 0.12)) + 1)
        traces.append(
            {
                "trace_id": f"TRACE-{trace_index + 1:02d}",
                "point_count": len(values),
                "u_range_m": [float(u05), float(u95)],
                "transverse_span_m": span,
                "slope_dz_du": float(slope),
                "intercept_z_m": float(intercept),
                "rmse_m": float(np.sqrt(np.mean(residual * residual))),
                "occupied_transverse_bin_ratio": float(occupied / expected),
                "endpoints_u_z_m": [
                    [float(u05), float(slope * u05 + intercept)],
                    [float(u95), float(slope * u95 + intercept)],
                ],
            }
        )
        # Remove the whole scanned thickness of the accepted trace.
        all_residual = np.abs(remaining[:, 1] - (slope * remaining[:, 0] + intercept))
        remaining = remaining[all_residual > 0.13]
    return traces


def _attachment_clusters(points: np.ndarray, main: dict[str, Any]) -> list[dict[str, Any]]:
    if not len(points):
        return []
    slope = float(main["slope_dz_du"])
    intercept = float(main["intercept_z_m"])
    residual = points[:, 2] - (slope * points[:, 1] + intercept)
    selected = points[
        (points[:, 1] >= max(0.20, float(main["u_range_m"][0]) - 0.20))
        & (points[:, 1] <= min(4.5, float(main["u_range_m"][1]) + 0.20))
        & (np.abs(residual) <= 0.55)
    ]
    if not len(selected):
        return []
    bins = np.floor(selected[:, 1] / 0.16).astype(np.int32)
    candidates: list[dict[str, Any]] = []
    for value in np.unique(bins):
        group = selected[bins == value]
        if len(group) < 12:
            continue
        z05, z95 = np.percentile(group[:, 2], (5.0, 95.0))
        z_span = float(z95 - z05)
        if z_span < 0.18:
            continue
        candidates.append(
            {
                "u_m": float(np.median(group[:, 1])),
                "z_m": float(np.median(group[:, 2])),
                "z_span_m": z_span,
                "point_count": len(group),
            }
        )
    candidates.sort(key=lambda item: item["u_m"])
    clusters: list[dict[str, Any]] = []
    for item in candidates:
        if clusters and item["u_m"] - clusters[-1]["u_m"] < 0.35:
            if item["z_span_m"] > clusters[-1]["z_span_m"]:
                clusters[-1] = item
        else:
            clusters.append(item)
    return clusters


def evaluate_top_equipment_evidence(
    points_ds_u_z: np.ndarray,
    *,
    shaft_top_z_m: float,
    station_half_window_m: float = 2.25,
    hardware_station_half_width_m: float = 0.55,
    maximum_inward_reach_m: float = 4.6,
    voxel_m: float = 0.15,
) -> dict[str, Any]:
    """Evaluate local point evidence without assuming a catalogue assembly.

    ``u`` is positive from the mast towards the track centre.  Points in a
    station-wide trace are treated as likely longitudinal conductors, while
    points localised at the mast station can support transverse hardware.
    """
    points = np.asarray(points_ds_u_z, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_ds_u_z must be an N x 3 array")
    if voxel_m <= 0.0:
        raise ValueError("voxel_m must be positive")
    if not len(points):
        return {
            "point_count": 0,
            "hardware_point_count": 0,
            "conductor_like_point_count": 0,
            "component_evidence": {},
            "gates": {"local_points_present": False},
            "passed": False,
            "disposition": "withhold_no_local_top_points",
        }

    minimum_z = shaft_top_z_m - 4.6
    search = points[
        (np.abs(points[:, 0]) <= station_half_window_m)
        & (points[:, 1] >= -0.25)
        & (points[:, 1] <= maximum_inward_reach_m)
        & (points[:, 2] >= minimum_z)
        & (points[:, 2] <= shaft_top_z_m + 0.30)
    ]
    if not len(search):
        return {
            "point_count": len(points),
            "hardware_point_count": 0,
            "conductor_like_point_count": 0,
            "component_evidence": {},
            "gates": {"local_points_present": False},
            "passed": False,
            "disposition": "withhold_no_local_top_points",
        }

    u_index = np.floor((search[:, 1] + 0.25) / voxel_m).astype(np.int32)
    z_index = np.floor((search[:, 2] - minimum_z) / voxel_m).astype(np.int32)
    width = int(np.ceil((maximum_inward_reach_m + 0.25) / voxel_m)) + 1
    key = z_index.astype(np.int64) * width + u_index
    order = np.argsort(key)
    sorted_key = key[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_key)) + 1]
    ends = np.r_[starts[1:], len(order)]
    conductor_like = np.zeros(len(search), dtype=bool)
    for start, end in zip(starts, ends, strict=True):
        indexes = order[start:end]
        station_span = float(np.ptp(search[indexes, 0])) if len(indexes) > 1 else 0.0
        # Longitudinal conductors persist through most of the review window.
        if station_span >= 0.70 * (2.0 * station_half_window_m):
            conductor_like[indexes] = True

    hardware = search[
        (~conductor_like)
        & (np.abs(search[:, 0]) <= hardware_station_half_width_m)
        & (search[:, 1] >= 0.12)
    ]

    def component(name: str, z_low: float, z_high: float, minimum_span: float) -> dict[str, Any]:
        selected = hardware[(hardware[:, 2] >= z_low) & (hardware[:, 2] <= z_high)]
        if not len(selected):
            return {
                "name": name,
                "point_count": 0,
                "transverse_span_m": 0.0,
                "occupied_transverse_bin_ratio": 0.0,
                "anchor_point_count": 0,
                "line_fit_rmse_m": None,
                "passed": False,
            }
        u05, u95 = np.percentile(selected[:, 1], (5.0, 95.0))
        span = float(u95 - u05)
        bins = np.unique(np.floor((selected[:, 1] - u05) / voxel_m).astype(np.int32))
        expected = max(1, int(np.ceil(span / voxel_m)) + 1)
        ratio = float(len(bins) / expected)
        anchor_count = int(np.count_nonzero(selected[:, 1] <= 0.75))
        if len(selected) >= 2 and span > 1.0e-6:
            slope, intercept = np.polyfit(selected[:, 1], selected[:, 2], 1)
            residual = selected[:, 2] - (slope * selected[:, 1] + intercept)
            rmse = float(np.sqrt(np.mean(residual * residual)))
            endpoints = [
                [float(u05), float(slope * u05 + intercept)],
                [float(u95), float(slope * u95 + intercept)],
            ]
        else:
            slope = 0.0
            rmse = None
            endpoints = None
        passed = (
            len(selected) >= 14
            and span >= minimum_span
            and ratio >= 0.30
            and anchor_count >= 2
            and rmse is not None
            and rmse <= 0.30
        )
        return {
            "name": name,
            "point_count": len(selected),
            "transverse_p05_p95_m": [float(u05), float(u95)],
            "transverse_span_m": span,
            "occupied_transverse_bin_ratio": ratio,
            "anchor_point_count": anchor_count,
            "line_slope_dz_du": float(slope),
            "line_fit_rmse_m": rmse,
            "fitted_u_z_endpoints_m": endpoints,
            "passed": bool(passed),
        }

    components = {
        "upper_cantilever": component(
            "upper_cantilever", shaft_top_z_m - 2.25, shaft_top_z_m - 1.20, 1.10
        ),
        "lower_cantilever": component(
            "lower_cantilever", shaft_top_z_m - 3.45, shaft_top_z_m - 2.30, 1.10
        ),
        "positioner": component(
            "positioner", shaft_top_z_m - 4.25, shaft_top_z_m - 3.15, 1.25
        ),
    }
    traces = _fit_linear_traces(hardware)
    main_candidates = [
        item
        for item in traces
        if abs(float(item["slope_dz_du"])) <= 0.18
        and float(item["transverse_span_m"]) >= 1.50
        and float(item["u_range_m"][0]) <= 0.90
        and float(item["occupied_transverse_bin_ratio"]) >= 0.45
    ]
    main = max(main_candidates, key=lambda item: item["transverse_span_m"], default=None)
    brace_candidates: list[dict[str, Any]] = []
    secondary_structure: dict[str, Any] = {
        "point_count": 0,
        "transverse_span_m": 0.0,
        "occupied_transverse_bin_ratio": 0.0,
        "vertex_u_z_m": None,
        "passed": False,
    }
    if main is not None:
        for item in traces:
            if item is main or abs(float(item["slope_dz_du"])) < 0.16:
                continue
            endpoints = np.asarray(item["endpoints_u_z_m"], dtype=np.float64)
            main_delta = np.abs(
                endpoints[:, 1]
                - (
                    float(main["slope_dz_du"]) * endpoints[:, 0]
                    + float(main["intercept_z_m"])
                )
            )
            if (
                float(item["transverse_span_m"]) >= 0.70
                and float(np.min(main_delta)) <= 0.30
            ):
                brace_candidates.append(item)
        main_slope = float(main["slope_dz_du"])
        main_intercept = float(main["intercept_z_m"])
        main_z = main_slope * hardware[:, 1] + main_intercept
        below = hardware[
            (hardware[:, 1] >= max(0.20, float(main["u_range_m"][0]) - 0.20))
            & (hardware[:, 1] <= min(4.5, float(main["u_range_m"][1]) + 0.35))
            & ((main_z - hardware[:, 2]) >= 0.15)
            & ((main_z - hardware[:, 2]) <= 1.80)
        ]
        if len(below):
            u05, u95 = np.percentile(below[:, 1], (5.0, 95.0))
            span = float(u95 - u05)
            occupied = len(np.unique(np.floor(below[:, 1] / 0.12).astype(np.int32)))
            expected = max(1, int(np.ceil(span / 0.12)) + 1)
            coverage = float(occupied / expected)
            lowest = below[below[:, 2] <= np.percentile(below[:, 2], 8.0)]
            secondary_structure = {
                "point_count": len(below),
                "transverse_span_m": span,
                "occupied_transverse_bin_ratio": coverage,
                "vertex_u_z_m": [
                    float(np.median(lowest[:, 1])),
                    float(np.median(lowest[:, 2])),
                ],
                "passed": bool(len(below) >= 30 and span >= 1.0 and coverage >= 0.40),
            }
    cantilever_passed = main is not None

    # An insulator is represented by a locally dense anchor-zone observation.
    # This gate proves location only; the later mesh remains a simplified node,
    # never a claimed catalogue profile.
    attachments = _attachment_clusters(hardware, main) if main is not None else []
    # Point clouds often sample the two ends of a single physical insulator
    # rather than its catalogue-shaped body.  One resolved attachment cluster
    # plus a connected brace is sufficient for a simplified insulator node.
    insulator_passed = cantilever_passed and bool(attachments)
    positioner_passed = cantilever_passed and (
        len(brace_candidates) >= 2 or bool(secondary_structure["passed"])
    )
    gates = {
        "local_points_present": len(search) >= 30,
        "cantilever_transverse_trace_supported": cantilever_passed,
        "insulator_attachment_cluster_supported": insulator_passed,
        "positioner_or_steady_arm_trace_supported": positioner_passed,
    }
    passed = all(gates.values())
    return {
        "point_count": len(points),
        "top_search_point_count": len(search),
        "hardware_point_count": len(hardware),
        "conductor_like_point_count": int(np.count_nonzero(conductor_like)),
        "component_evidence": components,
        "linear_traces": traces,
        "selected_main_cantilever_trace": main,
        "selected_brace_or_positioner_traces": brace_candidates,
        "secondary_structure_evidence": secondary_structure,
        "attachment_clusters": attachments,
        "gates": gates,
        "passed": passed,
        "disposition": (
            "build_simplified_point_supported_top_equipment"
            if passed
            else "withhold_top_equipment_insufficient_component_evidence"
        ),
        "limitations": [
            "Longitudinally persistent traces are excluded as likely conductors.",
            "Insulator evidence supports anchor locations, not a catalogue product profile.",
            "A simplified insulator node records a resolved attachment location, not a catalogue profile.",
            "All component gates must pass before any top-equipment geometry is emitted.",
        ],
    }


def _render_montage(records: list[dict[str, Any]], points: dict[str, np.ndarray], path: Path) -> None:
    panel_w, panel_h = 520, 320
    columns = 2
    rows = int(np.ceil(len(records) / columns))
    canvas = Image.new("RGB", (panel_w * columns, 76 + panel_h * rows), "#06141d")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), "CATENARY TOP EQUIPMENT / LOCAL POINT EVIDENCE", fill="#eaf9ff", font=_font(23))
    draw.text((24, 48), "cross-section: inward reach vs elevation; green=gate pass, orange=withheld", fill="#8faab3", font=_font(14))
    for index, record in enumerate(records):
        col, row = index % columns, index // columns
        x0, y0 = col * panel_w, 76 + row * panel_h
        draw.rectangle((x0 + 8, y0 + 8, x0 + panel_w - 8, y0 + panel_h - 8), outline="#244957")
        colour = "#b7ff3c" if record["evidence"]["passed"] else "#ff9f31"
        draw.text((x0 + 20, y0 + 18), record["asset_id"], fill=colour, font=_font(15))
        values = points[record["asset_id"]]
        if len(values):
            selected = values[(values[:, 1] >= -0.3) & (values[:, 1] <= 4.8)]
            for ds, u, z in selected[:: max(1, len(selected) // 3500)]:
                px = x0 + 38 + (u + 0.3) / 5.1 * (panel_w - 70)
                py = y0 + panel_h - 38 - (z - (record["shaft_top_z_m"] - 4.8)) / 5.2 * (panel_h - 82)
                if abs(ds) <= 0.55:
                    point_colour = "#60d8e8"
                else:
                    point_colour = "#435f68"
                draw.point((int(px), int(py)), fill=point_colour)
        draw.line((x0 + 38, y0 + 42, x0 + 38, y0 + panel_h - 38), fill="#2f5663")
        draw.line((x0 + 38, y0 + panel_h - 38, x0 + panel_w - 28, y0 + panel_h - 38), fill="#2f5663")
        gates = record["evidence"]["gates"]
        label = f"points={record['evidence']['hardware_point_count']}  gates={sum(gates.values())}/{len(gates)}"
        draw.text((x0 + 20, y0 + panel_h - 27), label, fill=colour, font=_font(13))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def extract_catenary_top_evidence(
    *,
    cloud_path: str | Path,
    obj_path: str | Path,
    origin_path: str | Path,
    registry_path: str | Path,
    frame_report_path: str | Path,
    inventory_path: str | Path,
    output_directory: str | Path,
    chunk_size: int = 2_000_000,
) -> dict[str, Path]:
    cloud = Path(cloud_path).resolve()
    obj = Path(obj_path).resolve()
    origin_file = Path(origin_path).resolve()
    registry_file = Path(registry_path).resolve()
    frame_file = Path(frame_report_path).resolve()
    inventory_file = Path(inventory_path).resolve()
    for source in (cloud, obj, origin_file, registry_file, frame_file, inventory_file):
        if not source.is_file():
            raise FileNotFoundError(source)
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty evidence directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    registry = load_json(registry_file)
    inventory = load_json(inventory_file)
    frame = load_json(frame_file)["frame"]
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    model_origin = np.asarray(load_json(origin_file)["origin_xyz"], dtype=np.float64)
    model = parse_obj_model(obj)
    by_id = {str(asset["id"]): asset for asset in registry.get("assets", [])}
    targets = [
        item
        for item in inventory.get("records", [])
        if item.get("priority") == "P0_extract_top_equipment_evidence"
    ]
    if len(targets) != 7:
        raise ValueError(f"Expected seven P0 mast targets; got {len(targets)}")

    metadata: list[dict[str, Any]] = []
    for item in targets:
        asset_id = str(item["asset_id"])
        asset = by_id[asset_id]
        parameters = asset.get("parameters", {})
        station = float(asset.get("chainage_m"))
        cross_position = float(parameters["cross_position_m"])
        node = str(asset.get("geometry", {}).get("node", asset_id))
        indices = object_vertex_indices(model, node)
        if not len(indices):
            raise ValueError(f"Mast mesh is missing: {node}")
        world = model.vertices[indices] + model_origin
        metadata.append(
            {
                "asset_id": asset_id,
                "node": node,
                "station_m": station,
                "cross_m": cross_position,
                "shaft_bottom_z_m": float(np.min(world[:, 2])),
                "shaft_top_z_m": float(np.max(world[:, 2])),
            }
        )
    corridor_cross = float(np.median([item["cross_m"] for item in metadata]))
    # The two mast rows straddle the track field; use their midpoint, not the
    # median member, to determine the inward arm direction.
    row_values = sorted({round(item["cross_m"], 1) for item in metadata})
    if len(row_values) >= 2:
        corridor_cross = float((row_values[0] + row_values[-1]) / 2.0)

    point_groups: dict[str, list[np.ndarray]] = {item["asset_id"]: [] for item in metadata}
    with laspy.open(cloud) as reader:
        source_point_count = int(reader.header.point_count)
        for chunk in reader.chunk_iterator(chunk_size):
            xyz = np.column_stack(
                (
                    np.asarray(chunk.x, dtype=np.float64),
                    np.asarray(chunk.y, dtype=np.float64),
                    np.asarray(chunk.z, dtype=np.float64),
                )
            )
            relative = xyz[:, :2] - frame_origin
            station = relative @ along
            lateral = relative @ cross
            for item in metadata:
                direction = 1.0 if corridor_cross >= item["cross_m"] else -1.0
                ds = station - item["station_m"]
                u = direction * (lateral - item["cross_m"])
                selected = (
                    (np.abs(ds) <= 2.25)
                    & (u >= -0.35)
                    & (u <= 4.8)
                    & (xyz[:, 2] >= item["shaft_top_z_m"] - 4.8)
                    & (xyz[:, 2] <= item["shaft_top_z_m"] + 0.35)
                )
                if np.any(selected):
                    point_groups[item["asset_id"]].append(
                        np.column_stack((ds[selected], u[selected], xyz[selected, 2]))
                    )

    materialized = {
        asset_id: np.vstack(chunks) if chunks else np.empty((0, 3), dtype=np.float64)
        for asset_id, chunks in point_groups.items()
    }
    np.savez_compressed(
        output / "catenary_top_equipment_local_points.npz",
        **{item["asset_id"].replace("-", "_"): values for item, values in zip(metadata, materialized.values(), strict=True)},
    )
    records: list[dict[str, Any]] = []
    for item in metadata:
        values = materialized[item["asset_id"]]
        direction = 1 if corridor_cross >= item["cross_m"] else -1
        evidence = evaluate_top_equipment_evidence(
            values, shaft_top_z_m=item["shaft_top_z_m"]
        )
        records.append(
            {
                **item,
                "inward_cross_direction": direction,
                "evidence": evidence,
            }
        )

    report_path = output / "catenary_top_equipment_evidence.json"
    montage_path = output / "catenary_top_equipment_evidence_montage.png"
    report = {
        "schema_version": "railway.catenary-top-equipment-evidence.v1",
        "inputs": {
            "cloud": str(cloud),
            "cloud_sha256": sha256_file(cloud),
            "cloud_point_count": source_point_count,
            "obj": str(obj),
            "obj_sha256": sha256_file(obj),
            "registry": str(registry_file),
            "frame_report": str(frame_file),
            "inventory": str(inventory_file),
        },
        "method": {
            "coordinate_system": "station / inward-cross / elevation",
            "station_half_window_m": 2.25,
            "hardware_station_half_width_m": 0.55,
            "longitudinal_conductor_rejection": "u-z voxel station span >= 70 percent of local window",
            "build_policy": "all cantilever, two-anchor-layer and positioner gates must pass",
        },
        "corridor_cross_midline_m": corridor_cross,
        "records": records,
        "summary": {
            "target_count": len(records),
            "build_eligible_count": sum(item["evidence"]["passed"] for item in records),
            "withheld_count": sum(not item["evidence"]["passed"] for item in records),
        },
        "geometry_write": False,
        "registry_write": False,
        "status": "top_equipment_evidence_extracted_geometry_unchanged",
    }
    write_json(report_path, report)
    _render_montage(records, materialized, montage_path)
    return {
        "report": report_path,
        "montage": montage_path,
        "local_points": output / "catenary_top_equipment_local_points.npz",
    }
