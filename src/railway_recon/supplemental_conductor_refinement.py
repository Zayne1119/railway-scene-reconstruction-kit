from __future__ import annotations

import copy
import os
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import laspy
import numpy as np

from .algorithms.mesh import ObjWriter, sweep_mesh
from .catenary_conductor_mesh import circular_profile
from .io import load_json, sha256_file, write_json
from .mesh_audit import audit_obj
from .model_point_support import object_vertex_indices, parse_obj_model
from .registry import new_registry, summarize_registry, validate_registry_value
from .targeted_canopy_integration import merge_candidate_objs


def fit_sagged_conductor_span(
    points_station_cross_z: np.ndarray,
    station_start_m: float,
    station_end_m: float,
    *,
    station_bin_m: float = 0.25,
    endpoint_exclusion_m: float = 0.6,
    minimum_points_per_bin: int = 5,
    maximum_residual_p90_m: float = 0.10,
) -> dict[str, Any]:
    if station_end_m <= station_start_m:
        raise ValueError("Conductor span station range must be increasing")
    points = np.asarray(points_station_cross_z, dtype=np.float64)
    selected = (
        (points[:, 0] >= station_start_m + endpoint_exclusion_m)
        & (points[:, 0] <= station_end_m - endpoint_exclusion_m)
    )
    span_points = points[selected]
    if len(span_points) < 50:
        raise ValueError("Insufficient conductor points inside support span")
    bin_index = np.floor((span_points[:, 0] - station_start_m) / station_bin_m).astype(
        np.int64
    )
    bins: list[list[float]] = []
    for value in np.unique(bin_index):
        member = bin_index == value
        if int(np.sum(member)) < minimum_points_per_bin:
            continue
        bins.append([float(item) for item in np.median(span_points[member], axis=0)])
    samples = np.asarray(bins, dtype=np.float64)
    if len(samples) < 20:
        raise ValueError("Insufficient occupied station bins for conductor fit")
    center = (station_start_m + station_end_m) / 2.0
    relative = samples[:, 0] - center
    keep = np.ones(len(samples), dtype=bool)
    cross_coefficients = np.zeros(2)
    z_coefficients = np.zeros(3)
    for _ in range(6):
        cross_coefficients = np.polyfit(relative[keep], samples[keep, 1], 1)
        z_coefficients = np.polyfit(relative[keep], samples[keep, 2], 2)
        cross_residual = samples[:, 1] - np.polyval(cross_coefficients, relative)
        z_residual = samples[:, 2] - np.polyval(z_coefficients, relative)
        residual = np.hypot(cross_residual, z_residual)
        median = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - median)))
        tolerance = min(0.15, max(0.035, median + 3.0 * 1.4826 * mad))
        updated = residual <= tolerance
        if int(np.sum(updated)) < 20 or np.array_equal(updated, keep):
            break
        keep = updated
    cross_residual = samples[:, 1] - np.polyval(cross_coefficients, relative)
    z_residual = samples[:, 2] - np.polyval(z_coefficients, relative)
    residual = np.hypot(cross_residual, z_residual)
    inlier_residual = residual[keep]
    residual_p90 = float(np.percentile(inlier_residual, 90))
    curvature = float(z_coefficients[0])
    half_length = (station_end_m - station_start_m) / 2.0
    endpoint_z = np.polyval(z_coefficients, np.asarray((-half_length, half_length)))
    middle_z = float(np.polyval(z_coefficients, 0.0))
    sag = float(np.mean(endpoint_z) - middle_z)
    gates = {
        "minimum_occupied_station_bins": len(samples) >= 20,
        "minimum_inlier_fraction": float(np.mean(keep)) >= 0.70,
        "residual_p90_within_limit": residual_p90 <= maximum_residual_p90_m,
        "positive_sag_curvature": curvature > 0.0,
        "visible_sag_depth": sag >= 0.05,
    }
    return {
        "station_start_m": station_start_m,
        "station_end_m": station_end_m,
        "station_center_m": center,
        "source_point_count": len(span_points),
        "occupied_station_bin_count": len(samples),
        "inlier_station_bin_count": int(np.sum(keep)),
        "inlier_fraction": float(np.mean(keep)),
        "cross_polynomial_relative_to_center": [
            float(value) for value in cross_coefficients
        ],
        "z_polynomial_relative_to_center": [float(value) for value in z_coefficients],
        "residual_p50_m": float(np.percentile(inlier_residual, 50)),
        "residual_p90_m": residual_p90,
        "sag_depth_m": sag,
        "gates": gates,
        "passed": all(gates.values()),
    }


def evaluate_span_fit(fit: dict[str, Any], station: np.ndarray) -> np.ndarray:
    values = np.asarray(station, dtype=np.float64)
    relative = values - float(fit["station_center_m"])
    cross = np.polyval(
        np.asarray(fit["cross_polynomial_relative_to_center"], dtype=np.float64),
        relative,
    )
    z = np.polyval(
        np.asarray(fit["z_polynomial_relative_to_center"], dtype=np.float64),
        relative,
    )
    return np.column_stack((values, cross, z))


def _catenary_supports(
    *,
    model: Any,
    origin_xyz: np.ndarray,
    registry: dict[str, Any],
    frame: dict[str, Any],
) -> list[dict[str, Any]]:
    origin_xy = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    supports: list[dict[str, Any]] = []
    for asset in registry.get("assets", []):
        if asset.get("type") != "catenary_mast":
            continue
        object_name = str(asset.get("geometry", {}).get("node", asset.get("id", "")))
        indices = object_vertex_indices(model, object_name)
        if not len(indices):
            continue
        center_xy = np.mean((model.vertices[indices] + origin_xyz)[:, :2], axis=0)
        local = center_xy - origin_xy
        supports.append(
            {
                "asset_id": str(asset["id"]),
                "station_m": float(local @ along),
                "cross_m": float(local @ cross),
            }
        )
    return supports


def _load_candidate_points(
    cloud_path: Path,
    candidates: list[dict[str, Any]],
    frame: dict[str, Any],
    *,
    cross_margin_m: float = 0.15,
    z_margin_m: float = 0.15,
    chunk_size: int = 2_000_000,
) -> dict[str, np.ndarray]:
    origin_xy = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    chunks: dict[str, list[np.ndarray]] = {str(item["candidate_id"]): [] for item in candidates}
    with laspy.open(cloud_path) as reader:
        for chunk in reader.chunk_iterator(chunk_size):
            x = np.asarray(chunk.x, dtype=np.float64)
            y = np.asarray(chunk.y, dtype=np.float64)
            z = np.asarray(chunk.z, dtype=np.float64)
            local_xy = np.column_stack((x, y)) - origin_xy
            station = local_xy @ along
            lateral = local_xy @ cross
            for candidate in candidates:
                station_range = candidate["station_range_m"]
                cross_range = candidate["cross_range_m"]
                z_range = candidate["z_range_m"]
                selected = (
                    (station >= station_range[0])
                    & (station <= station_range[1])
                    & (lateral >= cross_range[0] - cross_margin_m)
                    & (lateral <= cross_range[1] + cross_margin_m)
                    & (z >= z_range[0] - z_margin_m)
                    & (z <= z_range[1] + z_margin_m)
                )
                if np.any(selected):
                    chunks[str(candidate["candidate_id"])].append(
                        np.column_stack((station[selected], lateral[selected], z[selected]))
                    )
    return {
        candidate_id: np.vstack(values) if values else np.empty((0, 3), dtype=np.float64)
        for candidate_id, values in chunks.items()
    }


def _global_centerline(
    fits: list[dict[str, Any]],
    frame: dict[str, Any],
    *,
    sample_spacing_m: float = 0.5,
) -> np.ndarray:
    local_segments: list[np.ndarray] = []
    for index, fit in enumerate(fits):
        start = float(fit["station_start_m"])
        end = float(fit["station_end_m"])
        count = max(3, int(np.ceil((end - start) / sample_spacing_m)) + 1)
        station = np.linspace(start, end, count)
        local = evaluate_span_fit(fit, station)
        if index:
            previous = local_segments[-1][-1]
            joint = (previous + local[0]) / 2.0
            local_segments[-1][-1] = joint
            local[0] = joint
            local = local[1:]
        local_segments.append(local)
    local_centerline = np.vstack(local_segments)
    origin_xy = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    xy = (
        origin_xy[None, :]
        + local_centerline[:, 0, None] * along[None, :]
        + local_centerline[:, 1, None] * cross[None, :]
    )
    return np.column_stack((xy, local_centerline[:, 2]))


def _write_conductor_material(path: Path) -> None:
    content = """# Supplemental point-cloud conductor material
newmtl SupplementalAuxConductor
Kd 0.95 0.62 0.10
Ks 0.55 0.55 0.55
Ns 70
"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def build_supplemental_conductor_candidate(
    *,
    cloud_path: str | Path,
    gap_report_path: str | Path,
    frame_report_path: str | Path,
    source_obj: str | Path,
    source_origin: str | Path,
    source_registry: str | Path,
    output_directory: str | Path,
    render_radius_m: float = 0.018,
) -> dict[str, Path]:
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty conductor candidate: {output}")
    output.mkdir(parents=True, exist_ok=True)
    gap_report = load_json(Path(gap_report_path))
    frame = load_json(Path(frame_report_path))["frame"]
    candidates = [
        item
        for item in gap_report["overhead_linear_candidates"]
        if item["priority"] == "P0"
        and float(item["extent_station_cross_z_m"][0]) >= 100.0
    ]
    if len(candidates) != 2:
        raise ValueError(f"Expected two long P0 conductor candidates; got {len(candidates)}")
    model = parse_obj_model(source_obj)
    origin_value = load_json(Path(source_origin))
    origin = np.asarray(origin_value["origin_xyz"], dtype=np.float64)
    registry = load_json(Path(source_registry))
    supports = _catenary_supports(
        model=model,
        origin_xyz=origin,
        registry=registry,
        frame=frame,
    )
    candidate_points = _load_candidate_points(Path(cloud_path), candidates, frame)

    component_obj = output / "supplemental_aux_conductors.obj"
    component_mtl = output / "supplemental_aux_conductors.mtl"
    component_origin = output / "supplemental_aux_conductors_origin.json"
    writer = ObjWriter(origin, material_library=component_mtl.name)
    fit_records: list[dict[str, Any]] = []
    added_assets: list[dict[str, Any]] = []
    for sequence, candidate in enumerate(candidates, start=1):
        cross_center = float(np.mean(candidate["cross_range_m"]))
        candidate_supports = sorted(
            (
                item
                for item in supports
                if abs(float(item["cross_m"]) - cross_center) <= 1.0
                and float(candidate["station_range_m"][0]) - 1.0
                <= float(item["station_m"])
                <= float(candidate["station_range_m"][1]) + 1.0
            ),
            key=lambda item: float(item["station_m"]),
        )
        if len(candidate_supports) < 4:
            raise ValueError(
                f"Conductor candidate {candidate['candidate_id']} lacks four support masts"
            )
        points = candidate_points[str(candidate["candidate_id"])]
        fits: list[dict[str, Any]] = []
        for left, right in pairwise(candidate_supports):
            fit = fit_sagged_conductor_span(
                points,
                float(left["station_m"]),
                float(right["station_m"]),
            )
            fit["left_support_asset_id"] = left["asset_id"]
            fit["right_support_asset_id"] = right["asset_id"]
            if not fit["passed"]:
                raise ValueError(
                    f"Conductor span failed fit gates: {candidate['candidate_id']}: {fit['gates']}"
                )
            fits.append(fit)
        centerline = _global_centerline(fits, frame)
        vertices, faces = sweep_mesh(
            centerline,
            circular_profile(render_radius_m, segment_count=8),
        )
        asset_id = f"SUPPLEMENTAL-AUX-CONDUCTOR-{sequence:02d}"
        writer.add_mesh(asset_id, vertices, faces, "SupplementalAuxConductor")
        residual_p90 = max(float(fit["residual_p90_m"]) for fit in fits)
        fit_records.append(
            {
                "asset_id": asset_id,
                "source_candidate_id": candidate["candidate_id"],
                "source_point_count": len(points),
                "support_assets": [item["asset_id"] for item in candidate_supports],
                "span_fits": fits,
                "maximum_span_residual_p90_m": residual_p90,
                "centerline_point_count": len(centerline),
            }
        )
        added_assets.append(
            {
                "id": asset_id,
                "type": "catenary_auxiliary_conductor",
                "subtype": "point_supported_sagged_conductor_semantic_pending",
                "status": "candidate",
                "chainage_m": None,
                "evidence_level": "observed",
                "confidence": 0.90,
                "sources": [
                    {"kind": "point_cloud", "reference": str(Path(cloud_path).resolve())},
                    {
                        "kind": "rule",
                        "reference": str(Path(gap_report_path).resolve()),
                        "note": "Reverse cloud-to-model gap audit candidate and support-constrained sag fit.",
                    },
                ],
                "parameters": {
                    "source_candidate_id": candidate["candidate_id"],
                    "support_asset_ids": [item["asset_id"] for item in candidate_supports],
                    "span_count": len(fits),
                    "maximum_span_residual_p90_m": residual_p90,
                    "render_radius_m": render_radius_m,
                    "unsupported_extrapolation": False,
                },
                "geometry": {"file": "", "node": asset_id},
                "limitations": [
                    "Point-cloud geometry is confirmed; professional subtype remains auxiliary conductor pending domain review.",
                    "Render radius is enlarged for visibility and is not a measured cable diameter.",
                ],
            }
        )
    writer.write(component_obj)
    _write_conductor_material(component_mtl)
    write_json(component_origin, origin_value)

    output_name = output.name
    output_obj = output / f"{output_name}.obj"
    output_mtl = output / f"{output_name}.mtl"
    output_origin = output / "model_origin.json"
    merge_candidate_objs(
        [
            (Path(source_obj).resolve(), Path(source_origin).resolve()),
            (component_obj, component_origin),
        ],
        output_obj,
        output_mtl,
        output_origin,
    )
    output_registry = output / "asset_registry.json"
    updated_registry = copy.deepcopy(registry)
    for asset in updated_registry.get("assets", []):
        if isinstance(asset.get("geometry"), dict):
            asset["geometry"]["file"] = str(output_obj)
    for asset in added_assets:
        asset["geometry"]["file"] = str(output_obj)
        updated_registry["assets"].append(asset)
    updated_registry["release_id"] = output_name
    updated_registry["updated_at"] = datetime.now(UTC).isoformat()
    updated_registry["summary"] = summarize_registry(updated_registry)
    # The inherited run09 registry predates the public v1 vocabulary and contains
    # historical workflow statuses. Preserve it verbatim for traceability, but do
    # not let those inherited findings hide defects in this run's two new assets.
    inherited_validation_errors = validate_registry_value(registry)
    addition_registry = new_registry(str(registry.get("project_id", "site-b")))
    addition_registry["assets"] = copy.deepcopy(added_assets)
    addition_registry["summary"] = summarize_registry(addition_registry)
    new_asset_validation_errors = validate_registry_value(addition_registry)
    if new_asset_validation_errors:
        raise ValueError(
            "New conductor assets are invalid: " + "; ".join(new_asset_validation_errors)
        )
    write_json(output_registry, updated_registry)
    output_audit = output / "mesh_audit.json"
    mesh_audit = audit_obj(output_obj)
    write_json(output_audit, mesh_audit)
    if not mesh_audit["passed"]:
        raise ValueError("Supplemental conductor candidate failed mesh audit")
    output_report = output / "supplemental_conductor_refinement_report.json"
    write_json(
        output_report,
        {
            "schema_version": "railway.supplemental-conductor-refinement.v1",
            "source_obj": str(Path(source_obj).resolve()),
            "output_obj": str(output_obj),
            "gap_report": str(Path(gap_report_path).resolve()),
            "fit_records": fit_records,
            "added_asset_count": len(added_assets),
            "output_asset_count": int(updated_registry["summary"]["asset_count"]),
            "new_assets_schema_valid": True,
            "inherited_registry_validation_error_count": len(
                inherited_validation_errors
            ),
            "inherited_registry_validation_errors": inherited_validation_errors,
            "mesh_audit_passed": True,
            "output_obj_sha256": sha256_file(output_obj),
            "status": "candidate_requires_point_support_and_fixed_view_review",
        },
    )
    return {
        "obj": output_obj,
        "mtl": output_mtl,
        "origin": output_origin,
        "registry": output_registry,
        "mesh_audit": output_audit,
        "report": output_report,
    }
