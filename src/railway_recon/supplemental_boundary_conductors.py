from __future__ import annotations

import copy
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .algorithms.mesh import ObjWriter, sweep_mesh
from .catenary_conductor_mesh import circular_profile
from .io import load_json, sha256_file, write_json
from .mesh_audit import audit_obj
from .registry import new_registry, summarize_registry, validate_registry_value
from .supplemental_conductor_refinement import (
    _catenary_supports,
    _load_candidate_points,
)
from .targeted_canopy_integration import merge_candidate_objs


def fit_boundary_conductor_fragment(
    points_station_cross_z: np.ndarray,
    *,
    station_start_m: float,
    station_end_m: float,
    station_bin_m: float = 0.20,
    minimum_points_per_bin: int = 5,
) -> dict[str, Any]:
    """Fit a short observed outbound conductor without assuming a complete sag span."""
    points = np.asarray(points_station_cross_z, dtype=np.float64)
    selected = (
        (points[:, 0] >= station_start_m)
        & (points[:, 0] <= station_end_m)
    )
    points = points[selected]
    if len(points) < 50:
        raise ValueError("Insufficient boundary conductor points")
    bins = np.floor((points[:, 0] - station_start_m) / station_bin_m).astype(np.int64)
    medians: list[np.ndarray] = []
    for value in np.unique(bins):
        member = bins == value
        if int(np.sum(member)) >= minimum_points_per_bin:
            medians.append(np.median(points[member], axis=0))
    samples = np.asarray(medians, dtype=np.float64)
    if len(samples) < 15:
        raise ValueError("Insufficient occupied bins for boundary conductor fit")
    center = float(np.mean((station_start_m, station_end_m)))
    relative = samples[:, 0] - center
    keep = np.ones(len(samples), dtype=bool)
    cross_coefficients = np.zeros(2, dtype=np.float64)
    z_coefficients = np.zeros(3, dtype=np.float64)
    for _ in range(6):
        cross_coefficients = np.polyfit(relative[keep], samples[keep, 1], 1)
        z_coefficients = np.polyfit(relative[keep], samples[keep, 2], 2)
        residual = np.hypot(
            samples[:, 1] - np.polyval(cross_coefficients, relative),
            samples[:, 2] - np.polyval(z_coefficients, relative),
        )
        median = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - median)))
        tolerance = min(0.15, max(0.035, median + 3.0 * 1.4826 * mad))
        updated = residual <= tolerance
        if int(np.sum(updated)) < 15 or np.array_equal(updated, keep):
            break
        keep = updated
    residual = np.hypot(
        samples[:, 1] - np.polyval(cross_coefficients, relative),
        samples[:, 2] - np.polyval(z_coefficients, relative),
    )
    inlier_residual = residual[keep]
    residual_p90 = float(np.percentile(inlier_residual, 90))
    gates = {
        "minimum_occupied_station_bins": len(samples) >= 15,
        "minimum_inlier_fraction": float(np.mean(keep)) >= 0.65,
        "residual_p90_below_8cm": residual_p90 <= 0.08,
        "observed_length_at_least_5m": station_end_m - station_start_m >= 5.0,
    }
    return {
        "station_start_m": station_start_m,
        "station_end_m": station_end_m,
        "station_center_m": center,
        "source_point_count": len(points),
        "occupied_station_bin_count": len(samples),
        "inlier_station_bin_count": int(np.sum(keep)),
        "inlier_fraction": float(np.mean(keep)),
        "cross_polynomial_relative_to_center": cross_coefficients.tolist(),
        "z_polynomial_relative_to_center": z_coefficients.tolist(),
        "residual_p50_m": float(np.percentile(inlier_residual, 50)),
        "residual_p90_m": residual_p90,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _evaluate_fit(fit: dict[str, Any], station: np.ndarray) -> np.ndarray:
    relative = station - float(fit["station_center_m"])
    cross = np.polyval(fit["cross_polynomial_relative_to_center"], relative)
    z = np.polyval(fit["z_polynomial_relative_to_center"], relative)
    return np.column_stack((station, cross, z))


def _local_to_world(local: np.ndarray, frame: dict[str, Any]) -> np.ndarray:
    origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    xy = origin + local[:, 0, None] * along + local[:, 1, None] * cross
    return np.column_stack((xy, local[:, 2]))


def _write_material(path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "# Point-supported outbound boundary conductors\n"
        "newmtl SupplementalBoundaryConductor\n"
        "Kd 0.93 0.57 0.10\nKs 0.50 0.50 0.50\nNs 70\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def build_supplemental_boundary_conductors(
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
        raise FileExistsError(f"Refusing to overwrite non-empty candidate directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    gap_path = Path(gap_report_path).resolve()
    gap = load_json(gap_path)
    frame = load_json(Path(frame_report_path))["frame"]
    scope_max = float(gap["corridor_scope"]["station_maximum_m"])
    candidates = [
        item
        for item in gap["overhead_linear_candidates"]
        if item.get("asset_relation") == "unmatched_overhead_linear_candidate"
        and item.get("priority") in {"P0", "P1"}
        and item.get("corridor_scope_ownership") == "crosses_segment_boundary"
        and float(item["station_range_m"][1]) > scope_max
    ]
    if not candidates:
        raise ValueError("No unmatched next-boundary conductor candidates were found")

    model_origin_value = load_json(Path(source_origin))
    model_origin = np.asarray(model_origin_value["origin_xyz"], dtype=np.float64)
    source_model_path = Path(source_obj).resolve()
    source_registry_path = Path(source_registry).resolve()
    source_registry_value = load_json(source_registry_path)
    from .model_point_support import parse_obj_model

    model = parse_obj_model(source_model_path)
    supports = _catenary_supports(
        model=model,
        origin_xyz=model_origin,
        registry=source_registry_value,
        frame=frame,
    )
    points_by_id = _load_candidate_points(
        Path(cloud_path).resolve(),
        candidates,
        frame,
        cross_margin_m=0.12,
        z_margin_m=0.12,
    )

    component_obj = output / "supplemental_boundary_conductors.obj"
    component_mtl = output / "supplemental_boundary_conductors.mtl"
    component_origin = output / "supplemental_boundary_conductors_origin.json"
    writer = ObjWriter(model_origin, material_library=component_mtl.name)
    additions: list[dict[str, Any]] = []
    fit_records: list[dict[str, Any]] = []
    for sequence, candidate in enumerate(candidates, start=1):
        start = float(candidate["station_range_m"][0])
        cross_range = [float(value) for value in candidate["cross_range_m"]]
        candidate_supports = [
            support
            for support in supports
            if abs(float(support["station_m"]) - start) <= 1.0
            and max(
                cross_range[0] - float(support["cross_m"]),
                float(support["cross_m"]) - cross_range[1],
                0.0,
            )
            <= 0.75
        ]
        if not candidate_supports:
            raise ValueError(f"{candidate['candidate_id']} lacks a boundary support mast")
        support = min(
            candidate_supports,
            key=lambda item: abs(float(item["station_m"]) - start),
        )
        fit_start = max(start, float(support["station_m"])) + 0.45
        fit_end = min(float(candidate["station_range_m"][1]), scope_max) - 0.15
        fit = fit_boundary_conductor_fragment(
            points_by_id[str(candidate["candidate_id"])],
            station_start_m=fit_start,
            station_end_m=fit_end,
        )
        if not fit["passed"]:
            raise ValueError(f"{candidate['candidate_id']} failed boundary conductor fit gate")
        build_start = max(start, float(support["station_m"]))
        build_end = min(float(candidate["station_range_m"][1]), scope_max)
        station = np.linspace(
            build_start,
            build_end,
            max(3, int(np.ceil((build_end - build_start) / 0.25)) + 1),
        )
        centerline = _local_to_world(_evaluate_fit(fit, station), frame)
        vertices, faces = sweep_mesh(centerline, circular_profile(render_radius_m, 8))
        asset_id = f"SUPPLEMENTAL-BOUNDARY-CONDUCTOR-{sequence:02d}"
        writer.add_mesh(asset_id, vertices, faces, "SupplementalBoundaryConductor")
        fit_records.append(
            {
                "candidate_id": candidate["candidate_id"],
                "support_asset_id": support["asset_id"],
                "build_station_range_m": [build_start, build_end],
                "fit": fit,
            }
        )
        additions.append(
            {
                "id": asset_id,
                "type": "catenary_auxiliary_conductor",
                "subtype": "point_supported_next_boundary_fragment_semantic_pending",
                "status": "candidate",
                "chainage_m": float(np.mean((build_start, build_end))),
                "evidence_level": "observed",
                "confidence": 0.88,
                "sources": [
                    {"kind": "point_cloud", "reference": str(Path(cloud_path).resolve())},
                    {"kind": "rule", "reference": str(gap_path)},
                ],
                "parameters": {
                    "source_candidate_id": candidate["candidate_id"],
                    "support_asset_id": support["asset_id"],
                    "longitudinal_interval_m": [build_start, build_end],
                    "fit_residual_p90_m": fit["residual_p90_m"],
                    "render_radius_m": render_radius_m,
                    "unsupported_extrapolation": False,
                    "continues_into_adjacent_segment": True,
                },
                "geometry": {"file": "", "node": asset_id},
                "limitations": [
                    "Professional conductor subtype remains pending.",
                    "Only the current-segment observed fragment is emitted; the next segment owns continuation.",
                    "Render radius is enlarged for visibility and is not a measured cable diameter.",
                ],
            }
        )

    writer.write(component_obj)
    _write_material(component_mtl)
    write_json(component_origin, model_origin_value)
    output_name = output.name
    output_obj = output / f"{output_name}.obj"
    output_mtl = output / f"{output_name}.mtl"
    output_origin = output / "model_origin.json"
    merge_candidate_objs(
        [(source_model_path, Path(source_origin).resolve()), (component_obj, component_origin)],
        output_obj,
        output_mtl,
        output_origin,
    )

    registry = copy.deepcopy(source_registry_value)
    for asset in registry.get("assets", []):
        if isinstance(asset.get("geometry"), dict):
            asset["geometry"]["file"] = str(output_obj)
    for asset in additions:
        asset["geometry"]["file"] = str(output_obj)
        registry["assets"].append(asset)
    registry["release_id"] = output_name
    registry["updated_at"] = datetime.now(UTC).isoformat()
    registry["summary"] = summarize_registry(registry)
    additions_registry = new_registry(str(registry.get("project_id", "site-b")))
    additions_registry["assets"] = copy.deepcopy(additions)
    additions_registry["summary"] = summarize_registry(additions_registry)
    errors = validate_registry_value(additions_registry)
    if errors:
        raise ValueError("Boundary conductor assets are invalid: " + "; ".join(errors))
    output_registry = output / "asset_registry.json"
    write_json(output_registry, registry)

    mesh = audit_obj(output_obj)
    output_mesh_audit = output / "mesh_audit.json"
    write_json(output_mesh_audit, mesh)
    if not mesh["passed"]:
        raise ValueError("Boundary conductor candidate failed mesh audit")
    report = output / "supplemental_boundary_conductors_report.json"
    write_json(
        report,
        {
            "schema_version": "railway.supplemental-boundary-conductors.v1",
            "source_obj": str(source_model_path),
            "output_obj": str(output_obj),
            "gap_report": str(gap_path),
            "corridor_station_maximum_m": scope_max,
            "added_asset_count": len(additions),
            "new_assets": additions,
            "fit_records": fit_records,
            "mesh_audit_passed": True,
            "output_obj_sha256": sha256_file(output_obj),
            "status": "candidate_built_gap_regression_and_fixed_view_review_required",
        },
    )
    return {
        "obj": output_obj,
        "mtl": output_mtl,
        "origin": output_origin,
        "registry": output_registry,
        "mesh_audit": output_mesh_audit,
        "report": report,
    }
