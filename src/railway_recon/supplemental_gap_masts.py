from __future__ import annotations

import copy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from scipy.spatial import cKDTree

from .algorithms.mesh import ObjWriter
from .catenary_candidate_mesh import h_section_prism
from .io import load_json, sha256_file, write_json
from .mesh_audit import audit_obj
from .model_point_support import parse_obj_model
from .registry import new_registry, summarize_registry, validate_registry_value
from .supplemental_endpoint_columns import _sample_object_surface
from .targeted_canopy_integration import merge_candidate_objs


def fit_observed_mast_shaft(
    points_station_cross_z: np.ndarray,
    *,
    minimum_z_m: float,
    maximum_z_m: float,
) -> dict[str, float]:
    points = np.asarray(points_station_cross_z, dtype=np.float64)
    if len(points) < 40:
        raise ValueError("Insufficient observed mast points")
    height = maximum_z_m - minimum_z_m
    if height < 5.5:
        raise ValueError("Reviewed mast candidate is too short")
    shaft = points[
        (points[:, 2] >= minimum_z_m + min(0.60, 0.12 * height))
        & (points[:, 2] <= maximum_z_m - min(0.55, 0.10 * height))
    ]
    if len(shaft) < 30:
        raise ValueError("Insufficient mid-shaft observations")
    s05, s50, s95 = np.percentile(shaft[:, 0], (5.0, 50.0, 95.0))
    c05, c50, c95 = np.percentile(shaft[:, 1], (5.0, 50.0, 95.0))
    bin_size_m = 0.25
    edges = np.arange(minimum_z_m, maximum_z_m + bin_size_m, bin_size_m)
    counts, _ = np.histogram(points[:, 2], bins=edges)
    occupied = counts >= 5
    longest_empty_run = 0
    current_empty_run = 0
    for is_occupied in occupied:
        if is_occupied:
            current_empty_run = 0
        else:
            current_empty_run += 1
            longest_empty_run = max(longest_empty_run, current_empty_run)
    occupied_ratio = float(np.mean(occupied)) if len(occupied) else 0.0
    return {
        "station_m": float(s50),
        "cross_m": float(c50),
        "along_width_m": float(np.clip(s95 - s05, 0.18, 0.34)),
        "cross_depth_m": float(np.clip(c95 - c05, 0.18, 0.36)),
        "base_z_m": float(minimum_z_m),
        "top_z_m": float(maximum_z_m),
        "height_m": float(height),
        "source_point_count": len(points),
        "shaft_point_count": len(shaft),
        "vertical_bin_size_m": bin_size_m,
        "occupied_vertical_bin_ratio": occupied_ratio,
        "longest_empty_vertical_gap_m": float(longest_empty_run * bin_size_m),
        "continuous_shaft_observed": occupied_ratio >= 0.75 and longest_empty_run <= 2,
    }


def _load_candidate_points(
    cloud_path: Path,
    candidates: list[dict[str, Any]],
    frame: dict[str, Any],
    *,
    chunk_size: int = 2_000_000,
    horizontal_padding_m: float = 0.06,
) -> dict[str, np.ndarray]:
    origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    groups: dict[str, list[np.ndarray]] = {
        str(item["candidate_id"]): [] for item in candidates
    }
    with laspy.open(cloud_path) as reader:
        for chunk in reader.chunk_iterator(chunk_size):
            xyz = np.column_stack(
                (
                    np.asarray(chunk.x, dtype=np.float64),
                    np.asarray(chunk.y, dtype=np.float64),
                    np.asarray(chunk.z, dtype=np.float64),
                )
            )
            local = xyz[:, :2] - origin
            station = local @ along
            lateral = local @ cross
            for item in candidates:
                minimum = np.asarray(item["minimum_xyz_m"], dtype=np.float64).copy()
                maximum = np.asarray(item["maximum_xyz_m"], dtype=np.float64).copy()
                minimum[:2] -= horizontal_padding_m
                maximum[:2] += horizontal_padding_m
                selected = np.all((xyz >= minimum) & (xyz <= maximum), axis=1)
                if np.any(selected):
                    groups[str(item["candidate_id"])].append(
                        np.column_stack((station[selected], lateral[selected], xyz[selected, 2]))
                    )
    return {
        key: np.vstack(value) if value else np.empty((0, 3), dtype=np.float64)
        for key, value in groups.items()
    }


def _write_addon_material(path: Path) -> None:
    path.write_text(
        """# Supplemental observed catenary mast material
newmtl SupplementalObservedMastSteel
Ka 0.12 0.14 0.15
Kd 0.42 0.46 0.48
Ks 0.20 0.22 0.23
Ns 36.0
d 1.0
illum 2
""",
        encoding="utf-8",
        newline="\n",
    )


def _point_support(
    model: Any,
    origin: np.ndarray,
    object_name: str,
    points: np.ndarray,
    frame: dict[str, Any],
    minimum_z_m: float,
    maximum_z_m: float,
) -> dict[str, Any]:
    height = maximum_z_m - minimum_z_m
    shaft = points[
        (points[:, 2] >= minimum_z_m + min(0.60, 0.12 * height))
        & (points[:, 2] <= maximum_z_m - min(0.55, 0.10 * height))
    ]
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    shaft_xy = (
        frame_origin[None, :]
        + shaft[:, 0, None] * along[None, :]
        + shaft[:, 1, None] * cross[None, :]
    )
    shaft_world = np.column_stack((shaft_xy, shaft[:, 2]))
    surface = _sample_object_surface(model, object_name, origin, spacing_m=0.025)
    distances, _ = cKDTree(surface).query(shaft_world, workers=-1)
    gates = {
        "p90_within_0_16m": float(np.percentile(distances, 90)) <= 0.16,
        "coverage_at_0_20m_at_least_0_90": float(np.mean(distances <= 0.20)) >= 0.90,
    }
    return {
        "direction": "observed_lower_and_upper_candidate_bands_to_photo_interpreted_mesh_surface",
        "shaft_point_count": len(shaft),
        "p50_m": float(np.percentile(distances, 50)),
        "p90_m": float(np.percentile(distances, 90)),
        "p95_m": float(np.percentile(distances, 95)),
        "coverage_at_0_10m": float(np.mean(distances <= 0.10)),
        "coverage_at_0_20m": float(np.mean(distances <= 0.20)),
        "gates": gates,
        "passed": all(gates.values()),
        "continuity_interpretation": (
            "Point bands constrain plan position and vertical endpoints; the middle shaft is "
            "photo-interpreted because it is not continuously sampled."
        ),
    }


def build_supplemental_gap_masts(
    *,
    cloud_path: str | Path,
    gap_report_path: str | Path,
    disposition_path: str | Path,
    frame_report_path: str | Path,
    source_obj: str | Path,
    source_origin: str | Path,
    source_registry: str | Path,
    output_directory: str | Path,
) -> dict[str, Path]:
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty candidate directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    cloud = Path(cloud_path).resolve()
    gap_path = Path(gap_report_path).resolve()
    disposition_file = Path(disposition_path).resolve()
    source_model = Path(source_obj).resolve()
    source_origin_file = Path(source_origin).resolve()
    source_registry_file = Path(source_registry).resolve()
    gap = load_json(gap_path)
    dispositions = load_json(disposition_file)
    frame = load_json(Path(frame_report_path))["frame"]
    candidate_by_id = {
        str(item["candidate_id"]): item for item in gap["vertical_candidates"]
    }
    decisions = [
        item
        for item in dispositions.get("decisions", [])
        if item.get("reviewed_class") == "catenary_mast"
        and item.get("model_action") == "build_candidate"
    ]
    if len(decisions) != 2:
        raise ValueError(f"Expected two explicitly reviewed catenary masts; got {len(decisions)}")
    candidates = [candidate_by_id[str(item["candidate_id"])] for item in decisions]
    points_by_id = _load_candidate_points(cloud, candidates, frame)
    origin_value = load_json(source_origin_file)
    model_origin = np.asarray(origin_value["origin_xyz"], dtype=np.float64)
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)

    addon_obj = output / "supplemental_gap_masts.obj"
    addon_mtl = output / "supplemental_gap_masts.mtl"
    addon_origin = output / "supplemental_gap_masts_origin.json"
    writer = ObjWriter(model_origin, material_library=addon_mtl.name)
    records: list[dict[str, Any]] = []
    for decision, candidate in zip(decisions, candidates, strict=True):
        candidate_id = str(candidate["candidate_id"])
        fit = fit_observed_mast_shaft(
            points_by_id[candidate_id],
            minimum_z_m=float(candidate["minimum_xyz_m"][2]),
            maximum_z_m=float(candidate["maximum_xyz_m"][2]),
        )
        center_xy = frame_origin + fit["station_m"] * along + fit["cross_m"] * cross
        asset_id = f"SUPPLEMENTAL-CATENARY-MAST-{candidate_id.rsplit('-', 1)[-1]}"
        vertices, faces = h_section_prism(
            center_xy,
            along,
            cross,
            fit["along_width_m"],
            fit["cross_depth_m"],
            min(max(fit["cross_depth_m"] * 0.10, 0.018), fit["cross_depth_m"] * 0.20),
            min(max(fit["along_width_m"] * 0.08, 0.016), fit["along_width_m"] * 0.16),
            fit["base_z_m"],
            fit["top_z_m"],
        )
        writer.add_mesh(asset_id, vertices, faces, "SupplementalObservedMastSteel")
        records.append(
            {
                "candidate_id": candidate_id,
                "asset_id": asset_id,
                "confidence": float(decision["confidence"]),
                "evidence": decision["evidence"],
                "professional_subtype": decision.get("professional_subtype"),
                "fit": fit,
            }
        )
    writer.write(addon_obj)
    _write_addon_material(addon_mtl)
    write_json(addon_origin, origin_value)

    output_name = output.name
    output_obj = output / f"{output_name}.obj"
    output_mtl = output / f"{output_name}.mtl"
    output_origin = output / "model_origin.json"
    merge = merge_candidate_objs(
        [(source_model, source_origin_file), (addon_obj, addon_origin)],
        output_obj,
        output_mtl,
        output_origin,
    )

    addon_model = parse_obj_model(addon_obj)
    for record in records:
        candidate = candidate_by_id[record["candidate_id"]]
        record["point_to_mesh_support"] = _point_support(
            addon_model,
            model_origin,
            record["asset_id"],
            points_by_id[record["candidate_id"]],
            frame,
            float(candidate["minimum_xyz_m"][2]),
            float(candidate["maximum_xyz_m"][2]),
        )
    if not all(item["point_to_mesh_support"]["passed"] for item in records):
        raise ValueError("At least one supplemental catenary mast failed point-support gates")

    registry = copy.deepcopy(load_json(source_registry_file))
    for asset in registry.get("assets", []):
        if isinstance(asset.get("geometry"), dict):
            asset["geometry"]["file"] = str(output_obj)
    added_assets: list[dict[str, Any]] = []
    for record in records:
        fit = record["fit"]
        asset = {
            "id": record["asset_id"],
            "type": "catenary_mast",
            "subtype": "H_section_candidate",
            "status": "candidate",
            "chainage_m": fit["station_m"],
            "evidence_level": "photo_interpreted",
            "confidence": record["confidence"],
            "sources": [
                {"kind": "point_cloud", "reference": str(cloud)},
                {"kind": "panorama", "reference": str(record["evidence"][-1])},
                {"kind": "manual_review", "reference": str(disposition_file)},
            ],
            "parameters": {
                "source_candidate_id": record["candidate_id"],
                "longitudinal_position_m": fit["station_m"],
                "cross_position_m": fit["cross_m"],
                "height_m": fit["height_m"],
                "along_width_m": fit["along_width_m"],
                "cross_depth_m": fit["cross_depth_m"],
                "geometry_method": (
                    "multi_view_photo_interpreted_mast_with_disconnected_point_band_alignment"
                ),
            },
            "geometry": {"file": str(output_obj), "node": record["asset_id"]},
                "limitations": [
                    "The outer section envelope is observed; the exact catalogue profile is unresolved.",
                    "The middle shaft is not continuously sampled and is bridged from multi-view photo evidence.",
                    "Cantilever and insulator geometry are excluded until separately supported by evidence.",
            ],
        }
        added_assets.append(asset)
        registry["assets"].append(asset)
    registry["release_id"] = output_name
    registry["updated_at"] = datetime.now(UTC).isoformat()
    registry["summary"] = summarize_registry(registry)
    inherited_errors = validate_registry_value(load_json(source_registry_file))
    addition_registry = new_registry(str(registry.get("project_id", "site-b")))
    addition_registry["assets"] = copy.deepcopy(added_assets)
    addition_registry["summary"] = summarize_registry(addition_registry)
    addition_errors = validate_registry_value(addition_registry)
    if addition_errors:
        raise ValueError("Supplemental mast additions are invalid: " + "; ".join(addition_errors))
    output_registry = output / "asset_registry.json"
    write_json(output_registry, registry)

    mesh_audit = audit_obj(output_obj)
    output_mesh_audit = output / "mesh_audit.json"
    write_json(output_mesh_audit, mesh_audit)
    if not mesh_audit["passed"]:
        raise ValueError("Supplemental mast candidate failed mesh audit")
    output_report = output / "supplemental_gap_mast_report.json"
    write_json(
        output_report,
        {
            "schema_version": "railway.supplemental-gap-masts.v1",
            "source_obj": str(source_model),
            "output_obj": str(output_obj),
            "records": records,
            "merge": merge,
            "added_asset_count": len(records),
            "new_assets_schema_valid": True,
            "inherited_registry_validation_error_count": len(inherited_errors),
            "mesh_audit_passed": True,
            "output_obj_sha256": sha256_file(output_obj),
            "status": (
                "photo_interpreted_candidate_built_point_band_support_passed_"
                "fixed_view_and_clearance_review_required"
            ),
        },
    )
    return {
        "obj": output_obj,
        "mtl": output_mtl,
        "origin": output_origin,
        "registry": output_registry,
        "mesh_audit": output_mesh_audit,
        "report": output_report,
    }
