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
from .supplemental_boundary_conductors import (
    _evaluate_fit,
    _local_to_world,
    fit_boundary_conductor_fragment,
)
from .supplemental_conductor_refinement import _load_candidate_points
from .targeted_canopy_integration import merge_candidate_objs


def match_next_boundary_conductors(
    current_fit_records: list[dict[str, Any]],
    adjacent_candidates: list[dict[str, Any]],
    *,
    seam_station_m: float,
    maximum_cross_z_distance_m: float = 0.75,
) -> list[dict[str, Any]]:
    """Match renumbered residuals to the already-built current-side fragments."""
    current: list[dict[str, Any]] = []
    for record in current_fit_records:
        endpoint = _evaluate_fit(
            record["fit"], np.asarray([seam_station_m], dtype=np.float64)
        )[0]
        current.append({"record": record, "endpoint": endpoint})
    candidates = [
        item
        for item in adjacent_candidates
        if float(item["station_range_m"][0]) >= seam_station_m
    ]
    if len(current) < len(candidates):
        raise ValueError("Too few current-side conductor fragments for boundary matching")

    scored: list[tuple[float, int, int]] = []
    for candidate_index, candidate in enumerate(candidates):
        target_cross = float(np.mean(candidate["cross_range_m"]))
        target_z = float(np.mean(candidate["z_range_m"]))
        for current_index, item in enumerate(current):
            endpoint = item["endpoint"]
            distance = float(np.hypot(endpoint[1] - target_cross, endpoint[2] - target_z))
            scored.append((distance, candidate_index, current_index))
    matched_candidates: set[int] = set()
    matched_current: set[int] = set()
    matches: list[dict[str, Any]] = []
    for distance, candidate_index, current_index in sorted(scored):
        if candidate_index in matched_candidates or current_index in matched_current:
            continue
        if distance > maximum_cross_z_distance_m:
            continue
        matched_candidates.add(candidate_index)
        matched_current.add(current_index)
        matches.append(
            {
                "candidate": candidates[candidate_index],
                "current_record": current[current_index]["record"],
                "current_endpoint_station_cross_z": current[current_index][
                    "endpoint"
                ].tolist(),
                "cross_z_match_distance_m": distance,
            }
        )
    if len(matches) != len(candidates):
        missing = sorted(
            str(candidates[index]["candidate_id"])
            for index in range(len(candidates))
            if index not in matched_candidates
        )
        raise ValueError(f"Unmatched next-boundary conductor residuals: {missing}")
    return sorted(matches, key=lambda item: str(item["candidate"]["candidate_id"]))


def continuation_centerline(
    fit: dict[str, Any],
    *,
    seam_station_m: float,
    observed_start_m: float,
    build_end_m: float,
    seam_endpoint_station_cross_z: list[float],
    seam_clearance_m: float = 0.002,
    blend_length_m: float = 0.75,
    sample_spacing_m: float = 0.20,
) -> np.ndarray:
    """Create a continuous handoff while retaining separate non-overlapping mesh caps."""
    if build_end_m <= observed_start_m or observed_start_m < seam_station_m:
        raise ValueError("Invalid next-boundary conductor interval")
    start = seam_station_m + seam_clearance_m
    if build_end_m <= start:
        raise ValueError("Boundary continuation is shorter than the seam clearance")
    count = max(3, int(np.ceil((build_end_m - start) / sample_spacing_m)) + 1)
    station = np.linspace(start, build_end_m, count)
    local = _evaluate_fit(fit, station)
    blend_end = min(build_end_m, max(observed_start_m, seam_station_m) + blend_length_m)
    seam_endpoint = np.asarray(seam_endpoint_station_cross_z, dtype=np.float64)
    target = _evaluate_fit(fit, np.asarray([blend_end], dtype=np.float64))[0]
    blend = station <= blend_end
    amount = np.clip(
        (station[blend] - seam_station_m) / max(blend_end - seam_station_m, 1.0e-9),
        0.0,
        1.0,
    )
    local[blend, 1:] = (
        seam_endpoint[1:][None, :] * (1.0 - amount[:, None])
        + target[1:][None, :] * amount[:, None]
    )
    return local


def _write_material(path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "# Adjacent-segment observed conductor handoff\n"
        "newmtl AdjacentBoundaryConductor\n"
        "Kd 0.98 0.42 0.08\nKs 0.45 0.45 0.45\nNs 60\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def build_adjacent_boundary_conductors(
    *,
    cloud_path: str | Path,
    gap_report_path: str | Path,
    frame_report_path: str | Path,
    current_boundary_report_path: str | Path,
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
    scope = gap["corridor_scope"]
    scope_min = float(scope["station_minimum_m"])
    scope_max = float(scope["station_maximum_m"])
    candidates = [
        item
        for item in gap["overhead_linear_candidates"]
        if item.get("asset_relation") == "unmatched_overhead_linear_candidate"
        and item.get("priority") in {"P0", "P1"}
        and item.get("corridor_scope_ownership")
        in {"crosses_segment_boundary", "outside_current_segment"}
    ]
    next_candidates = [
        item for item in candidates if float(item["station_range_m"][1]) > scope_max
    ]
    previous_candidates = [
        item for item in candidates if float(item["station_range_m"][1]) < scope_min
    ]
    if len(next_candidates) != 3 or len(previous_candidates) != 1:
        raise ValueError(
            "Expected three next-boundary and one previous-boundary conductor residual"
        )

    current_report_path = Path(current_boundary_report_path).resolve()
    current_report = load_json(current_report_path)
    matches = match_next_boundary_conductors(
        current_report["fit_records"],
        next_candidates,
        seam_station_m=scope_max,
    )
    cloud = Path(cloud_path).resolve()
    points_by_id = _load_candidate_points(
        cloud,
        candidates,
        frame,
        cross_margin_m=0.12,
        z_margin_m=0.12,
    )
    origin_value = load_json(Path(source_origin))
    model_origin = np.asarray(origin_value["origin_xyz"], dtype=np.float64)
    component_obj = output / "adjacent_boundary_conductors.obj"
    component_mtl = output / "adjacent_boundary_conductors.mtl"
    component_origin = output / "adjacent_boundary_conductors_origin.json"
    writer = ObjWriter(model_origin, material_library=component_mtl.name)
    records: list[dict[str, Any]] = []
    additions: list[dict[str, Any]] = []

    for sequence, match in enumerate(matches, start=1):
        candidate = match["candidate"]
        candidate_id = str(candidate["candidate_id"])
        observed_start = float(candidate["station_range_m"][0])
        observed_end = float(candidate["station_range_m"][1])
        fit = fit_boundary_conductor_fragment(
            points_by_id[candidate_id],
            station_start_m=observed_start,
            station_end_m=observed_end,
        )
        if not fit["passed"]:
            raise ValueError(f"{candidate_id} failed adjacent conductor fit gates")
        local = continuation_centerline(
            fit,
            seam_station_m=scope_max,
            observed_start_m=observed_start,
            build_end_m=observed_end,
            seam_endpoint_station_cross_z=match[
                "current_endpoint_station_cross_z"
            ],
        )
        centerline = _local_to_world(local, frame)
        vertices, faces = sweep_mesh(centerline, circular_profile(render_radius_m, 8))
        asset_id = f"ADJACENT-NEXT-BOUNDARY-CONDUCTOR-{sequence:02d}"
        writer.add_mesh(asset_id, vertices, faces, "AdjacentBoundaryConductor")
        seam_gap = float(local[0, 0] - scope_max)
        record = {
            "candidate_id": candidate_id,
            "asset_id": asset_id,
            "ownership": "adjacent_next_segment",
            "source_current_candidate_id": match["current_record"]["candidate_id"],
            "cross_z_match_distance_m": match["cross_z_match_distance_m"],
            "observed_station_range_m": [observed_start, observed_end],
            "build_station_range_m": [float(local[0, 0]), float(local[-1, 0])],
            "seam_centerline_clearance_m": seam_gap,
            "unsupported_seam_bridge_m": max(0.0, observed_start - scope_max),
            "fit": fit,
        }
        records.append(record)
        additions.append(
            {
                "id": asset_id,
                "type": "catenary_auxiliary_conductor",
                "subtype": "adjacent_next_observed_boundary_continuation",
                "status": "candidate",
                "chainage_m": float(np.mean(record["build_station_range_m"])),
                "evidence_level": "observed",
                "confidence": 0.86,
                "sources": [
                    {"kind": "point_cloud", "reference": str(cloud)},
                    {"kind": "rule", "reference": str(gap_path)},
                ],
                "parameters": {
                    "source_candidate_id": candidate_id,
                    "segment_ownership": "adjacent_next_segment",
                    "longitudinal_interval_m": record["build_station_range_m"],
                    "fit_residual_p90_m": fit["residual_p90_m"],
                    "seam_centerline_clearance_m": seam_gap,
                    "unsupported_seam_bridge_m": record["unsupported_seam_bridge_m"],
                    "render_radius_m": render_radius_m,
                },
                "geometry": {"file": "", "node": asset_id},
                "limitations": [
                    "Professional conductor subtype remains pending.",
                    "A sub-0.30 m boundary evidence gap is interpolated to the current-side endpoint.",
                    "The 2 mm cap clearance prevents coplanar cap flicker in downstream renderers.",
                ],
            }
        )

    previous = previous_candidates[0]
    previous_id = str(previous["candidate_id"])
    previous_start = float(previous["station_range_m"][0])
    previous_end = float(previous["station_range_m"][1])
    previous_fit = fit_boundary_conductor_fragment(
        points_by_id[previous_id],
        station_start_m=previous_start,
        station_end_m=previous_end,
        minimum_points_per_bin=3,
    )
    if not previous_fit["passed"]:
        raise ValueError(f"{previous_id} failed adjacent conductor fit gates")
    station = np.linspace(
        previous_start,
        previous_end,
        max(3, int(np.ceil((previous_end - previous_start) / 0.20)) + 1),
    )
    previous_local = _evaluate_fit(previous_fit, station)
    previous_centerline = _local_to_world(previous_local, frame)
    vertices, faces = sweep_mesh(
        previous_centerline, circular_profile(render_radius_m, 8)
    )
    previous_asset_id = "ADJACENT-PREVIOUS-BOUNDARY-CONDUCTOR-01"
    writer.add_mesh(
        previous_asset_id, vertices, faces, "AdjacentBoundaryConductor"
    )
    previous_gap = scope_min - previous_end
    records.append(
        {
            "candidate_id": previous_id,
            "asset_id": previous_asset_id,
            "ownership": "adjacent_previous_segment",
            "observed_station_range_m": [previous_start, previous_end],
            "build_station_range_m": [previous_start, previous_end],
            "unbridged_gap_to_current_scope_m": previous_gap,
            "fit": previous_fit,
        }
    )
    additions.append(
        {
            "id": previous_asset_id,
            "type": "catenary_auxiliary_conductor",
            "subtype": "adjacent_previous_observed_fragment",
            "status": "candidate",
            "chainage_m": float(np.mean((previous_start, previous_end))),
            "evidence_level": "observed",
            "confidence": 0.82,
            "sources": [
                {"kind": "point_cloud", "reference": str(cloud)},
                {"kind": "rule", "reference": str(gap_path)},
            ],
            "parameters": {
                "source_candidate_id": previous_id,
                "segment_ownership": "adjacent_previous_segment",
                "longitudinal_interval_m": [previous_start, previous_end],
                "fit_residual_p90_m": previous_fit["residual_p90_m"],
                "unbridged_gap_to_current_scope_m": previous_gap,
                "render_radius_m": render_radius_m,
            },
            "geometry": {"file": "", "node": previous_asset_id},
            "limitations": [
                "Professional conductor subtype remains pending.",
                "No geometry is invented across the unsupported gap to the current model.",
                "Continuation requires the previous adjacent segment point cloud.",
            ],
        }
    )

    writer.write(component_obj)
    _write_material(component_mtl)
    write_json(component_origin, origin_value)
    output_name = output.name
    output_obj = output / f"{output_name}.obj"
    output_mtl = output / f"{output_name}.mtl"
    output_origin = output / "model_origin.json"
    merge = merge_candidate_objs(
        [
            (Path(source_obj).resolve(), Path(source_origin).resolve()),
            (component_obj, component_origin),
        ],
        output_obj,
        output_mtl,
        output_origin,
    )

    registry = copy.deepcopy(load_json(Path(source_registry)))
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
        raise ValueError("Adjacent conductor additions are invalid: " + "; ".join(errors))
    output_registry = output / "asset_registry.json"
    write_json(output_registry, registry)

    mesh = audit_obj(output_obj)
    output_mesh_audit = output / "mesh_audit.json"
    write_json(output_mesh_audit, mesh)
    if not mesh["passed"]:
        raise ValueError("Adjacent boundary conductor candidate failed mesh audit")
    report = output / "adjacent_boundary_conductors_report.json"
    gates = {
        "three_next_continuations_built": len(matches) == 3,
        "one_previous_fragment_built": len(previous_candidates) == 1,
        "next_fit_residuals_below_8cm": all(
            item["fit"]["residual_p90_m"] <= 0.08
            for item in records
            if item["ownership"] == "adjacent_next_segment"
        ),
        "next_seam_cap_clearance_at_most_3mm": all(
            item["seam_centerline_clearance_m"] <= 0.003
            for item in records
            if item["ownership"] == "adjacent_next_segment"
        ),
        "mesh_audit_passed": bool(mesh["passed"]),
    }
    write_json(
        report,
        {
            "schema_version": "railway.adjacent-boundary-conductors.v1",
            "source_obj": str(Path(source_obj).resolve()),
            "output_obj": str(output_obj),
            "gap_report": str(gap_path),
            "current_boundary_report": str(current_report_path),
            "corridor_scope_station_range_m": [scope_min, scope_max],
            "added_asset_count": len(additions),
            "records": records,
            "merge": merge,
            "gates": gates,
            "passed": all(gates.values()),
            "mesh_audit_passed": True,
            "output_obj_sha256": sha256_file(output_obj),
            "status": "adjacent_boundary_candidate_built_not_promoted",
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
