from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from .algorithms.mesh import ObjWriter, oriented_box
from .catenary_candidate_mesh import h_section_prism
from .config import ProjectConfig
from .io import load_json, sha256_file, write_json
from .mesh_audit import audit_obj
from .registry import new_registry, summarize_registry, validate_registry_value


def _resource_settings() -> dict[str, Any]:
    path = resources.files("railway_recon.resources").joinpath(
        "corridor-catenary-pipeline.default.json"
    )
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _candidate_checks(candidate: dict[str, Any], settings: dict[str, Any]) -> dict[str, bool]:
    features = candidate.get("features", {})
    cable = candidate.get("cable_context", {})
    return {
        "height_range": float(settings["minimum_height_m"])
        <= float(candidate.get("height_m", 0.0))
        <= float(settings["maximum_height_m"]),
        "footprint_range": float(settings["minimum_footprint_m"])
        <= float(candidate.get("footprint_m", 0.0))
        <= float(settings["maximum_footprint_m"]),
        "vertical_axis": float(features.get("vertical_axis_z", 0.0))
        >= float(settings["minimum_vertical_axis_z"]),
        "vertical_occupancy": float(features.get("vertical_occupied_ratio", 0.0))
        >= float(settings["minimum_vertical_occupied_ratio"]),
        "linearity": float(features.get("linearity", 0.0))
        >= float(settings["minimum_linearity"]),
        "cable_context": bool(cable.get("available", False)),
        "cable_alignment": bool(cable.get("available", False))
        and float(cable.get("cross_difference_m", float("inf")))
        <= float(settings["maximum_cable_cross_difference_m"]),
    }


def _candidate_quality(candidate: dict[str, Any]) -> float:
    features = candidate.get("features", {})
    return float(
        0.45 * float(features.get("vertical_axis_z", 0.0))
        + 0.35 * float(features.get("linearity", 0.0))
        + 0.20 * float(features.get("vertical_occupied_ratio", 0.0))
    )


def _track_clearance_checks(
    candidate: dict[str, Any], settings: dict[str, Any]
) -> dict[str, bool]:
    required_settings = {
        "minimum_mast_distance_from_nearest_rail_m",
        "maximum_mast_distance_from_nearest_rail_m",
        "minimum_renderable_mast_height_m",
    }
    if not required_settings.issubset(settings):
        return {
            "configured": False,
            "rail_context_available": True,
            "nearest_rail_distance": True,
            "renderable_height": True,
        }
    rail = candidate.get("rail_context", {})
    available = bool(rail.get("available", False))
    distance = float(rail.get("nearest_rail_distance_m", float("inf")))
    return {
        "configured": True,
        "rail_context_available": available,
        "nearest_rail_distance": available
        and float(settings["minimum_mast_distance_from_nearest_rail_m"])
        <= distance
        <= float(settings["maximum_mast_distance_from_nearest_rail_m"]),
        "renderable_height": float(candidate.get("height_m", 0.0))
        >= float(settings["minimum_renderable_mast_height_m"]),
    }


def _clearance_checks_passed(checks: dict[str, bool]) -> bool:
    return all(value for key, value in checks.items() if key != "configured")


def _merge_owned_candidates(
    candidates: list[dict[str, Any]], settings: dict[str, Any]
) -> list[dict[str, Any]]:
    station_tolerance = float(settings["duplicate_station_tolerance_m"])
    xy_tolerance = float(settings["duplicate_xy_tolerance_m"])
    pending = sorted(candidates, key=lambda item: float(item["chainage_m"]))
    clusters: list[list[dict[str, Any]]] = []
    for candidate in pending:
        match: list[dict[str, Any]] | None = None
        for cluster in reversed(clusters):
            anchor = max(cluster, key=_candidate_quality)
            delta_s = float(candidate["chainage_m"]) - float(anchor["chainage_m"])
            if delta_s > station_tolerance:
                break
            distance = float(
                np.hypot(
                    float(candidate["center_x"]) - float(anchor["center_x"]),
                    float(candidate["center_y"]) - float(anchor["center_y"]),
                )
            )
            if abs(delta_s) <= station_tolerance and distance <= xy_tolerance:
                match = cluster
                break
        if match is None:
            clusters.append([candidate])
        else:
            match.append(candidate)
    merged: list[dict[str, Any]] = []
    for cluster in clusters:
        representative = dict(max(cluster, key=_candidate_quality))
        representative["source_candidates"] = [
            {
                "segment_id": str(item["segment_id"]),
                "candidate_id": str(item["candidate_id"]),
                "report": str(item["report"]),
            }
            for item in cluster
        ]
        representative["duplicate_candidate_count"] = len(cluster)
        representative["quality_score"] = _candidate_quality(representative)
        merged.append(representative)
    return sorted(merged, key=lambda item: float(item["chainage_m"]))


def _periodic_components(
    candidates: list[dict[str, Any]], settings: dict[str, Any]
) -> list[list[int]]:
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    minimum_spacing = float(settings["minimum_periodic_spacing_m"])
    maximum_spacing = float(settings["maximum_periodic_spacing_m"])
    maximum_cross_change = float(settings["maximum_periodic_cross_change_m"])
    for left, first in enumerate(candidates):
        for right in range(left + 1, len(candidates)):
            second = candidates[right]
            spacing = float(second["chainage_m"]) - float(first["chainage_m"])
            if spacing > maximum_spacing:
                break
            if (
                spacing >= minimum_spacing
                and abs(float(second["cross_position_m"]) - float(first["cross_position_m"]))
                <= maximum_cross_change
            ):
                union(left, right)
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(candidates)):
        grouped[find(index)].append(index)
    return sorted(grouped.values(), key=lambda indexes: candidates[indexes[0]]["chainage_m"])


def select_corridor_catenary_supports(
    manifest: dict[str, Any],
    reports: dict[str, dict[str, Any]],
    report_paths: dict[str, str],
    settings: dict[str, Any],
) -> dict[str, Any]:
    segments = {str(item["id"]): item for item in manifest.get("segments", [])}
    decisions: list[dict[str, Any]] = []
    owned_candidates: list[dict[str, Any]] = []
    missing_reports = sorted(set(segments) - set(reports))
    for segment_id, report in reports.items():
        if segment_id not in segments:
            raise ValueError(f"Vertical report is not in the segment manifest: {segment_id}")
        segment = segments[segment_id]
        start = float(segment["chainage_start_m"])
        end = float(segment["chainage_end_m"])
        length = end - start
        frame = report["frame"]
        for candidate in report.get("candidates", []):
            local_station = float(candidate.get("longitudinal_position_m", float("nan")))
            owned = 0.0 <= local_station < length or (
                end == max(float(item["chainage_end_m"]) for item in segments.values())
                and local_station == length
            )
            checks = _candidate_checks(candidate, settings)
            accepted = owned and all(checks.values())
            decision = {
                "segment_id": segment_id,
                "candidate_id": str(candidate.get("id")),
                "local_station_m": local_station,
                "chainage_m": start + local_station,
                "owned_by_segment": owned,
                "checks": checks,
                "accepted_as_geometric_seed": accepted,
                "original_predicted_class": candidate.get("predicted_class"),
            }
            decisions.append(decision)
            if not accepted:
                continue
            owned_candidates.append(
                {
                    **candidate,
                    "candidate_id": str(candidate["id"]),
                    "segment_id": segment_id,
                    "report": report_paths[segment_id],
                    "chainage_m": start + local_station,
                    "along_xy": [float(value) for value in frame["along_xy"]],
                    "cross_xy": [float(value) for value in frame["cross_xy"]],
                }
            )
    merged = _merge_owned_candidates(owned_candidates, settings)
    components = _periodic_components(merged, settings)
    minimum_run = int(settings["minimum_periodic_run_count"])
    periodic_accepted_indexes = {
        index for component in components if len(component) >= minimum_run for index in component
    }
    run_records: list[dict[str, Any]] = []
    for run_index, component in enumerate(components, start=1):
        members = [merged[index] for index in component]
        spacings = np.diff([float(item["chainage_m"]) for item in members])
        run_records.append(
            {
                "id": f"CATENARY-RUN-{run_index:03d}",
                "candidate_indexes": component,
                "candidate_count": len(component),
                "chainage_start_m": float(members[0]["chainage_m"]),
                "chainage_end_m": float(members[-1]["chainage_m"]),
                "median_spacing_m": float(np.median(spacings)) if len(spacings) else None,
                "accepted": len(component) >= minimum_run,
            }
        )
    selected: list[dict[str, Any]] = []
    for index, candidate in enumerate(merged):
        record = dict(candidate)
        record["periodic_run_id"] = next(
            item["id"] for item in run_records if index in item["candidate_indexes"]
        )
        clearance_checks = _track_clearance_checks(candidate, settings)
        periodic_gate_passed = index in periodic_accepted_indexes
        clearance_gate_passed = _clearance_checks_passed(clearance_checks)
        record["periodic_gate_passed"] = periodic_gate_passed
        record["track_clearance_checks"] = clearance_checks
        record["track_clearance_gate_passed"] = clearance_gate_passed
        record["accepted_for_candidate_mesh"] = (
            periodic_gate_passed and clearance_gate_passed
        )
        record["candidate_mesh_decision"] = (
            "accepted_periodic_and_track_clearance"
            if record["accepted_for_candidate_mesh"]
            else (
                "withheld_track_clearance_conflict"
                if periodic_gate_passed
                else "withheld_nonperiodic"
            )
        )
        selected.append(record)
    accepted_support_count = sum(
        bool(item["accepted_for_candidate_mesh"]) for item in selected
    )
    withheld_by_clearance_count = sum(
        bool(item["periodic_gate_passed"])
        and not bool(item["track_clearance_gate_passed"])
        for item in selected
    )
    return {
        "schema_version": "railway.corridor-catenary-selection.v1",
        "segment_count": len(segments),
        "vertical_report_count": len(reports),
        "missing_vertical_report_segment_ids": missing_reports,
        "raw_candidate_count": len(decisions),
        "geometric_seed_count": len(owned_candidates),
        "deduplicated_seed_count": len(merged),
        "periodic_run_count": sum(item["accepted"] for item in run_records),
        "periodic_support_count": len(periodic_accepted_indexes),
        "accepted_support_count": accepted_support_count,
        "withheld_by_clearance_count": withheld_by_clearance_count,
        "withheld_seed_count": len(merged) - accepted_support_count,
        "decisions": decisions,
        "runs": run_records,
        "candidates": selected,
        "status": "periodic_point_supported_catenary_candidates_selected",
        "limitations": [
            "Original per-segment semantic predictions are retained but do not override corridor periodicity evidence.",
            "Periodic seeds must also pass a configured nearest-rail distance and renderable-height gate before mesh generation.",
            "No missing mast position is inferred into renderable geometry.",
            "Automatic candidates remain non-final until photo or owner review.",
        ],
    }


def _write_materials(path: Path) -> None:
    content = """# Corridor catenary candidate materials
newmtl CatenaryMastCandidate
Kd 0.15 0.58 0.70
Ks 0.20 0.20 0.20
Ns 35

newmtl CatenaryFoundationCandidate
Kd 0.45 0.46 0.43
Ks 0.04 0.04 0.04
Ns 5
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def run_corridor_catenary_pipeline(
    project: ProjectConfig,
    output_dir_value: str | Path,
    output_name: str,
    *,
    settings_value: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not output_name or Path(output_name).name != output_name:
        raise ValueError("output_name must be one safe path component")
    settings_path = project.resolve(settings_value) if settings_value else None
    settings = load_json(settings_path) if settings_path else _resource_settings()
    if settings.get("schema_version") != "railway.corridor-catenary-settings.v1":
        raise ValueError("Unsupported corridor-catenary settings")
    manifest_path = project.workspace_path("segment_manifest")
    manifest = load_json(manifest_path)
    reports: dict[str, dict[str, Any]] = {}
    report_paths: dict[str, str] = {}
    for segment in manifest.get("segments", []):
        segment_id = str(segment["id"])
        path = project.workspace_path("reports") / f"{segment_id}_vertical_hypotheses.json"
        if path.is_file():
            reports[segment_id] = load_json(path)
            report_paths[segment_id] = str(path)
    selection = select_corridor_catenary_supports(
        manifest, reports, report_paths, settings
    )
    accepted = [
        item for item in selection["candidates"] if item["accepted_for_candidate_mesh"]
    ]
    if not accepted:
        raise ValueError("No periodic point-supported catenary candidates passed")
    output_dir = project.resolve(output_dir_value)
    obj_path = output_dir / f"{output_name}.obj"
    mtl_path = output_dir / f"{output_name}.mtl"
    origin_path = output_dir / "model_origin.json"
    audit_path = output_dir / "mesh_audit.json"
    registry_path = output_dir / "asset_registry.json"
    selection_path = output_dir / "selection_report.json"
    report_path = output_dir / "pipeline_report.json"
    outputs = (
        obj_path,
        mtl_path,
        origin_path,
        audit_path,
        registry_path,
        selection_path,
        report_path,
    )
    if not overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite corridor catenary outputs: {existing}")
    first = accepted[0]
    origin = np.asarray(
        [float(first["center_x"]), float(first["center_y"]), 0.0],
        dtype=np.float64,
    )
    writer = ObjWriter(origin, material_library=mtl_path.name)
    registry = new_registry(project.project_id)
    for sequence, candidate in enumerate(accepted, start=1):
        mast_id = f"CATENARY-MAST-AUTO-{sequence:04d}"
        foundation_id = f"{mast_id}-FOUNDATION"
        center_xy = np.asarray(
            [float(candidate["center_x"]), float(candidate["center_y"])],
            dtype=np.float64,
        )
        along = np.asarray(candidate["along_xy"], dtype=np.float64)
        cross = np.asarray(candidate["cross_xy"], dtype=np.float64)
        base_z = float(candidate["minimum_z"])
        top_z = float(candidate["maximum_z"])
        foundation_top = min(
            base_z + float(settings["foundation_height_m"]), top_z - 0.5
        )
        foundation_vertices, foundation_faces = oriented_box(
            np.asarray([center_xy[0], center_xy[1], base_z]),
            np.asarray([along[0], along[1], 0.0]),
            float(settings["foundation_along_m"]),
            float(settings["foundation_cross_m"]),
            base_z,
            foundation_top,
        )
        writer.add_mesh(
            foundation_id,
            foundation_vertices,
            foundation_faces,
            "CatenaryFoundationCandidate",
        )
        mast_vertices, mast_faces = h_section_prism(
            center_xy,
            along,
            cross,
            float(settings["mast_along_width_m"]),
            float(settings["mast_cross_depth_m"]),
            float(settings["mast_flange_thickness_m"]),
            float(settings["mast_web_thickness_m"]),
            foundation_top - 0.08,
            top_z,
        )
        writer.add_mesh(
            mast_id, mast_vertices, mast_faces, "CatenaryMastCandidate"
        )
        confidence = min(
            float(settings["maximum_automatic_confidence"]),
            max(0.0, float(candidate["quality_score"])),
        )
        sources = [
            {
                "kind": "point_cloud",
                "reference": f"{item['report']}#{item['candidate_id']}",
            }
            for item in candidate["source_candidates"]
        ] + [
            {
                "kind": "rule",
                "reference": f"{selection_path}#{candidate['periodic_run_id']}",
                "note": "Corridor periodicity gate; no missing position was synthesized",
            }
        ]
        common = {
            "status": "candidate",
            "evidence_level": "rule_inferred",
            "confidence": confidence,
            "chainage_m": float(candidate["chainage_m"]),
            "sources": sources,
        }
        registry["assets"].extend(
            [
                {
                    "id": foundation_id,
                    "type": "catenary_foundation",
                    **common,
                    "parameters": {
                        "source_candidate_ids": candidate["source_candidates"],
                        "height_m": foundation_top - base_z,
                    },
                    "geometry": {"file": str(obj_path), "node": foundation_id},
                    "limitations": [
                        "Foundation dimensions are conservative display envelopes."
                    ],
                },
                {
                    "id": mast_id,
                    "type": "catenary_mast",
                    "subtype": "H_section_candidate",
                    **common,
                    "parameters": {
                        "source_candidate_ids": candidate["source_candidates"],
                        "periodic_run_id": candidate["periodic_run_id"],
                        "height_m": top_z - (foundation_top - 0.08),
                        "profile": "configured_H_section_candidate",
                    },
                    "geometry": {"file": str(obj_path), "node": mast_id},
                    "limitations": [
                        "Exact mast product section is not identified.",
                        "Cantilever, insulators and conductors are not generated in this stage.",
                    ],
                },
            ]
        )
        registry["relations"].append(
            {
                "id": f"REL-{foundation_id}-SUPPORTS-{mast_id}",
                "type": "supports",
                "from": foundation_id,
                "to": mast_id,
            }
        )
    writer.write(obj_path)
    _write_materials(mtl_path)
    write_json(
        origin_path,
        {"origin_xyz": origin.tolist(), "units": "metre", "axis": "Z-up"},
    )
    write_json(selection_path, selection)
    audit = audit_obj(obj_path)
    audit["status"] = "pass" if audit["passed"] else "fail"
    write_json(audit_path, audit)
    if not audit["passed"]:
        raise ValueError("Corridor catenary candidate OBJ failed mesh audit")
    registry["updated_at"] = datetime.now(UTC).isoformat()
    registry["summary"] = summarize_registry(registry)
    errors = validate_registry_value(registry)
    if errors:
        raise ValueError("Corridor catenary registry is invalid: " + "; ".join(errors))
    write_json(registry_path, registry)
    result = {
        "schema_version": "railway.corridor-catenary-pipeline.v1",
        "project_id": project.project_id,
        "status": "candidate_mesh_written_review_required_not_formal_release",
        "formal_release": False,
        "canonical_registry_updated": False,
        "segment_manifest": str(manifest_path),
        "settings_source": str(settings_path) if settings_path else "bundled_default",
        "output_obj": str(obj_path),
        "output_mtl": str(mtl_path),
        "output_origin": str(origin_path),
        "output_mesh_audit": str(audit_path),
        "output_asset_registry": str(registry_path),
        "selection_report": str(selection_path),
        "output_obj_sha256": sha256_file(obj_path),
        "mast_count": len(accepted),
        "asset_count": len(registry["assets"]),
        "relation_count": len(registry["relations"]),
        "selection_summary": {
            key: selection[key]
            for key in (
                "vertical_report_count",
                "missing_vertical_report_segment_ids",
                "raw_candidate_count",
                "geometric_seed_count",
                "deduplicated_seed_count",
                "periodic_run_count",
                "periodic_support_count",
                "accepted_support_count",
                "withheld_by_clearance_count",
                "withheld_seed_count",
            )
        },
        "limitations": selection["limitations"]
        + [
            "Only mast shafts and foundation candidates are generated.",
            "This stage is designed to prevent per-segment duplicate poles and semantic drift.",
        ],
    }
    write_json(report_path, result)
    return result
