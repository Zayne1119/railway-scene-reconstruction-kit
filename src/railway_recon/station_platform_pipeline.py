from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .io import load_json, write_json
from .mesh_assembly import assemble_candidate_meshes
from .platform_candidate_gate import evaluate_platform_candidate_gate
from .platform_gap_point_evidence import analyze_platform_gap_point_evidence
from .platform_mesh import build_platform_mesh
from .segment_mesh_ownership import (
    audit_owned_mesh_seams,
    build_segment_ownership_plan,
    clip_segment_mesh_ownership,
)


def _contiguous_runs(order: list[str], accepted: set[str]) -> list[list[str]]:
    runs: list[list[str]] = []
    current: list[str] = []
    for segment_id in order:
        if segment_id in accepted:
            current.append(segment_id)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def run_station_platform_pipeline(
    project: ProjectConfig,
    ownership_plan_value: str | Path,
    segment_ids: list[str],
    output_name: str,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build every safe platform segment and isolate blocked intervals automatically."""

    if len(segment_ids) < 2:
        raise ValueError("Station platform pipeline needs at least two segments")
    if len(set(segment_ids)) != len(segment_ids):
        raise ValueError("Station platform segment IDs must be unique")
    if not output_name or Path(output_name).name != output_name:
        raise ValueError("output_name must be one safe path component")
    ownership_path = project.resolve(ownership_plan_value)
    ownership = load_json(ownership_path)
    entries = {str(item["segment_id"]): item for item in ownership["segments"]}
    missing = sorted(set(segment_ids) - set(entries))
    if missing:
        raise ValueError(f"Segments missing from ownership plan: {missing}")
    ordered = [str(item["segment_id"]) for item in ownership["segments"]]
    selected_order = [segment_id for segment_id in ordered if segment_id in set(segment_ids)]
    if selected_order != segment_ids:
        raise ValueError("Segments must follow ownership-plan order")

    reports = project.workspace_path("reports")
    exports = project.workspace_path("exports")
    run_reports = reports / output_name
    run_reports.mkdir(parents=True, exist_ok=True)
    segment_results: list[dict[str, Any]] = []
    accepted_meshes: dict[str, Path] = {}
    for segment_id in segment_ids:
        platform_path = reports / f"{segment_id}_platform_surface.json"
        if not platform_path.is_file():
            segment_results.append(
                {
                    "segment_id": segment_id,
                    "passed": False,
                    "reason": "missing_platform_surface_report",
                }
            )
            continue
        platform = load_json(platform_path)
        if len(platform.get("platform_components", [])) != 1:
            segment_results.append(
                {
                    "segment_id": segment_id,
                    "passed": False,
                    "reason": "platform_component_count_not_one",
                    "component_count": len(platform.get("platform_components", [])),
                }
            )
            continue
        gap_path = run_reports / f"{segment_id}_gap_point_evidence.json"
        analyze_platform_gap_point_evidence(
            project,
            segment_id,
            platform_path,
            gap_path,
            overwrite=overwrite,
        )
        gate_path = run_reports / f"{segment_id}_platform_gate.json"
        gate = evaluate_platform_candidate_gate(
            project,
            segment_id,
            platform_path,
            gate_path,
            ownership_plan_value=ownership_path,
            gap_point_evidence_value=gap_path,
            overwrite=overwrite,
        )
        record = {
            "segment_id": segment_id,
            "passed": bool(gate["passed"]),
            "gate": str(gate_path),
            "gap_point_evidence": str(gap_path),
            "unresolved_gap_count": int(gate["unresolved_gap_count"]),
            "suppressed_candidate_gap_count": int(gate["suppressed_candidate_gap_count"]),
            "maximum_fit_residual_p90_m": float(gate["maximum_fit_residual_p90_m"]),
            "minimum_fit_inlier_fraction": float(gate["minimum_fit_inlier_fraction"]),
        }
        if not gate["passed"]:
            record["reason"] = "automatic_candidate_gate_blocked"
            record["failed_checks"] = [
                name for name, passed in gate["checks"].items() if not passed
            ]
            segment_results.append(record)
            continue
        mesh = build_platform_mesh(
            project,
            segment_id,
            gate_path,
            overwrite=overwrite,
            platform_report_path=platform_path,
        )
        entry = entries[segment_id]
        owned_dir = exports / segment_id / f"{output_name}_owned"
        owned = clip_segment_mesh_ownership(
            project,
            mesh["output_obj"],
            mesh["output_origin"],
            ownership_path,
            owned_dir,
            minimum_longitudinal_m=float(entry["owned_interval_m"][0]),
            maximum_longitudinal_m=float(entry["owned_interval_m"][1]),
            overwrite=overwrite,
        )
        accepted_meshes[segment_id] = Path(owned["output_obj"])
        record.update(
            {
                "mesh_build": mesh["output_obj"],
                "owned_mesh": owned["output_obj"],
                "owned_mesh_audit": owned["output_mesh_audit"],
            }
        )
        segment_results.append(record)

    runs = _contiguous_runs(segment_ids, set(accepted_meshes))
    run_results = []
    seam_limit = float(project.value["quality"]["platform_fit_p90_max_m"])
    for index, run in enumerate(runs, start=1):
        run_id = f"{output_name}_run{index:02d}"
        if len(run) == 1:
            run_results.append(
                {
                    "run_id": run_id,
                    "segments": run,
                    "status": "single_owned_mesh_no_cross_segment_assembly_needed",
                    "output_obj": str(accepted_meshes[run[0]]),
                }
            )
            continue
        subset_plan_path = run_reports / f"{run_id}_ownership.json"
        subset_plan = build_segment_ownership_plan(
            project,
            {segment_id: entries[segment_id]["frame_report"] for segment_id in run},
            subset_plan_path,
            overwrite=overwrite,
        )
        seam_path = run_reports / f"{run_id}_seams.json"
        seam = audit_owned_mesh_seams(
            project,
            subset_plan_path,
            {segment_id: accepted_meshes[segment_id] for segment_id in run},
            seam_path,
            p90_max_m=seam_limit,
            boundary_tolerance_m=0.005,
            sample_spacing_m=0.02,
            overwrite=overwrite,
        )
        if not seam["passed"]:
            run_results.append(
                {
                    "run_id": run_id,
                    "segments": run,
                    "status": "blocked_cross_segment_seam_audit",
                    "seam_audit": str(seam_path),
                }
            )
            continue
        assembly = assemble_candidate_meshes(
            project,
            exports / run_id,
            run_id,
            [(segment_id.upper(), accepted_meshes[segment_id]) for segment_id in run],
            overwrite=overwrite,
        )
        run_results.append(
            {
                "run_id": run_id,
                "segments": run,
                "ownership_plan": str(subset_plan_path),
                "ownership_coverage_length_m": float(subset_plan["coverage_length_m"]),
                "seam_audit": str(seam_path),
                "worst_seam_p90_m": max(
                    float(item["symmetric_nearest_edge_distance"]["p90_m"])
                    for item in seam["seams"]
                ),
                "output_obj": assembly["output_obj"],
                "output_mesh_audit": assembly["output_mesh_audit"],
                "status": "continuous_candidate_platform_assembly_written",
            }
        )

    result = {
        "schema_version": "railway.station-platform-pipeline.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "project_id": project.project_id,
        "output_name": output_name,
        "ownership_plan": str(ownership_path),
        "requested_segment_count": len(segment_ids),
        "passed_segment_count": len(accepted_meshes),
        "blocked_segment_count": len(segment_ids) - len(accepted_meshes),
        "segment_results": segment_results,
        "continuous_runs": run_results,
        "status": (
            "candidate_platform_pipeline_complete"
            if len(accepted_meshes) == len(segment_ids)
            else "candidate_platform_pipeline_partial_blocked_segments_reported"
        ),
        "formal_release": False,
    }
    output_report = run_reports / "pipeline_report.json"
    write_json(output_report, result)
    result["output_report"] = str(output_report)
    return result
