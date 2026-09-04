from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .algorithms.linear_candidates import detect_linear_candidates
from .algorithms.rail_candidates import detect_rail_candidates
from .canopy_structure import analyze_canopy_structure
from .config import ProjectConfig
from .io import load_json, write_json
from .multi_source import input_source_manifest_path, prepare_input_sources
from .platform_surface import analyze_platform_surface
from .segments import crop_segments, plan_segments
from .vertical_hypotheses import analyze_vertical_hypotheses

STAGE_ORDER = ("crop", "rails", "linear", "vertical", "platform", "canopy")


def _stage_output(project: ProjectConfig, segment_id: str, stage: str) -> Path:
    if stage == "crop":
        return project.workspace_path("segments") / f"{segment_id}.laz"
    suffixes = {
        "rails": "rail_candidates",
        "linear": "linear_candidates",
        "vertical": "vertical_hypotheses",
        "platform": "platform_surface",
        "canopy": "canopy_structure",
    }
    return project.workspace_path("reports") / f"{segment_id}_{suffixes[stage]}.json"


def _report_status(path: Path) -> str | None:
    if not path.is_file() or path.suffix.lower() != ".json":
        return None
    try:
        return str(load_json(path).get("status") or "completed")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return "invalid_report"


def _priority(stage: str, state: str, report_status: str | None) -> str:
    if state == "failed" or (state == "missing" and stage in {"crop", "rails"}):
        return "P0"
    if state == "missing":
        return "P1"
    lowered = (report_status or "").lower()
    if "invalid" in lowered or "failed" in lowered:
        return "P0"
    if "review_required" in lowered:
        return "P2"
    return "P3"


def build_full_corridor_plan(project: ProjectConfig) -> dict[str, Any]:
    manifest_path = project.workspace_path("segment_manifest")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Segment manifest not found: {manifest_path}; run prepare-inputs and plan-segments"
        )
    manifest = load_json(manifest_path)
    segments: list[dict[str, Any]] = []
    counts = {stage: {"complete": 0, "missing": 0} for stage in STAGE_ORDER}
    queue: list[dict[str, Any]] = []
    for item in manifest.get("segments", []):
        segment_id = str(item["id"])
        stages: list[dict[str, Any]] = []
        for stage in STAGE_ORDER:
            output = _stage_output(project, segment_id, stage)
            state = "complete" if output.is_file() else "missing"
            status = _report_status(output)
            priority = _priority(stage, state, status)
            counts[stage][state] += 1
            record = {
                "stage": stage,
                "state": state,
                "priority": priority,
                "output": str(output),
                "report_status": status,
            }
            stages.append(record)
            if priority != "P3":
                queue.append(
                    {
                        "priority": priority,
                        "segment_id": segment_id,
                        "stage": stage,
                        "state": state,
                        "report_status": status,
                    }
                )
        segments.append(
            {
                "id": segment_id,
                "chainage_start_m": float(item["chainage_start_m"]),
                "chainage_end_m": float(item["chainage_end_m"]),
                "primary_source_id": str(item.get("primary_source_id", "primary")),
                "context_source_ids": list(item.get("context_source_ids", ["primary"])),
                "stages": stages,
            }
        )
    priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    queue.sort(
        key=lambda value: (
            priority_rank[str(value["priority"])],
            str(value["segment_id"]),
            STAGE_ORDER.index(str(value["stage"])),
        )
    )
    result = {
        "schema_version": "railway.full-corridor-plan.v1",
        "project_id": project.project_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "strategy": "coverage_first_then_risk_ranked_refinement",
        "trajectory_length_m": float(manifest["trajectory_length_m"]),
        "segment_count": len(segments),
        "stage_order": list(STAGE_ORDER),
        "stage_counts": counts,
        "segments": segments,
        "priority_queue": queue,
        "policy": {
            "automatic_pass": ["crop", "rails", "linear", "vertical", "platform", "canopy"],
            "manual_review": "only_P0_and_P1_before_geometry_freeze",
            "uncertain_assets": "retain_as_candidate_or_low_confidence_do_not_block_coverage",
            "release_gate": "fail_closed_after_global_topology_mesh_and_evidence_QA",
        },
    }
    output = project.workspace_path("reports") / "full_corridor_coverage_plan.json"
    write_json(output, result)
    result["output"] = str(output)
    return result


def _run_stage(project: ProjectConfig, segment_id: str, stage: str) -> dict[str, Any]:
    runners: dict[str, Callable[[], dict[str, Any]]] = {
        "rails": lambda: detect_rail_candidates(project, segment_id),
        "linear": lambda: detect_linear_candidates(project, segment_id),
        "vertical": lambda: analyze_vertical_hypotheses(project, segment_id),
        "platform": lambda: analyze_platform_surface(project, segment_id),
        "canopy": lambda: analyze_canopy_structure(project, segment_id),
    }
    return runners[stage]()


def run_full_corridor(
    project: ProjectConfig,
    through: str = "canopy",
    continue_on_error: bool = True,
) -> dict[str, Any]:
    if through not in STAGE_ORDER:
        raise ValueError(f"Unknown terminal stage: {through}")
    if "point_clouds" in project.value.get("inputs", {}):
        source_manifest = input_source_manifest_path(project)
        if not source_manifest.is_file():
            prepare_input_sources(project)
    segment_manifest = project.workspace_path("segment_manifest")
    if not segment_manifest.is_file():
        plan_segments(project)
    manifest = load_json(segment_manifest)
    segments = [str(item["id"]) for item in manifest.get("segments", [])]
    terminal_index = STAGE_ORDER.index(through)
    selected_stages = STAGE_ORDER[: terminal_index + 1]
    state_path = project.workspace_path("reports") / "full_corridor_run_state.json"
    state: dict[str, Any] = {
        "schema_version": "railway.full-corridor-run-state.v1",
        "project_id": project.project_id,
        "strategy": "coverage_first_then_risk_ranked_refinement",
        "started_at": datetime.now(UTC).isoformat(),
        "through": through,
        "continue_on_error": continue_on_error,
        "events": [],
    }

    if "crop" in selected_stages:
        missing = {
            segment_id
            for segment_id in segments
            if not _stage_output(project, segment_id, "crop").is_file()
        }
        if missing:
            try:
                crop_segments(project, missing)
                state["events"].append(
                    {"stage": "crop", "state": "complete", "segment_count": len(missing)}
                )
            except Exception as exc:
                state["events"].append(
                    {"stage": "crop", "state": "failed", "error": str(exc)}
                )
                state["updated_at"] = datetime.now(UTC).isoformat()
                write_json(state_path, state)
                if not continue_on_error:
                    raise

    for stage in selected_stages:
        if stage == "crop":
            continue
        for segment_id in segments:
            output = _stage_output(project, segment_id, stage)
            if output.is_file():
                state["events"].append(
                    {"segment_id": segment_id, "stage": stage, "state": "skipped_existing"}
                )
                continue
            dependency_missing = any(
                not _stage_output(project, segment_id, dependency).is_file()
                for dependency in STAGE_ORDER[: STAGE_ORDER.index(stage)]
            )
            if dependency_missing:
                state["events"].append(
                    {"segment_id": segment_id, "stage": stage, "state": "blocked_dependency"}
                )
                continue
            try:
                result = _run_stage(project, segment_id, stage)
                state["events"].append(
                    {
                        "segment_id": segment_id,
                        "stage": stage,
                        "state": "complete",
                        "report_status": result.get("status"),
                    }
                )
            except Exception as exc:
                state["events"].append(
                    {
                        "segment_id": segment_id,
                        "stage": stage,
                        "state": "failed",
                        "error": str(exc),
                    }
                )
                if not continue_on_error:
                    state["updated_at"] = datetime.now(UTC).isoformat()
                    write_json(state_path, state)
                    raise
            state["updated_at"] = datetime.now(UTC).isoformat()
            write_json(state_path, state)

    state["finished_at"] = datetime.now(UTC).isoformat()
    state["plan"] = build_full_corridor_plan(project)["output"]
    write_json(state_path, state)
    state["output"] = str(state_path)
    return state
