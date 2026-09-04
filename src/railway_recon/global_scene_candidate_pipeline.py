from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .algorithms.track_graph_mesh import build_track_graph_mesh
from .config import ProjectConfig
from .corridor_catenary_pipeline import run_corridor_catenary_pipeline
from .corridor_conductor_pipeline import run_corridor_conductor_pipeline
from .io import load_json, write_json
from .mesh_assembly import assemble_candidate_meshes


def run_global_scene_candidate_pipeline(
    project: ProjectConfig,
    graph_value: str | Path,
    graph_audit_value: str | Path,
    output_root_value: str | Path,
    output_name: str,
    *,
    station_assets_obj: str | Path | None = None,
    include_inferred_track_gaps: bool = False,
    maximum_inferred_track_gap_m: float = 250.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build one non-registering global scene candidate from validated component stages."""

    if not output_name or Path(output_name).name != output_name:
        raise ValueError("output_name must be one safe path component")
    graph_path = project.resolve(graph_value)
    graph_audit_path = project.resolve(graph_audit_value)
    for path in (graph_path, graph_audit_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_root = project.resolve(output_root_value)
    state_path = output_root / "pipeline_state.json"
    report_path = output_root / "pipeline_report.json"
    if not overwrite:
        existing = [str(path) for path in (state_path, report_path) if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite global pipeline outputs: {existing}")
    audit = load_json(graph_audit_path)
    candidate_nonpassing = audit.get("status") != "pass"
    state: dict[str, Any] = {
        "schema_version": "railway.global-scene-candidate-state.v1",
        "project_id": project.project_id,
        "started_at": datetime.now(UTC).isoformat(),
        "formal_release": False,
        "canonical_registry_updated": False,
        "events": [],
    }
    write_json(state_path, state)

    def run_stage(stage: str, action: Any) -> dict[str, Any]:
        event: dict[str, Any] = {
            "stage": stage,
            "started_at": datetime.now(UTC).isoformat(),
        }
        state["events"].append(event)
        write_json(state_path, state)
        try:
            result = action()
        except Exception as exc:
            event.update(
                {
                    "status": "failed",
                    "finished_at": datetime.now(UTC).isoformat(),
                    "error": str(exc),
                }
            )
            state["status"] = "failed_closed"
            state["failed_stage"] = stage
            state["finished_at"] = datetime.now(UTC).isoformat()
            write_json(state_path, state)
            raise
        event.update(
            {
                "status": "complete",
                "finished_at": datetime.now(UTC).isoformat(),
                "result_status": result.get("status"),
            }
        )
        write_json(state_path, state)
        return result

    track = run_stage(
        "track_candidate",
        lambda: build_track_graph_mesh(
            project,
            graph_path,
            graph_audit_path,
            overwrite=overwrite,
            output_dir_value=output_root / "track",
            report_dir_value=output_root / "track" / "reports",
            update_registry=False,
            candidate_only_nonpassing_audit=candidate_nonpassing,
            include_inferred_gap_hypotheses=include_inferred_track_gaps,
            maximum_inferred_gap_m=maximum_inferred_track_gap_m,
        ),
    )
    catenary = run_stage(
        "catenary_candidate",
        lambda: run_corridor_catenary_pipeline(
            project,
            output_root / "catenary",
            "corridor_catenary_candidate",
            overwrite=overwrite,
        ),
    )
    conductor = run_stage(
        "conductor_candidate",
        lambda: run_corridor_conductor_pipeline(
            project,
            graph_path,
            output_root / "conductor",
            "corridor_conductor_candidate",
            overwrite=overwrite,
        ),
    )
    sources = [
        ("TRACK", Path(track["output_obj"])),
        ("CATENARY", Path(catenary["output_obj"])),
        ("CONDUCTOR", Path(conductor["output_obj"])),
    ]
    station_path: Path | None = None
    if station_assets_obj is not None:
        station_path = project.resolve(station_assets_obj)
        if not station_path.is_file():
            raise FileNotFoundError(station_path)
        sources.append(("STATION", station_path))
    assembly = run_stage(
        "scene_assembly",
        lambda: assemble_candidate_meshes(
            project,
            output_root / "scene",
            output_name,
            sources,
            overwrite=overwrite,
        ),
    )
    state["status"] = "complete_candidate_only"
    state["finished_at"] = datetime.now(UTC).isoformat()
    write_json(state_path, state)
    report = {
        "schema_version": "railway.global-scene-candidate-pipeline.v1",
        "project_id": project.project_id,
        "status": "global_candidate_assembled_optimization_and_review_required",
        "formal_release": False,
        "canonical_registry_updated": False,
        "source_track_graph": str(graph_path),
        "source_track_graph_audit": str(graph_audit_path),
        "source_track_graph_audit_status": audit.get("status"),
        "candidate_nonpassing_track_audit_mode": candidate_nonpassing,
        "include_inferred_track_gaps": include_inferred_track_gaps,
        "maximum_inferred_track_gap_m": (
            float(maximum_inferred_track_gap_m)
            if include_inferred_track_gaps
            else None
        ),
        "station_assets_obj": str(station_path) if station_path else None,
        "track": {
            "status": track["status"],
            "output_obj": track["output_obj"],
            "track_count": len(track["tracks"]),
            "face_count": track["face_count"],
            "include_inferred_gap_hypotheses": track[
                "include_inferred_gap_hypotheses"
            ],
        },
        "catenary": {
            "status": catenary["status"],
            "output_obj": catenary["output_obj"],
            "mast_count": catenary["mast_count"],
        },
        "conductor": {
            "status": conductor["status"],
            "output_obj": conductor["output_obj"],
            "span_count": conductor["span_count"],
            "passing_seam_count": conductor["passing_seam_count"],
            "failing_adjacent_seam_count": conductor[
                "failing_adjacent_seam_count"
            ],
        },
        "scene": {
            "status": assembly["status"],
            "output_obj": assembly["output_obj"],
            "output_mtl": assembly["output_mtl"],
            "output_origin": assembly["output_origin"],
            "output_mesh_audit": assembly["output_mesh_audit"],
            "component_count": assembly["component_count"],
        },
        "pipeline_state": str(state_path),
        "limitations": [
            "This command produces a complete candidate-layer assembly, not a formal release.",
            "Non-passing source evidence remains explicit in component reports.",
            "Station assets are optional and must already have passed their own candidate gates.",
            "No canonical asset registry is updated by this pipeline.",
        ],
    }
    write_json(report_path, report)
    return report
