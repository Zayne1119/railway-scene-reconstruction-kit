from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import __version__
from .algorithms.linear_candidates import detect_linear_candidates
from .algorithms.rail_candidates import detect_rail_candidates
from .algorithms.reviewed_scene import build_reviewed_scene
from .algorithms.track import build_parametric_track
from .algorithms.track_graph_mesh import build_track_graph_mesh
from .annotation_tasks import (
    create_vertical_annotation_package,
    validate_vertical_annotation_package,
)
from .audit import audit_project
from .benchmark import (
    freeze_benchmark,
    initialize_benchmark,
    validate_benchmark_file,
    validate_benchmark_root,
)
from .benchmark_metrics import evaluate_benchmark_file
from .canopy_photo_evidence import analyze_canopy_photo_evidence
from .canopy_structure import analyze_canopy_structure
from .catenary_candidate_mesh import (
    audit_catenary_candidate_section,
    audit_catenary_track_clearance_files,
    build_catenary_candidate_mesh,
)
from .config import initialize_project, load_project, validate_project_value
from .corridor_canopy_candidate import build_corridor_canopy_candidate
from .corridor_catenary_pipeline import run_corridor_catenary_pipeline
from .corridor_column_grid import recover_corridor_column_grid
from .corridor_column_support import validate_corridor_column_grid
from .corridor_conductor_pipeline import run_corridor_conductor_pipeline
from .defect_injection import evaluate_defect_injection
from .experiment_lock import lock_benchmark_experiment
from .fixed_view_auto_review import write_fixed_view_auto_review
from .full_corridor import STAGE_ORDER, build_full_corridor_plan, run_full_corridor
from .gislab_baseline import (
    normalize_gislab_railtrack_points,
    prepare_gislab_railtrack_input,
    record_gislab_no_detection,
    record_gislab_timeout,
)
from .global_scene_candidate_pipeline import run_global_scene_candidate_pipeline
from .holdout_bootstrap import bootstrap_rail_holdout_comparison
from .holdout_comparison import compare_rail_holdout_reports
from .io import load_json
from .manifest import write_run_manifest
from .mesh_assembly import assemble_candidate_meshes, parse_mesh_source_specs
from .mesh_audit import audit_obj
from .mesh_object_registry import build_mesh_object_registry
from .mesh_seam_reconciliation import reconcile_mesh_seams
from .multi_source import prepare_input_sources
from .open3d_baseline import detect_open3d_rail_baseline
from .opposite_platform_reconstruction import reconstruct_opposite_platform
from .platform_candidate_gate import evaluate_platform_candidate_gate
from .platform_gap_point_evidence import analyze_platform_gap_point_evidence
from .platform_interface_audit import audit_platform_interfaces
from .platform_mesh import build_platform_mesh
from .platform_photo_evidence import analyze_platform_photo_evidence
from .platform_surface import analyze_platform_surface
from .point_holdout import (
    split_existing_point_cloud_holdout,
    split_point_cloud_holdout,
    validate_point_cloud_holdout,
)
from .prediction_lock import lock_vertical_predictions
from .projection import calibrate_projection, calibrate_projection_consensus
from .qa import quality_report
from .quality_gate import evaluate_quality_gate, validate_gate_result
from .rail_comparison_figure import render_rail_method_comparison
from .rail_holdout_metrics import evaluate_rail_holdout
from .rail_pair_curation import curate_rail_pairs
from .rail_production_regression import audit_rail_production_regression
from .rail_recovery_selection import select_corridor_rail_recoveries
from .rail_review import (
    apply_rail_owner_override,
    apply_rail_review_package,
    create_rail_review_package,
)
from .rail_targeted_recovery import (
    apply_targeted_rail_recovery,
    evaluate_targeted_rail_recovery,
)
from .rail_truth_evidence import render_neutral_rail_evidence
from .rail_truth_tasks import (
    create_rail_truth_annotation_package,
    validate_rail_truth_annotation_package,
)
from .rapid_candidate_audit import write_rapid_candidate_audit
from .recovery_binding import bind_targeted_recovery_to_scene
from .registry import (
    freeze_release_registry,
    import_registry_records,
    initialize_registry,
    summarize_registry,
    validate_registry_file,
)
from .release import build_web_acceptance_config
from .review_agreement import compare_vertical_reviews
from .safety import safety_check
from .seam_evidence import compare_seam_rail_reports
from .segment_mesh_ownership import (
    audit_owned_mesh_seams,
    build_segment_ownership_plan,
    clip_segment_mesh_ownership,
)
from .segments import crop_segment_context_sources, crop_segments, plan_segments
from .side_asset_pipeline import run_side_asset_pipeline
from .small_asset_candidates import analyze_small_platform_asset_candidates
from .small_asset_photo_evidence import analyze_small_asset_photo_evidence
from .station_platform_pipeline import run_station_platform_pipeline
from .support_relationship_gate import (
    SupportRelationshipPolicy,
    validate_support_relationship_patch,
)
from .targeted_canopy_integration import integrate_targeted_canopy_candidate
from .targeted_canopy_recovery import (
    recover_targeted_canopy,
    targeted_canopy_photo_evidence,
)
from .topology_ablation import evaluate_topology_ablation
from .track_graph import build_track_graph, validate_track_graph
from .vertical_conflict_photo_evidence import (
    analyze_selected_vertical_photo_evidence,
    analyze_vertical_conflict_photo_evidence,
)
from .vertical_fit_selection import select_rail_vertical_fit
from .vertical_followup import audit_rail_vertical_followup
from .vertical_hypotheses import analyze_vertical_hypotheses


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _segment_sources(values: list[str]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --source {value!r}; expected SEGMENT=PATH")
        segment_id, path = value.split("=", 1)
        if not segment_id or not path:
            raise ValueError(f"Invalid --source {value!r}; expected SEGMENT=PATH")
        result.append((segment_id, path))
    return result


def _assignments(values: list[str], option: str) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid {option} {value!r}; expected ID=PATH")
        identifier, path = value.split("=", 1)
        if not identifier or not path:
            raise ValueError(f"Invalid {option} {value!r}; expected ID=PATH")
        result.append((identifier, path))
    return result


def _projection_samples(values: list[str]) -> list[tuple[str, int]]:
    result: list[tuple[str, int]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --sample {value!r}; expected SEGMENT=CAMERA_INDEX")
        segment_id, camera_index = value.split("=", 1)
        if not segment_id or not camera_index:
            raise ValueError(f"Invalid --sample {value!r}; expected SEGMENT=CAMERA_INDEX")
        try:
            result.append((segment_id, int(camera_index)))
        except ValueError as exc:
            raise ValueError(
                f"Invalid --sample {value!r}; camera index must be an integer"
            ) from exc
    return result


def _vertical_fit_variants(values: list[str]) -> list[tuple[str, str, str]]:
    result: list[tuple[str, str, str]] = []
    for value in values:
        parts = value.split("=", 2)
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"Invalid --variant {value!r}; expected NAME=EVALUATION=SETTINGS")
        result.append((parts[0], parts[1], parts[2]))
    return result


def _project_command(
    args: argparse.Namespace,
    name: str,
    action: Callable[..., Any],
    required_status: set[str] | None = None,
) -> int:
    project = load_project(args.project)
    try:
        result = action(project)
        outputs: list[str] = []
        if isinstance(result, dict):
            outputs = [
                str(value)
                for key, value in result.items()
                if key.endswith("_path") or key.startswith("output_") or key == "origin"
            ]
        manifest = write_run_manifest(project, name, "completed", outputs=outputs)
        if isinstance(result, dict):
            result["run_manifest"] = str(manifest)
            _print(result)
        else:
            _print({"result": str(result), "run_manifest": str(manifest)})
        if required_status is not None:
            status = result.get("status") if isinstance(result, dict) else None
            if status not in required_status:
                return 5
        return 0
    except Exception as exc:
        write_run_manifest(project, name, "failed", metrics={"error": str(exc)})
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="railway-recon",
        description="Evidence-aware railway scene reconstruction workflow",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="Create a new data-safe project skeleton")
    init.add_argument("target")
    init.add_argument("--name", required=True)
    init.add_argument("--project-id")

    validate = commands.add_parser("validate", help="Validate project.json")
    validate.add_argument("--project", required=True)

    audit = commands.add_parser("audit", help="Audit point cloud, cameras and panoramas")
    audit.add_argument("--project", required=True)
    audit.add_argument("--full-hash", action="store_true")

    prepare_inputs = commands.add_parser(
        "prepare-inputs",
        help="Prepare camera coverage and ownership for one or more point-cloud sources",
    )
    prepare_inputs.add_argument("--project", required=True)
    prepare_inputs.add_argument("--full-hash", action="store_true")
    prepare_inputs.add_argument("--camera-bbox-margin-m", type=float, default=0.0)

    plan = commands.add_parser("plan-segments", help="Plan corridor segments from camera poses")
    plan.add_argument("--project", required=True)

    segment = commands.add_parser("segment", help="Crop planned LAS/LAZ corridor segments")
    segment.add_argument("--project", required=True)
    segment.add_argument("--ids", nargs="*")
    segment.add_argument("--overwrite", action="store_true")

    segment_context = commands.add_parser(
        "segment-context",
        help="Crop one identical seam envelope from every declared context LAZ",
    )
    segment_context.add_argument("--project", required=True)
    segment_context.add_argument("--segment", required=True)
    segment_context.add_argument("--overwrite", action="store_true")

    full_corridor_plan = commands.add_parser(
        "full-corridor-plan",
        help="Build a resumable coverage-first plan and risk-ranked review queue",
    )
    full_corridor_plan.add_argument("--project", required=True)

    full_corridor_run = commands.add_parser(
        "full-corridor-run",
        help="Run the automatic full-corridor first pass with checkpointed failure isolation",
    )
    full_corridor_run.add_argument("--project", required=True)
    full_corridor_run.add_argument("--through", choices=STAGE_ORDER, default="canopy")
    full_corridor_run.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first segment failure instead of completing unaffected segments",
    )

    global_scene_candidate = commands.add_parser(
        "run-global-scene-candidate",
        help="Run track, catenary, conductor and optional station assembly as one fail-closed candidate stage",
    )
    global_scene_candidate.add_argument("--project", required=True)
    global_scene_candidate.add_argument("--track-graph", required=True)
    global_scene_candidate.add_argument("--track-graph-audit", required=True)
    global_scene_candidate.add_argument("--station-assets-obj")
    global_scene_candidate.add_argument(
        "--include-inferred-track-gaps",
        action="store_true",
        help="Add a separate orange candidate layer across bounded rail-evidence gaps",
    )
    global_scene_candidate.add_argument(
        "--maximum-inferred-track-gap-m", type=float, default=250.0
    )
    global_scene_candidate.add_argument("--output-root", required=True)
    global_scene_candidate.add_argument("--output-name", required=True)
    global_scene_candidate.add_argument("--overwrite", action="store_true")

    detect_rails = commands.add_parser(
        "detect-rails", help="Extract baseline rail candidates from a pilot segment"
    )
    detect_rails.add_argument("--project", required=True)
    detect_rails.add_argument("--segment", required=True)
    detect_rails.add_argument(
        "--source", help="Optional existing LAS/LAZ segment; input is read-only"
    )
    detect_rails.add_argument("--settings", help="Optional frozen rail detector settings")
    detect_rails.add_argument("--report", help="Optional explicit JSON report output")
    detect_rails.add_argument("--overwrite", action="store_true")

    detect_linear = commands.add_parser(
        "detect-linear", help="Extract vertical and cable geometry candidates"
    )
    detect_linear.add_argument("--project", required=True)
    detect_linear.add_argument("--segment", required=True)
    detect_linear.add_argument("--overwrite", action="store_true")

    classify_vertical = commands.add_parser(
        "classify-vertical",
        help="Merge and score vertical candidates without writing reviewed assets",
    )
    classify_vertical.add_argument("--project", required=True)
    classify_vertical.add_argument("--segment", required=True)
    classify_vertical.add_argument("--linear-report")
    classify_vertical.add_argument("--rail-report")
    classify_vertical.add_argument("--settings")
    classify_vertical.add_argument("--overwrite", action="store_true")

    analyze_canopy = commands.add_parser(
        "analyze-canopy",
        help="Extract column-seeded roof surfaces and audit candidate contacts",
    )
    analyze_canopy.add_argument("--project", required=True)
    analyze_canopy.add_argument("--segment", required=True)
    analyze_canopy.add_argument("--vertical-report")
    analyze_canopy.add_argument("--settings")
    analyze_canopy.add_argument("--overwrite", action="store_true")

    canopy_photo = commands.add_parser(
        "canopy-photo-evidence",
        help="Project canopy column tops into calibrated panoramas and score directional edges",
    )
    canopy_photo.add_argument("--project", required=True)
    canopy_photo.add_argument("--segment", required=True)
    canopy_photo.add_argument("--projection-consensus", required=True)
    canopy_photo.add_argument("--canopy-report")
    canopy_photo.add_argument("--vertical-report")
    canopy_photo.add_argument("--settings")
    canopy_photo.add_argument("--overwrite", action="store_true")

    analyze_platform = commands.add_parser(
        "analyze-platform",
        help="Extract observed platform-top components and segmented boundary/elevation fits",
    )
    analyze_platform.add_argument("--project", required=True)
    analyze_platform.add_argument("--segment", required=True)
    analyze_platform.add_argument("--vertical-report")
    analyze_platform.add_argument("--settings")
    analyze_platform.add_argument("--overwrite", action="store_true")

    platform_photo = commands.add_parser(
        "platform-photo-evidence",
        help="Project platform edges and safety-line probes into calibrated panoramas",
    )
    platform_photo.add_argument("--project", required=True)
    platform_photo.add_argument("--segment", required=True)
    platform_photo.add_argument("--projection-consensus", required=True)
    platform_photo.add_argument("--platform-report")
    platform_photo.add_argument("--settings")
    platform_photo.add_argument("--overwrite", action="store_true")

    platform_mesh = commands.add_parser(
        "build-platform-mesh",
        help="Build a continuous candidate platform mesh after a passed gap review gate",
    )
    platform_mesh.add_argument("--project", required=True)
    platform_mesh.add_argument("--segment", required=True)
    platform_mesh.add_argument("--mesh-gate", required=True)
    platform_mesh.add_argument("--platform-report")
    platform_mesh.add_argument("--settings")
    platform_mesh.add_argument("--overwrite", action="store_true")

    platform_candidate_gate = commands.add_parser(
        "create-platform-candidate-gate",
        help="Authorize candidate-only platform mesh when deterministic gap and fit QA pass",
    )
    platform_candidate_gate.add_argument("--project", required=True)
    platform_candidate_gate.add_argument("--segment", required=True)
    platform_candidate_gate.add_argument("--platform-report", required=True)
    platform_candidate_gate.add_argument("--output", required=True)
    platform_candidate_gate.add_argument("--component-id")
    platform_candidate_gate.add_argument("--ownership-plan")
    platform_candidate_gate.add_argument("--gap-point-evidence")
    platform_candidate_gate.add_argument("--overwrite", action="store_true")

    platform_gap_points = commands.add_parser(
        "analyze-platform-gap-points",
        help="Classify platform surface gaps from above/below-surface point evidence",
    )
    platform_gap_points.add_argument("--project", required=True)
    platform_gap_points.add_argument("--segment", required=True)
    platform_gap_points.add_argument("--platform-report", required=True)
    platform_gap_points.add_argument("--output", required=True)
    platform_gap_points.add_argument("--settings")
    platform_gap_points.add_argument("--overwrite", action="store_true")

    platform_interfaces = commands.add_parser(
        "audit-platform-interfaces",
        help="Audit platform contacts and conflicts with rails, verticals and canopy evidence",
    )
    platform_interfaces.add_argument("--project", required=True)
    platform_interfaces.add_argument("--segment", required=True)
    platform_interfaces.add_argument("--mesh-build-report")
    platform_interfaces.add_argument("--platform-report")
    platform_interfaces.add_argument("--vertical-report")
    platform_interfaces.add_argument("--canopy-report")
    platform_interfaces.add_argument("--settings")
    platform_interfaces.add_argument("--overwrite", action="store_true")

    opposite_platform = commands.add_parser(
        "reconstruct-opposite-platform",
        help="Run a multi-density gate and build one conservative opposite platform",
    )
    opposite_platform.add_argument("--project", required=True)
    opposite_platform.add_argument("--segment", required=True)
    opposite_platform.add_argument("--point-cloud", required=True)
    opposite_platform.add_argument("--vertical-report", required=True)
    opposite_platform.add_argument("--output-dir", required=True)
    opposite_platform.add_argument("--side", choices=("left", "right"), required=True)
    opposite_platform.add_argument("--minimum-longitudinal-m", type=float)
    opposite_platform.add_argument("--maximum-longitudinal-m", type=float)
    opposite_platform.add_argument(
        "--ownership-plan",
        help="Preferred for adjacent segments; derives this segment's local owned interval",
    )
    opposite_platform.add_argument("--settings")
    opposite_platform.add_argument("--overwrite", action="store_true")

    vertical_conflict_photo = commands.add_parser(
        "vertical-conflict-photo-evidence",
        help="Project platform-contacting semantic conflicts into calibrated panoramas",
    )
    vertical_conflict_photo.add_argument("--project", required=True)
    vertical_conflict_photo.add_argument("--segment", required=True)
    vertical_conflict_photo.add_argument("--projection-consensus", required=True)
    vertical_conflict_photo.add_argument("--interface-report")
    vertical_conflict_photo.add_argument("--vertical-report")
    vertical_conflict_photo.add_argument("--settings")
    vertical_conflict_photo.add_argument("--overwrite", action="store_true")

    selected_vertical_photo = commands.add_parser(
        "vertical-candidate-photo-evidence",
        help="Project explicitly selected vertical hypotheses into calibrated panoramas",
    )
    selected_vertical_photo.add_argument("--project", required=True)
    selected_vertical_photo.add_argument("--segment", required=True)
    selected_vertical_photo.add_argument("--candidate", action="append", required=True)
    selected_vertical_photo.add_argument("--projection-consensus", required=True)
    selected_vertical_photo.add_argument("--output-name", required=True)
    selected_vertical_photo.add_argument("--vertical-report")
    selected_vertical_photo.add_argument("--settings")
    selected_vertical_photo.add_argument("--overwrite", action="store_true")

    catenary_section = commands.add_parser(
        "catenary-section-audit",
        help="Fit a robust local section to one photo-reviewable catenary candidate",
    )
    catenary_section.add_argument("--project", required=True)
    catenary_section.add_argument("--segment", required=True)
    catenary_section.add_argument("--candidate", required=True)
    catenary_section.add_argument("--vertical-report")
    catenary_section.add_argument("--output")
    catenary_section.add_argument("--overwrite", action="store_true")

    catenary_mesh = commands.add_parser(
        "catenary-candidate-mesh",
        help="Build an evidence-reviewed catenary mast and foundation candidate",
    )
    catenary_mesh.add_argument("--project", required=True)
    catenary_mesh.add_argument("--vertical-report", required=True)
    catenary_mesh.add_argument("--semantic-review", required=True)
    catenary_mesh.add_argument("--section-audit", required=True)
    catenary_mesh.add_argument("--output-dir", required=True)
    catenary_mesh.add_argument("--report-dir", required=True)
    catenary_mesh.add_argument("--overwrite", action="store_true")

    corridor_catenary = commands.add_parser(
        "run-corridor-catenary-pipeline",
        help="Deduplicate corridor verticals and build only periodic point-supported mast candidates",
    )
    corridor_catenary.add_argument("--project", required=True)
    corridor_catenary.add_argument("--output-dir", required=True)
    corridor_catenary.add_argument("--output-name", required=True)
    corridor_catenary.add_argument("--settings")
    corridor_catenary.add_argument("--overwrite", action="store_true")

    corridor_conductor = commands.add_parser(
        "run-corridor-conductor-pipeline",
        help="Bind cable fragments to recovered tracks by lateral offset and rail-top height",
    )
    corridor_conductor.add_argument("--project", required=True)
    corridor_conductor.add_argument("--track-graph", required=True)
    corridor_conductor.add_argument("--output-dir", required=True)
    corridor_conductor.add_argument("--output-name", required=True)
    corridor_conductor.add_argument("--settings")
    corridor_conductor.add_argument("--overwrite", action="store_true")

    catenary_clearance = commands.add_parser(
        "catenary-track-clearance-audit",
        help="Check a reviewed mast foundation envelope against track-bed envelopes",
    )
    catenary_clearance.add_argument("--project", required=True)
    catenary_clearance.add_argument("--track-graph", required=True)
    catenary_clearance.add_argument("--section-audit", required=True)
    catenary_clearance.add_argument("--track-build", required=True)
    catenary_clearance.add_argument("--output", required=True)
    catenary_clearance.add_argument("--overwrite", action="store_true")

    ownership_plan = commands.add_parser(
        "build-segment-ownership-plan",
        help="Build one shared midpoint ownership frame for adjacent segments",
    )
    ownership_plan.add_argument("--project", required=True)
    ownership_plan.add_argument(
        "--segment-frame",
        required=True,
        action="append",
        help="Repeat as SEGMENT_ID=FRAME_REPORT_JSON",
    )
    ownership_plan.add_argument("--output", required=True)
    ownership_plan.add_argument("--overwrite", action="store_true")

    mesh_ownership = commands.add_parser(
        "clip-segment-mesh-ownership",
        help="Clip a buffered candidate mesh to one segment's owned longitudinal interval",
    )
    mesh_ownership.add_argument("--project", required=True)
    mesh_ownership.add_argument("--source-obj", required=True)
    mesh_ownership.add_argument("--source-origin", required=True)
    mesh_ownership.add_argument("--frame-report", required=True)
    mesh_ownership.add_argument("--output-dir", required=True)
    mesh_ownership.add_argument("--minimum-longitudinal-m", required=True, type=float)
    mesh_ownership.add_argument("--maximum-longitudinal-m", required=True, type=float)
    mesh_ownership.add_argument("--overwrite", action="store_true")

    seam_audit = commands.add_parser(
        "audit-owned-mesh-seams",
        help="Audit adjacent owned mesh cut edges in one shared corridor frame",
    )
    seam_audit.add_argument("--project", required=True)
    seam_audit.add_argument("--ownership-plan", required=True)
    seam_audit.add_argument(
        "--segment-mesh",
        required=True,
        action="append",
        help="Repeat as SEGMENT_ID=OWNED_OBJ",
    )
    seam_audit.add_argument("--output", required=True)
    seam_audit.add_argument("--p90-max-m", required=True, type=float)
    seam_audit.add_argument("--boundary-tolerance-m", type=float, default=0.002)
    seam_audit.add_argument("--sample-spacing-m", type=float, default=0.02)
    seam_audit.add_argument("--overwrite", action="store_true")

    mesh_assembly = commands.add_parser(
        "assemble-candidate-meshes",
        help="Merge candidate OBJ components with source namespaces and mesh QA",
    )
    mesh_assembly.add_argument("--project", required=True)
    mesh_assembly.add_argument("--source", required=True, action="append")
    mesh_assembly.add_argument("--output-dir", required=True)
    mesh_assembly.add_argument("--output-name", required=True)
    mesh_assembly.add_argument("--overwrite", action="store_true")

    station_platform = commands.add_parser(
        "run-station-platform-pipeline",
        help="Build, clip, seam-audit and assemble safe station platform segments",
    )
    station_platform.add_argument("--project", required=True)
    station_platform.add_argument("--ownership-plan", required=True)
    station_platform.add_argument("--segment", required=True, action="append")
    station_platform.add_argument("--output-name", required=True)
    station_platform.add_argument("--overwrite", action="store_true")

    column_grid = commands.add_parser(
        "recover-corridor-column-grid",
        help="Fit one canopy-column grid phase across adjacent segment ownership ranges",
    )
    column_grid.add_argument("--project", required=True)
    column_grid.add_argument("--ownership-plan", required=True)
    column_grid.add_argument(
        "--vertical-report",
        action="append",
        required=True,
        help="Repeat as SEGMENT_ID=VERTICAL_HYPOTHESES_JSON",
    )
    column_grid.add_argument("--output", required=True)
    column_grid.add_argument("--side", choices=("left", "right"), required=True)
    column_grid.add_argument("--settings")
    column_grid.add_argument("--overwrite", action="store_true")

    corridor_canopy = commands.add_parser(
        "build-corridor-canopy-candidate",
        help="Build a roof-and-column candidate from a point-supported corridor grid",
    )
    corridor_canopy.add_argument("--project", required=True)
    corridor_canopy.add_argument("--ownership-plan", required=True)
    corridor_canopy.add_argument("--supported-grid", required=True)
    corridor_canopy.add_argument("--output-name", required=True)
    corridor_canopy.add_argument("--settings")
    corridor_canopy.add_argument("--overwrite", action="store_true")

    column_support = commands.add_parser(
        "validate-corridor-column-grid",
        help="Check every inferred corridor column position against local point support",
    )
    column_support.add_argument("--project", required=True)
    column_support.add_argument("--grid", required=True)
    column_support.add_argument("--output", required=True)
    column_support.add_argument("--settings")
    column_support.add_argument("--overwrite", action="store_true")

    seam_reconcile = commands.add_parser(
        "reconcile-mesh-seams",
        help="Weld configured segment mesh endpoints and remove proven duplicate caps",
    )
    seam_reconcile.add_argument("--project", required=True)
    seam_reconcile.add_argument("--source-obj", required=True)
    seam_reconcile.add_argument("--source-origin", required=True)
    seam_reconcile.add_argument("--frame-report", required=True)
    seam_reconcile.add_argument("--settings", required=True)
    seam_reconcile.add_argument("--output-dir", required=True)
    seam_reconcile.add_argument("--registry")
    seam_reconcile.add_argument("--overwrite", action="store_true")

    side_asset_pipeline = commands.add_parser(
        "run-side-asset-pipeline",
        help="Run opposite-platform, corridor-column-grid and seam-repair stages from one plan",
    )
    side_asset_pipeline.add_argument("--project", required=True)
    side_asset_pipeline.add_argument("--plan", required=True)
    side_asset_pipeline.add_argument("--output", required=True)
    side_asset_pipeline.add_argument("--overwrite", action="store_true")

    mesh_registry = commands.add_parser(
        "build-mesh-object-registry",
        help="Register every namespaced renderable object in a composed OBJ",
    )
    mesh_registry.add_argument("--project", required=True)
    mesh_registry.add_argument("--obj", required=True)
    mesh_registry.add_argument("--output", required=True)
    mesh_registry.add_argument("--evidence-reference", required=True)
    mesh_registry.add_argument("--overwrite", action="store_true")

    small_assets = commands.add_parser(
        "analyze-small-platform-assets",
        help="Extract owned above-platform connected components for photo review",
    )
    small_assets.add_argument("--project", required=True)
    small_assets.add_argument("--segment", required=True)
    small_assets.add_argument("--platform-report", required=True)
    small_assets.add_argument("--ownership-plan", required=True)
    small_assets.add_argument("--overwrite", action="store_true")

    small_asset_photo = commands.add_parser(
        "small-asset-photo-evidence",
        help="Project owned small above-platform candidates into calibrated panoramas",
    )
    small_asset_photo.add_argument("--project", required=True)
    small_asset_photo.add_argument("--segment", required=True)
    small_asset_photo.add_argument("--small-asset-report", required=True)
    small_asset_photo.add_argument("--projection-consensus", required=True)
    small_asset_photo.add_argument("--settings")
    small_asset_photo.add_argument("--overwrite", action="store_true")

    targeted_canopy = commands.add_parser(
        "targeted-canopy-recover",
        help="Recover a reviewed platform canopy without mutating source predictions",
    )
    targeted_canopy.add_argument("--project", required=True)
    targeted_canopy.add_argument("--segment", required=True)
    targeted_canopy.add_argument("--semantic-review", required=True)
    targeted_canopy.add_argument("--vertical-report")
    targeted_canopy.add_argument("--platform-report")
    targeted_canopy.add_argument("--settings")
    targeted_canopy.add_argument("--grid-review")
    targeted_canopy.add_argument("--overwrite", action="store_true")

    targeted_canopy_photo = commands.add_parser(
        "targeted-canopy-photo-evidence",
        help="Project inferred canopy grid positions into calibrated panoramas",
    )
    targeted_canopy_photo.add_argument("--project", required=True)
    targeted_canopy_photo.add_argument("--segment", required=True)
    targeted_canopy_photo.add_argument("--recovery-report", required=True)
    targeted_canopy_photo.add_argument("--projection-consensus", required=True)
    targeted_canopy_photo.add_argument("--settings")
    targeted_canopy_photo.add_argument("--overwrite", action="store_true")

    targeted_canopy_integrate = commands.add_parser(
        "targeted-canopy-integrate",
        help="Merge reviewed canopy nodes into a versioned candidate scene and registry",
    )
    targeted_canopy_integrate.add_argument("--project", required=True)
    targeted_canopy_integrate.add_argument("--segment", required=True)
    targeted_canopy_integrate.add_argument("--recovery-report", required=True)
    targeted_canopy_integrate.add_argument("--visual-review", required=True)
    targeted_canopy_integrate.add_argument("--release-id", required=True)
    targeted_canopy_integrate.add_argument(
        "--approve-candidate-merge",
        action="store_true",
        help="Explicitly authorize candidate model and canonical registry writes",
    )
    targeted_canopy_integrate.add_argument("--overwrite", action="store_true")

    build_track = commands.add_parser(
        "build-track", help="Build a baseline parametric track OBJ and register assets"
    )
    build_track.add_argument("--project", required=True)
    build_track.add_argument("--segment", required=True)
    build_track.add_argument("--overwrite", action="store_true")

    build_graph = commands.add_parser(
        "track-graph-build",
        help="Build global track identities and topology before any cross-segment mesh",
    )
    build_graph.add_argument("--project", required=True)
    build_graph.add_argument(
        "--source", action="append", required=True, help="Repeat SEGMENT=RAIL_REPORT"
    )
    build_graph.add_argument("--output", help="Optional versioned TrackGraph JSON output")
    build_graph.add_argument("--audit", help="Optional versioned TrackGraph audit output")
    build_graph.add_argument("--overwrite", action="store_true")

    validate_graph = commands.add_parser(
        "track-graph-validate",
        help="Fail closed on reversed, broken, duplicated or unsupported tracks",
    )
    validate_graph.add_argument("--project", required=True)
    validate_graph.add_argument("--graph", required=True)
    validate_graph.add_argument("--output")

    targeted_recovery = commands.add_parser(
        "track-continuity-recover",
        help="Recheck unresolved rail pairs at their fixed TrackGraph centers",
    )
    targeted_recovery.add_argument("--graph", required=True)
    targeted_recovery.add_argument("--output", required=True)
    targeted_recovery.add_argument(
        "--updated-graph",
        help="Optional new TrackGraph with recovered evidence; source graph is never overwritten",
    )
    targeted_recovery.add_argument(
        "--observation-id",
        action="append",
        default=[],
        help=(
            "Repeat an observation ID to re-evaluate an already resolved fixed center; "
            "explicit IDs are added to unresolved recovery tasks"
        ),
    )
    targeted_recovery.add_argument("--longitudinal-bin-m", type=float, default=0.50)
    targeted_recovery.add_argument("--minimum-bin-points", type=int, default=3)
    targeted_recovery.add_argument("--cross-half-width-m", type=float, default=0.10)
    targeted_recovery.add_argument("--vertical-below-m", type=float, default=0.12)
    targeted_recovery.add_argument("--vertical-above-m", type=float, default=0.05)
    targeted_recovery.add_argument("--minimum-joint-support-ratio", type=float, default=0.60)
    targeted_recovery.add_argument("--maximum-asymmetric-support-ratio", type=float, default=0.25)
    targeted_recovery.add_argument("--maximum-internal-joint-gap-m", type=float, default=5.0)

    recovery_selection = commands.add_parser(
        "track-recovery-select",
        help="Run fail-closed per-segment rail recovery A/B selection over corridor gaps",
    )
    recovery_selection.add_argument("--project", required=True)
    recovery_selection.add_argument("--baseline-graph", required=True)
    recovery_selection.add_argument("--baseline-audit", required=True)
    recovery_selection.add_argument("--settings", required=True)
    recovery_selection.add_argument("--output-name", default="rail_recovery_selection_v1")
    recovery_selection.add_argument("--overwrite", action="store_true")

    bind_recovery = commands.add_parser(
        "track-recovery-bind-scene",
        help="Bind fixed-center recovery evidence to a new GLB without changing geometry",
    )
    bind_recovery.add_argument("--source-glb", required=True)
    bind_recovery.add_argument("--registry", required=True)
    bind_recovery.add_argument("--recovered-graph", required=True)
    bind_recovery.add_argument("--recovery", required=True)
    bind_recovery.add_argument("--output-glb", required=True)
    bind_recovery.add_argument("--output-registry", required=True)
    bind_recovery.add_argument("--report", required=True)
    bind_recovery.add_argument("--profile", required=True)

    build_graph_mesh = commands.add_parser(
        "build-track-graph-mesh",
        help="Build continuous rails, globally phased sleepers and track beds from a passing TrackGraph",
    )
    build_graph_mesh.add_argument("--project", required=True)
    build_graph_mesh.add_argument("--graph")
    build_graph_mesh.add_argument("--graph-audit")
    build_graph_mesh.add_argument("--output-dir")
    build_graph_mesh.add_argument("--report-dir")
    build_graph_mesh.add_argument("--skip-registry", action="store_true")
    build_graph_mesh.add_argument(
        "--candidate-only-nonpassing-audit",
        action="store_true",
        help=(
            "Write a non-registering review candidate when core topology checks pass but "
            "the source TrackGraph still has explicit evidence or measurement failures"
        ),
    )
    build_graph_mesh.add_argument(
        "--include-inferred-gap-hypotheses",
        action="store_true",
        help="Render bounded evidence gaps as separate orange candidate objects",
    )
    build_graph_mesh.add_argument(
        "--maximum-inferred-gap-m", type=float, default=250.0
    )
    build_graph_mesh.add_argument("--overwrite", action="store_true")

    create_track_review = commands.add_parser(
        "track-review-create",
        help="Create a hash-bound independent review queue for rail candidates",
    )
    create_track_review.add_argument("--project", required=True)
    create_track_review.add_argument(
        "--source", action="append", required=True, help="Repeat SEGMENT=RAIL_REPORT"
    )
    create_track_review.add_argument("--output-name", default="track_graph_rail_review_v1")

    apply_track_review = commands.add_parser(
        "track-review-apply",
        help="Validate completed rail review decisions and emit reviewed source copies",
    )
    apply_track_review.add_argument("--project", required=True)
    apply_track_review.add_argument("--path", required=True)

    override_track_review = commands.add_parser(
        "track-review-owner-override",
        help="Accept hash-bound rail candidates by explicit project-owner waiver",
    )
    override_track_review.add_argument("--project", required=True)
    override_track_review.add_argument("--path", required=True)
    override_track_review.add_argument("--approved-by", required=True)
    override_track_review.add_argument("--reason", required=True)

    curate_track_pairs = commands.add_parser(
        "track-pair-curate",
        help="Apply hash-bound pair selection, topology boundaries and explicit inferred pairs",
    )
    curate_track_pairs.add_argument("--project", required=True)
    curate_track_pairs.add_argument("--selection", required=True)
    curate_track_pairs.add_argument("--output-name", required=True)

    build_scene = commands.add_parser(
        "build-reviewed-scene",
        help="Generate station/catenary assets from an evidence-reviewed layout",
    )
    build_scene.add_argument("--project", required=True)
    build_scene.add_argument("--layout", required=True)
    build_scene.add_argument("--overwrite", action="store_true")

    projection = commands.add_parser(
        "calibrate-projection",
        help="Search point-to-panorama pose conventions and render an evidence overlay",
    )
    projection.add_argument("--project", required=True)
    projection.add_argument("--segment", required=True)
    projection.add_argument("--camera-index", required=True, type=int)
    projection.add_argument("--overwrite", action="store_true")

    projection_consensus = commands.add_parser(
        "calibrate-projection-consensus",
        help="Select one point-to-panorama convention from at least three cameras",
    )
    projection_consensus.add_argument("--project", required=True)
    projection_consensus.add_argument(
        "--sample",
        action="append",
        required=True,
        help="Repeat SEGMENT=CAMERA_INDEX for three or more independent panoramas",
    )
    projection_consensus.add_argument("--output-name", required=True)
    projection_consensus.add_argument("--overwrite", action="store_true")

    seam_compare = commands.add_parser(
        "seam-rail-compare",
        help="Compare rail heads extracted from two LAZ sources in one shared segment frame",
    )
    seam_compare.add_argument("--project", required=True)
    seam_compare.add_argument("--left-report", required=True)
    seam_compare.add_argument("--right-report", required=True)
    seam_compare.add_argument("--output", required=True)

    registry_init = commands.add_parser("registry-init", help="Create an empty asset registry")
    registry_init.add_argument("--project", required=True)
    registry_init.add_argument("--overwrite", action="store_true")

    registry_check = commands.add_parser("registry-check", help="Validate an asset registry")
    registry_check.add_argument("--project", required=True)

    registry_import = commands.add_parser(
        "registry-import", help="Import reviewed assets from CSV or JSON"
    )
    registry_import.add_argument("--project", required=True)
    registry_import.add_argument("--source", required=True)
    registry_import.add_argument("--replace-existing-ids", action="store_true")

    registry_freeze = commands.add_parser(
        "registry-freeze", help="Freeze an accepted-only registry for one release"
    )
    registry_freeze.add_argument("--project", required=True)
    registry_freeze.add_argument("--release-id", required=True)
    registry_freeze.add_argument("--output", required=True)

    qa = commands.add_parser("qa", help="Build the current project quality report")
    qa.add_argument("--project", required=True)

    gate_evaluate = commands.add_parser(
        "gate-evaluate",
        help="Evaluate an immutable Quality Gate v2 evidence package",
    )
    gate_evaluate.add_argument("--project", required=True)
    gate_evaluate.add_argument("--release-id", required=True)
    gate_evaluate.add_argument("--gate-id", required=True)
    gate_evaluate.add_argument(
        "--check", action="append", default=[], help="Repeat CHECK_ID=REPORT_PATH"
    )
    gate_evaluate.add_argument(
        "--artifact", action="append", default=[], help="Repeat ROLE=ARTIFACT_PATH"
    )
    gate_evaluate.add_argument("--previous-gate")
    gate_evaluate.add_argument("--registry")
    gate_evaluate.add_argument("--waiver", action="append", default=[])
    gate_evaluate.add_argument("--approval", action="append", default=[])
    gate_evaluate.add_argument("--allow-waiver", action="append", default=[])
    gate_evaluate.add_argument("--require-approval", action="append", default=[])

    gate_validate = commands.add_parser(
        "gate-validate", help="Recompute Quality Gate v2 hashes and chain references"
    )
    gate_validate.add_argument("--path", required=True)

    web_release = commands.add_parser(
        "web-release-config",
        help="Build a fail-closed Web config bound to model and registry hashes",
    )
    web_release.add_argument("--project", required=True)
    web_release.add_argument("--release-id", required=True)
    web_release.add_argument("--title", required=True)
    web_release.add_argument("--model", required=True)
    web_release.add_argument("--registry", required=True)
    web_release.add_argument("--output", required=True)
    web_release.add_argument("--model-url", required=True)
    web_release.add_argument("--registry-url", required=True)

    benchmark_init = commands.add_parser(
        "benchmark-init", help="Create a paper benchmark protocol and annotation skeleton"
    )
    benchmark_init.add_argument("target")
    benchmark_init.add_argument("--dataset-id", required=True)
    benchmark_init.add_argument("--scene-id", required=True)

    benchmark_validate = commands.add_parser(
        "benchmark-validate", help="Validate a benchmark JSON document"
    )
    benchmark_validate.add_argument("--path", required=True)

    benchmark_check = commands.add_parser(
        "benchmark-check", help="Validate benchmark documents and cross-file references"
    )
    benchmark_check.add_argument("--root", required=True)

    benchmark_freeze = commands.add_parser(
        "benchmark-freeze", help="Freeze benchmark inputs, protocol and split hashes"
    )
    benchmark_freeze.add_argument("--root", required=True)
    benchmark_freeze.add_argument("--full-hash", action="store_true")

    benchmark_evaluate = commands.add_parser(
        "benchmark-evaluate", help="Compute unified paper metrics from a frozen evaluation input"
    )
    benchmark_evaluate.add_argument("--input", required=True)
    benchmark_evaluate.add_argument("--output", required=True)

    benchmark_holdout = commands.add_parser(
        "benchmark-split-point-holdout",
        help="Split complete spatial voxels into model-building and held-out point clouds",
    )
    benchmark_holdout.add_argument("--project", required=True)
    benchmark_holdout.add_argument("--root", required=True)
    benchmark_holdout.add_argument("--segment", action="append", required=True)
    benchmark_holdout.add_argument("--output-name", default="point_holdout_v1")
    benchmark_holdout.add_argument("--fraction", type=float, default=0.20)
    benchmark_holdout.add_argument("--voxel-size-m", type=float, default=0.50)
    benchmark_holdout.add_argument("--seed", type=int, default=20260827)

    benchmark_holdout_check = commands.add_parser(
        "benchmark-check-point-holdout",
        help="Verify holdout hashes, point counts and complementary splits",
    )
    benchmark_holdout_check.add_argument("--manifest", required=True)

    benchmark_nested_holdout = commands.add_parser(
        "benchmark-split-point-sources",
        help="Create a nested spatial holdout from already-cropped outer-train clouds",
    )
    benchmark_nested_holdout.add_argument(
        "--source", action="append", required=True, help="Repeat SEGMENT=LAS_OR_LAZ"
    )
    benchmark_nested_holdout.add_argument("--output-dir", required=True)
    benchmark_nested_holdout.add_argument("--holdout-id", default="nested_point_holdout_v1")
    benchmark_nested_holdout.add_argument("--fraction", type=float, default=0.25)
    benchmark_nested_holdout.add_argument("--voxel-size-m", type=float, default=0.50)
    benchmark_nested_holdout.add_argument("--seed", type=int, default=20260828)

    benchmark_rail_holdout = commands.add_parser(
        "benchmark-evaluate-rail-holdout",
        help="Measure split repeatability and model-to-held-out-point support",
    )
    benchmark_rail_holdout.add_argument(
        "--train-report", action="append", required=True, help="Repeat SEGMENT=REPORT"
    )
    benchmark_rail_holdout.add_argument(
        "--holdout-report", action="append", required=True, help="Repeat SEGMENT=REPORT"
    )
    benchmark_rail_holdout.add_argument(
        "--holdout-cloud", action="append", required=True, help="Repeat SEGMENT=LAS_OR_LAZ"
    )
    benchmark_rail_holdout.add_argument("--holdout-manifest", required=True)
    benchmark_rail_holdout.add_argument("--output", required=True)
    benchmark_rail_holdout.add_argument("--maximum-match-distance-m", type=float, default=0.30)
    benchmark_rail_holdout.add_argument("--core-length-m", type=float, default=50.0)
    benchmark_rail_holdout.add_argument("--sample-step-m", type=float, default=0.25)

    benchmark_topology_ablation = commands.add_parser(
        "benchmark-evaluate-topology-ablation",
        help="Quantify cross-segment checks absent from a local-only track assembly",
    )
    benchmark_topology_ablation.add_argument("--graph", required=True)
    benchmark_topology_ablation.add_argument("--audit", required=True)
    benchmark_topology_ablation.add_argument("--output", required=True)

    benchmark_defect_injection = commands.add_parser(
        "benchmark-evaluate-defect-injection",
        help="Inject known TrackGraph faults and measure quality-gate detection",
    )
    benchmark_defect_injection.add_argument("--graph", required=True)
    benchmark_defect_injection.add_argument("--settings", required=True)
    benchmark_defect_injection.add_argument("--output", required=True)

    benchmark_open3d = commands.add_parser(
        "benchmark-detect-open3d-rails",
        help="Run a generic Open3D RANSAC/DBSCAN rail candidate baseline",
    )
    benchmark_open3d.add_argument("--source", required=True)
    benchmark_open3d.add_argument("--reference-report", required=True)
    benchmark_open3d.add_argument("--segment", required=True)
    benchmark_open3d.add_argument("--output", required=True)
    benchmark_open3d.add_argument("--voxel-size-m", type=float, default=0.08)
    benchmark_open3d.add_argument("--plane-distance-m", type=float, default=0.035)
    benchmark_open3d.add_argument("--dbscan-eps-m", type=float, default=0.12)
    benchmark_open3d.add_argument("--dbscan-min-points", type=int, default=8)
    benchmark_open3d.add_argument("--longitudinal-scale", type=float, default=0.02)
    benchmark_open3d.add_argument("--minimum-cluster-length-m", type=float, default=15.0)
    benchmark_open3d.add_argument("--maximum-cluster-width-m", type=float, default=0.35)
    benchmark_open3d.add_argument("--maximum-input-points", type=int, default=2_000_000)
    benchmark_open3d.add_argument("--seed", type=int, default=20260827)

    benchmark_gislab = commands.add_parser(
        "benchmark-normalize-gislab-rails",
        help="Normalize GISLab RailTrack point output into the common rail schema",
    )
    benchmark_gislab.add_argument("--source", required=True)
    benchmark_gislab.add_argument("--reference-report", required=True)
    benchmark_gislab.add_argument("--segment", required=True)
    benchmark_gislab.add_argument("--output", required=True)
    benchmark_gislab.add_argument("--cross-bin-m", type=float, default=0.025)
    benchmark_gislab.add_argument("--smoothing-sigma-m", type=float, default=0.05)
    benchmark_gislab.add_argument("--minimum-peak-spacing-m", type=float, default=0.40)
    benchmark_gislab.add_argument("--assignment-radius-m", type=float, default=0.14)
    benchmark_gislab.add_argument("--minimum-cluster-length-m", type=float, default=15.0)
    benchmark_gislab.add_argument("--minimum-cluster-points", type=int, default=25)
    benchmark_gislab.add_argument("--maximum-input-points", type=int, default=2_000_000)
    benchmark_gislab.add_argument(
        "--source-coordinate-space",
        choices=["world_xy", "corridor_local_xy"],
        default="world_xy",
    )

    benchmark_gislab_prepare = commands.add_parser(
        "benchmark-prepare-gislab-input",
        help="Apply the common frozen frame and height envelope before RailTrack",
    )
    benchmark_gislab_prepare.add_argument("--source", required=True)
    benchmark_gislab_prepare.add_argument("--reference-report", required=True)
    benchmark_gislab_prepare.add_argument("--segment", required=True)
    benchmark_gislab_prepare.add_argument("--output", required=True)
    benchmark_gislab_prepare.add_argument("--manifest", required=True)
    benchmark_gislab_prepare.add_argument(
        "--coordinate-space",
        choices=["world_xy", "corridor_local_xy"],
        default="world_xy",
    )

    benchmark_gislab_empty = commands.add_parser(
        "benchmark-record-gislab-no-detection",
        help="Record a proven upstream zero-pair empty-result crash without adding geometry",
    )
    benchmark_gislab_empty.add_argument("--run-manifest", required=True)
    benchmark_gislab_empty.add_argument("--reference-report", required=True)
    benchmark_gislab_empty.add_argument("--segment", required=True)
    benchmark_gislab_empty.add_argument("--output", required=True)
    benchmark_gislab_empty.add_argument(
        "--source-coordinate-space",
        choices=["world_xy", "corridor_local_xy"],
        default="corridor_local_xy",
    )

    benchmark_gislab_timeout = commands.add_parser(
        "benchmark-record-gislab-timeout",
        help="Record a resource-capped GISLab run as a non-result without geometry",
    )
    benchmark_gislab_timeout.add_argument("--run-manifest", required=True)
    benchmark_gislab_timeout.add_argument("--reference-report", required=True)
    benchmark_gislab_timeout.add_argument("--segment", required=True)
    benchmark_gislab_timeout.add_argument("--output", required=True)
    benchmark_gislab_timeout.add_argument(
        "--source-coordinate-space",
        choices=["world_xy", "corridor_local_xy"],
        default="corridor_local_xy",
    )

    benchmark_holdout_comparison = commands.add_parser(
        "benchmark-compare-rail-holdout",
        help="Compare two methods evaluated on the same frozen rail holdout",
    )
    benchmark_holdout_comparison.add_argument("--baseline", required=True)
    benchmark_holdout_comparison.add_argument("--method", required=True)
    benchmark_holdout_comparison.add_argument("--baseline-name", default="baseline")
    benchmark_holdout_comparison.add_argument("--method-name", default="method")
    benchmark_holdout_comparison.add_argument("--output", required=True)

    benchmark_holdout_bootstrap = commands.add_parser(
        "benchmark-bootstrap-rail-holdout",
        help="Bootstrap paired rail metrics over whole spatial segments",
    )
    benchmark_holdout_bootstrap.add_argument("--baseline", required=True)
    benchmark_holdout_bootstrap.add_argument("--method", required=True)
    benchmark_holdout_bootstrap.add_argument("--output", required=True)
    benchmark_holdout_bootstrap.add_argument("--iterations", type=int, default=10_000)
    benchmark_holdout_bootstrap.add_argument("--seed", type=int, default=20260827)
    benchmark_holdout_bootstrap.add_argument("--minimum-confirmatory-blocks", type=int, default=10)

    benchmark_vertical_fit_selection = commands.add_parser(
        "benchmark-select-rail-vertical-fit",
        help="Select a rail-height estimator on a nested outer-train validation split",
    )
    benchmark_vertical_fit_selection.add_argument(
        "--variant",
        action="append",
        required=True,
        help="Repeat NAME=EVALUATION=SETTINGS",
    )
    benchmark_vertical_fit_selection.add_argument("--baseline", required=True)
    benchmark_vertical_fit_selection.add_argument("--output", required=True)

    benchmark_vertical_followup = commands.add_parser(
        "benchmark-audit-rail-vertical-followup",
        help="Audit a post-holdout rail-height correction without overstating evidence",
    )
    benchmark_vertical_followup.add_argument("--baseline", required=True)
    benchmark_vertical_followup.add_argument("--initial-selection", required=True)
    benchmark_vertical_followup.add_argument("--trigger-followup", required=True)
    benchmark_vertical_followup.add_argument("--nested-correction", required=True)
    benchmark_vertical_followup.add_argument("--final-followup", required=True)
    benchmark_vertical_followup.add_argument("--corrected-settings", required=True)
    benchmark_vertical_followup.add_argument("--output", required=True)

    rail_regression = commands.add_parser(
        "rail-regression-audit",
        help="Compare two rail candidate sets on identical production point clouds",
    )
    rail_regression.add_argument(
        "--baseline-report", action="append", required=True, help="Repeat SEGMENT=REPORT"
    )
    rail_regression.add_argument(
        "--candidate-report", action="append", required=True, help="Repeat SEGMENT=REPORT"
    )
    rail_regression.add_argument(
        "--support-cloud",
        action="append",
        help="Optional common SEGMENT=LAS_OR_LAZ support source for both report sets",
    )
    rail_regression.add_argument("--output", required=True)
    rail_regression.add_argument("--core-length-m", type=float, default=50.0)
    rail_regression.add_argument("--sample-step-m", type=float, default=0.25)
    rail_regression.add_argument("--maximum-cross-change-m", type=float, default=0.001)
    rail_regression.add_argument("--allow-additional-lines", action="store_true")
    rail_regression.add_argument("--per-segment-support-p90-tolerance-m", type=float, default=0.005)
    rail_regression.add_argument(
        "--maximum-additional-line-support-p90-m", type=float, default=0.075
    )
    rail_regression.add_argument("--maximum-matched-line-support-p90-m", type=float, default=0.060)

    benchmark_comparison_figure = commands.add_parser(
        "benchmark-render-rail-comparison",
        help="Overlay two prediction sets on neutral raw-point cross sections",
    )
    benchmark_comparison_figure.add_argument("--evidence-manifest", required=True)
    benchmark_comparison_figure.add_argument(
        "--baseline-report", action="append", required=True, help="Repeat SEGMENT=REPORT"
    )
    benchmark_comparison_figure.add_argument(
        "--method-report", action="append", required=True, help="Repeat SEGMENT=REPORT"
    )
    benchmark_comparison_figure.add_argument("--output-dir", required=True)

    benchmark_experiment_lock = commands.add_parser(
        "benchmark-lock-experiment",
        help="Hash-bind an executable method variant before ground-truth inspection",
    )
    benchmark_experiment_lock.add_argument("--root", required=True)
    benchmark_experiment_lock.add_argument("--experiment", required=True)
    benchmark_experiment_lock.add_argument(
        "--binding", action="append", required=True, help="Repeat ROLE=FILE"
    )
    benchmark_experiment_lock.add_argument("--freeze-manifest")
    benchmark_experiment_lock.add_argument("--output")

    benchmark_annotation = commands.add_parser(
        "benchmark-make-vertical-tasks",
        help="Build two independently shuffled blind-review CSVs for vertical candidates",
    )
    benchmark_annotation.add_argument("--root", required=True)
    benchmark_annotation.add_argument(
        "--source", action="append", required=True, help="Repeat SEGMENT=FEATURE_REPORT"
    )
    benchmark_annotation.add_argument("--output-name", default="vertical_candidates_blind_v1")
    benchmark_annotation.add_argument("--seed", type=int, default=20260826)

    benchmark_annotation_check = commands.add_parser(
        "benchmark-check-vertical-tasks",
        help="Verify blind-review task equality, empty labels, source hashes and evidence files",
    )
    benchmark_annotation_check.add_argument("--path", required=True)

    benchmark_rail_evidence = commands.add_parser(
        "benchmark-render-rail-truth-evidence",
        help="Render model-free rail cross-sections and plan strips from the raw cloud",
    )
    benchmark_rail_evidence.add_argument("--project", required=True)
    benchmark_rail_evidence.add_argument("--root", required=True)
    benchmark_rail_evidence.add_argument("--scene-id", required=True)
    benchmark_rail_evidence.add_argument("--start-m", required=True, type=float)
    benchmark_rail_evidence.add_argument("--end-m", required=True, type=float)
    benchmark_rail_evidence.add_argument("--spacing-m", type=float, default=2.0)
    benchmark_rail_evidence.add_argument("--slice-half-width-m", type=float, default=0.30)
    benchmark_rail_evidence.add_argument("--cross-min-m", type=float, default=-20.0)
    benchmark_rail_evidence.add_argument("--cross-max-m", type=float, default=20.0)
    benchmark_rail_evidence.add_argument("--z-min-m", type=float)
    benchmark_rail_evidence.add_argument("--z-max-m", type=float)
    benchmark_rail_evidence.add_argument("--output-name", default="rail_neutral_evidence_v1")

    benchmark_rail_annotation = commands.add_parser(
        "benchmark-make-rail-truth-tasks",
        help="Build double-blind rail geometry tasks from neutral raw-point evidence",
    )
    benchmark_rail_annotation.add_argument("--root", required=True)
    benchmark_rail_annotation.add_argument("--evidence-manifest", action="append", required=True)
    benchmark_rail_annotation.add_argument("--output-name", default="rail_geometry_blind_v1")
    benchmark_rail_annotation.add_argument("--seed", type=int, default=20260827)

    benchmark_rail_annotation_check = commands.add_parser(
        "benchmark-check-rail-truth-tasks",
        help="Verify neutral evidence, empty labels, source hashes and reviewer task sets",
    )
    benchmark_rail_annotation_check.add_argument("--path", required=True)

    benchmark_prediction_lock = commands.add_parser(
        "benchmark-lock-vertical-predictions",
        help="Freeze legacy vertical classifications before independent ground truth",
    )
    benchmark_prediction_lock.add_argument("--root", required=True)
    benchmark_prediction_lock.add_argument("--annotation-package", required=True)
    benchmark_prediction_lock.add_argument(
        "--source", action="append", required=True, help="Repeat SEGMENT=CLASSIFICATION_REPORT"
    )
    benchmark_prediction_lock.add_argument(
        "--output-name", default="vertical_predictions_legacy_hitl_v1"
    )

    benchmark_review_compare = commands.add_parser(
        "benchmark-compare-vertical-reviews",
        help="Measure two completed blind reviews and create an adjudication queue",
    )
    benchmark_review_compare.add_argument("--annotation-package", required=True)
    benchmark_review_compare.add_argument("--output-name", default="vertical_review_comparison_v1")

    mesh_audit = commands.add_parser("mesh-audit", help="Audit an OBJ before Blender/UE import")
    mesh_audit.add_argument("--path", required=True)
    mesh_audit.add_argument("--area-tolerance", type=float, default=1e-12)

    support_gate = commands.add_parser(
        "audit-support-relationships",
        help="Fail closed on candidate bearing/support claims without a measured owner",
    )
    support_gate.add_argument("--registry", required=True)
    support_gate.add_argument("--patch", required=True)
    support_gate.add_argument("--output", required=True)
    support_gate.add_argument("--maximum-contact-residual-m", type=float, default=0.05)
    support_gate.add_argument("--minimum-direct-evidence-kinds", type=int, default=1)
    support_gate.add_argument("--allow-unmodeled-owner", action="store_true")

    rapid_audit = commands.add_parser(
        "audit-rapid-candidate",
        help="Run the model-first internal candidate gate without relaxing artifact integrity",
    )
    rapid_audit.add_argument("--release-directory", required=True)
    rapid_audit.add_argument("--manifest", required=True)
    rapid_audit.add_argument("--delivery-gate", required=True)
    rapid_audit.add_argument("--lineage")
    rapid_audit.add_argument("--expected-release-id")
    rapid_audit.add_argument("--expected-parent-release-id")
    rapid_audit.add_argument("--output", required=True)

    auto_review = commands.add_parser(
        "auto-review-fixed-views",
        help="Verify every fixed-view render and create a small risk-ranked spot-check queue",
    )
    auto_review.add_argument("--manifest", required=True)
    auto_review.add_argument("--output", required=True)
    auto_review.add_argument("--maximum-spot-checks", type=int, default=6)

    safety = commands.add_parser("safety-check", help="Check a repository before GitHub push")
    safety.add_argument("--root", default=".")
    safety.add_argument("--max-mb", type=float, default=10.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init":
            target = Path(args.target)
            project_id = args.project_id or target.name.lower().replace(" ", "-")
            path = initialize_project(target, project_id, args.name)
            _print({"created": str(path), "next": f"railway-recon validate --project {path}"})
            return
        if args.command == "validate":
            value = load_json(Path(args.project))
            errors = validate_project_value(value)
            _print(
                {
                    "project": str(Path(args.project).resolve()),
                    "valid": not errors,
                    "errors": errors,
                }
            )
            if errors:
                raise SystemExit(2)
            return
        if args.command == "audit":
            code = _project_command(
                args, "audit", lambda project: audit_project(project, full_hash=args.full_hash)
            )
        elif args.command == "prepare-inputs":
            code = _project_command(
                args,
                "prepare_inputs",
                lambda project: prepare_input_sources(
                    project,
                    full_hash=args.full_hash,
                    camera_bbox_margin_m=args.camera_bbox_margin_m,
                ),
            )
        elif args.command == "plan-segments":
            code = _project_command(args, "plan_segments", plan_segments)
        elif args.command == "segment":
            code = _project_command(
                args,
                "segment",
                lambda project: crop_segments(project, set(args.ids or []), args.overwrite),
            )
        elif args.command == "segment-context":
            code = _project_command(
                args,
                "segment_context",
                lambda project: crop_segment_context_sources(project, args.segment, args.overwrite),
            )
        elif args.command == "full-corridor-plan":
            code = _project_command(args, "full_corridor_plan", build_full_corridor_plan)
        elif args.command == "full-corridor-run":
            code = _project_command(
                args,
                "full_corridor_run",
                lambda project: run_full_corridor(
                    project,
                    through=args.through,
                    continue_on_error=not args.fail_fast,
                ),
            )
        elif args.command == "run-global-scene-candidate":
            code = _project_command(
                args,
                "run_global_scene_candidate_pipeline",
                lambda project: run_global_scene_candidate_pipeline(
                    project,
                    args.track_graph,
                    args.track_graph_audit,
                    args.output_root,
                    args.output_name,
                    station_assets_obj=args.station_assets_obj,
                    include_inferred_track_gaps=args.include_inferred_track_gaps,
                    maximum_inferred_track_gap_m=args.maximum_inferred_track_gap_m,
                    overwrite=args.overwrite,
                ),
            )
        elif args.command == "detect-rails":
            code = _project_command(
                args,
                "detect_rails",
                lambda project: detect_rail_candidates(
                    project,
                    args.segment,
                    args.overwrite,
                    args.source,
                    args.settings,
                    args.report,
                ),
            )
        elif args.command == "detect-linear":
            code = _project_command(
                args,
                "detect_linear",
                lambda project: detect_linear_candidates(project, args.segment, args.overwrite),
            )
        elif args.command == "classify-vertical":
            code = _project_command(
                args,
                "classify_vertical",
                lambda project: analyze_vertical_hypotheses(
                    project,
                    args.segment,
                    overwrite=args.overwrite,
                    linear_report_path=args.linear_report,
                    rail_report_path=args.rail_report,
                    settings_path=args.settings,
                ),
            )
        elif args.command == "analyze-canopy":
            code = _project_command(
                args,
                "analyze_canopy",
                lambda project: analyze_canopy_structure(
                    project,
                    args.segment,
                    overwrite=args.overwrite,
                    vertical_report_path=args.vertical_report,
                    settings_path=args.settings,
                ),
            )
        elif args.command == "canopy-photo-evidence":
            code = _project_command(
                args,
                "canopy_photo_evidence",
                lambda project: analyze_canopy_photo_evidence(
                    project,
                    args.segment,
                    args.projection_consensus,
                    overwrite=args.overwrite,
                    canopy_report_path=args.canopy_report,
                    vertical_report_path=args.vertical_report,
                    settings_path=args.settings,
                ),
            )
        elif args.command == "analyze-platform":
            code = _project_command(
                args,
                "analyze_platform",
                lambda project: analyze_platform_surface(
                    project,
                    args.segment,
                    overwrite=args.overwrite,
                    vertical_report_path=args.vertical_report,
                    settings_path=args.settings,
                ),
            )
        elif args.command == "platform-photo-evidence":
            code = _project_command(
                args,
                "platform_photo_evidence",
                lambda project: analyze_platform_photo_evidence(
                    project,
                    args.segment,
                    args.projection_consensus,
                    overwrite=args.overwrite,
                    platform_report_path=args.platform_report,
                    settings_path=args.settings,
                ),
            )
        elif args.command == "build-platform-mesh":
            code = _project_command(
                args,
                "build_platform_mesh",
                lambda project: build_platform_mesh(
                    project,
                    args.segment,
                    args.mesh_gate,
                    overwrite=args.overwrite,
                    platform_report_path=args.platform_report,
                    settings_path=args.settings,
                ),
            )
        elif args.command == "create-platform-candidate-gate":
            code = _project_command(
                args,
                "create_platform_candidate_gate",
                lambda project: evaluate_platform_candidate_gate(
                    project,
                    args.segment,
                    args.platform_report,
                    args.output,
                    component_id=args.component_id,
                    ownership_plan_value=args.ownership_plan,
                    gap_point_evidence_value=args.gap_point_evidence,
                    overwrite=args.overwrite,
                ),
                required_status={"pass_candidate"},
            )
        elif args.command == "analyze-platform-gap-points":
            code = _project_command(
                args,
                "analyze_platform_gap_points",
                lambda project: analyze_platform_gap_point_evidence(
                    project,
                    args.segment,
                    args.platform_report,
                    args.output,
                    settings_value=args.settings,
                    overwrite=args.overwrite,
                ),
            )
        elif args.command == "audit-platform-interfaces":
            code = _project_command(
                args,
                "audit_platform_interfaces",
                lambda project: audit_platform_interfaces(
                    project,
                    args.segment,
                    mesh_build_report_path=args.mesh_build_report,
                    platform_report_path=args.platform_report,
                    vertical_report_path=args.vertical_report,
                    canopy_report_path=args.canopy_report,
                    settings_path=args.settings,
                    overwrite=args.overwrite,
                ),
            )
        elif args.command == "reconstruct-opposite-platform":
            code = _project_command(
                args,
                "reconstruct_opposite_platform",
                lambda project: reconstruct_opposite_platform(
                    project,
                    args.segment,
                    args.point_cloud,
                    args.vertical_report,
                    args.output_dir,
                    side=args.side,
                    owned_interval_m=(
                        (args.minimum_longitudinal_m, args.maximum_longitudinal_m)
                        if args.minimum_longitudinal_m is not None
                        and args.maximum_longitudinal_m is not None
                        else None
                    ),
                    ownership_plan_path=args.ownership_plan,
                    settings_path=args.settings,
                    overwrite=args.overwrite,
                ),
            )
        elif args.command == "vertical-conflict-photo-evidence":
            code = _project_command(
                args,
                "vertical_conflict_photo_evidence",
                lambda project: analyze_vertical_conflict_photo_evidence(
                    project,
                    args.segment,
                    args.projection_consensus,
                    interface_report_path=args.interface_report,
                    vertical_report_path=args.vertical_report,
                    settings_path=args.settings,
                    overwrite=args.overwrite,
                ),
            )
        elif args.command == "vertical-candidate-photo-evidence":
            code = _project_command(
                args,
                "vertical_candidate_photo_evidence",
                lambda project: analyze_selected_vertical_photo_evidence(
                    project,
                    args.segment,
                    args.candidate,
                    args.projection_consensus,
                    args.output_name,
                    vertical_report_path=args.vertical_report,
                    settings_path=args.settings,
                    overwrite=args.overwrite,
                ),
            )
        elif args.command == "catenary-section-audit":
            code = _project_command(
                args,
                "catenary_section_audit",
                lambda project: audit_catenary_candidate_section(
                    project,
                    args.segment,
                    args.candidate,
                    vertical_report_path=args.vertical_report,
                    output_path=args.output,
                    overwrite=args.overwrite,
                ),
            )
        elif args.command == "catenary-candidate-mesh":
            code = _project_command(
                args,
                "catenary_candidate_mesh",
                lambda project: build_catenary_candidate_mesh(
                    project,
                    args.vertical_report,
                    args.semantic_review,
                    args.section_audit,
                    output_dir=args.output_dir,
                    report_dir=args.report_dir,
                    overwrite=args.overwrite,
                ),
            )
        elif args.command == "catenary-track-clearance-audit":
            code = _project_command(
                args,
                "catenary_track_clearance_audit",
                lambda project: audit_catenary_track_clearance_files(
                    project,
                    args.track_graph,
                    args.section_audit,
                    args.track_build,
                    args.output,
                    overwrite=args.overwrite,
                ),
            )
        elif args.command == "run-corridor-catenary-pipeline":
            code = _project_command(
                args,
                "run_corridor_catenary_pipeline",
                lambda project: run_corridor_catenary_pipeline(
                    project,
                    args.output_dir,
                    args.output_name,
                    settings_value=args.settings,
                    overwrite=args.overwrite,
                ),
            )
        elif args.command == "run-corridor-conductor-pipeline":
            code = _project_command(
                args,
                "run_corridor_conductor_pipeline",
                lambda project: run_corridor_conductor_pipeline(
                    project,
                    args.track_graph,
                    args.output_dir,
                    args.output_name,
                    settings_value=args.settings,
                    overwrite=args.overwrite,
                ),
            )
        elif args.command == "build-segment-ownership-plan":
            segment_frames: dict[str, str] = {}
            for item in args.segment_frame:
                if "=" not in item:
                    raise ValueError("--segment-frame must be SEGMENT_ID=PATH")
                segment_id, frame_path = item.split("=", 1)
                if not segment_id or not frame_path or segment_id in segment_frames:
                    raise ValueError(
                        "Segment frame entries must have unique non-empty IDs and paths"
                    )
                segment_frames[segment_id] = frame_path
            code = _project_command(
                args,
                "build_segment_ownership_plan",
                lambda project: build_segment_ownership_plan(
                    project,
                    segment_frames,
                    args.output,
                    overwrite=args.overwrite,
                ),
            )
        elif args.command == "clip-segment-mesh-ownership":
            code = _project_command(
                args,
                "clip_segment_mesh_ownership",
                lambda project: clip_segment_mesh_ownership(
                    project,
                    args.source_obj,
                    args.source_origin,
                    args.frame_report,
                    args.output_dir,
                    minimum_longitudinal_m=args.minimum_longitudinal_m,
                    maximum_longitudinal_m=args.maximum_longitudinal_m,
                    overwrite=args.overwrite,
                ),
            )
        elif args.command == "audit-owned-mesh-seams":
            segment_meshes: dict[str, str] = {}
            for item in args.segment_mesh:
                if "=" not in item:
                    raise ValueError("--segment-mesh must be SEGMENT_ID=PATH")
                segment_id, mesh_path = item.split("=", 1)
                if not segment_id or not mesh_path or segment_id in segment_meshes:
                    raise ValueError(
                        "Segment mesh entries must have unique non-empty IDs and paths"
                    )
                segment_meshes[segment_id] = mesh_path
            code = _project_command(
                args,
                "audit_owned_mesh_seams",
                lambda project: audit_owned_mesh_seams(
                    project,
                    args.ownership_plan,
                    segment_meshes,
                    args.output,
                    p90_max_m=args.p90_max_m,
                    boundary_tolerance_m=args.boundary_tolerance_m,
                    sample_spacing_m=args.sample_spacing_m,
                    overwrite=args.overwrite,
                ),
            )
        elif args.command == "assemble-candidate-meshes":
            code = _project_command(
                args,
                "assemble_candidate_meshes",
                lambda project: assemble_candidate_meshes(
                    project,
                    args.output_dir,
                    args.output_name,
                    parse_mesh_source_specs(args.source),
                    overwrite=args.overwrite,
                ),
            )
        elif args.command == "run-station-platform-pipeline":
            code = _project_command(
                args,
                "run_station_platform_pipeline",
                lambda project: run_station_platform_pipeline(
                    project,
                    args.ownership_plan,
                    args.segment,
                    args.output_name,
                    overwrite=args.overwrite,
                ),
            )
        elif args.command == "recover-corridor-column-grid":
            vertical_report_items = _assignments(args.vertical_report, "--vertical-report")
            vertical_reports = dict(vertical_report_items)
            if len(vertical_reports) != len(vertical_report_items):
                raise ValueError("--vertical-report segment IDs must be unique")
            code = _project_command(
                args,
                "recover_corridor_column_grid",
                lambda project: recover_corridor_column_grid(
                    project,
                    args.ownership_plan,
                    vertical_reports,
                    args.output,
                    side=args.side,
                    settings_path=args.settings,
                    overwrite=args.overwrite,
                ),
            )
        elif args.command == "validate-corridor-column-grid":
            code = _project_command(
                args,
                "validate_corridor_column_grid",
                lambda project: validate_corridor_column_grid(
                    project,
                    args.grid,
                    args.output,
                    settings_value=args.settings,
                    overwrite=args.overwrite,
                ),
                required_status={
                    "local_point_support_evaluated_geometry_remains_candidate"
                },
            )
        elif args.command == "reconcile-mesh-seams":
            code = _project_command(
                args,
                "reconcile_mesh_seams",
                lambda project: reconcile_mesh_seams(
                    project,
                    args.source_obj,
                    args.source_origin,
                    args.frame_report,
                    args.settings,
                    args.output_dir,
                    registry_path=args.registry,
                    overwrite=args.overwrite,
                ),
            )
        elif args.command == "build-corridor-canopy-candidate":
            code = _project_command(
                args,
                "build_corridor_canopy_candidate",
                lambda project: build_corridor_canopy_candidate(
                    project,
                    args.ownership_plan,
                    args.supported_grid,
                    args.output_name,
                    settings_value=args.settings,
                    overwrite=args.overwrite,
                ),
            )
        elif args.command == "run-side-asset-pipeline":
            code = _project_command(
                args,
                "run_side_asset_pipeline",
                lambda project: run_side_asset_pipeline(
                    project,
                    args.plan,
                    args.output,
                    overwrite=args.overwrite,
                ),
            )
        elif args.command == "build-mesh-object-registry":
            code = _project_command(
                args,
                "build_mesh_object_registry",
                lambda project: build_mesh_object_registry(
                    project,
                    args.obj,
                    args.output,
                    evidence_reference=args.evidence_reference,
                    overwrite=args.overwrite,
                ),
            )
        elif args.command == "analyze-small-platform-assets":
            code = _project_command(
                args,
                "analyze_small_platform_asset_candidates",
                lambda project: analyze_small_platform_asset_candidates(
                    project,
                    args.segment,
                    args.platform_report,
                    args.ownership_plan,
                    overwrite=args.overwrite,
                ),
            )
        elif args.command == "small-asset-photo-evidence":
            code = _project_command(
                args,
                "small_asset_photo_evidence",
                lambda project: analyze_small_asset_photo_evidence(
                    project,
                    args.segment,
                    args.small_asset_report,
                    args.projection_consensus,
                    settings_path=args.settings,
                    overwrite=args.overwrite,
                ),
            )
        elif args.command == "targeted-canopy-recover":
            code = _project_command(
                args,
                "targeted_canopy_recover",
                lambda project: recover_targeted_canopy(
                    project,
                    args.segment,
                    args.semantic_review,
                    overwrite=args.overwrite,
                    vertical_report_path=args.vertical_report,
                    platform_report_path=args.platform_report,
                    settings_path=args.settings,
                    grid_review_path=args.grid_review,
                ),
            )
        elif args.command == "targeted-canopy-photo-evidence":
            code = _project_command(
                args,
                "targeted_canopy_photo_evidence",
                lambda project: targeted_canopy_photo_evidence(
                    project,
                    args.segment,
                    args.recovery_report,
                    args.projection_consensus,
                    settings_path=args.settings,
                    overwrite=args.overwrite,
                ),
            )
        elif args.command == "targeted-canopy-integrate":
            code = _project_command(
                args,
                "targeted_canopy_integrate",
                lambda project: integrate_targeted_canopy_candidate(
                    project,
                    args.segment,
                    args.recovery_report,
                    args.visual_review,
                    args.release_id,
                    approve_candidate_merge=args.approve_candidate_merge,
                    overwrite=args.overwrite,
                ),
                required_status={"candidate_scene_integrated"},
            )
        elif args.command == "build-track":
            code = _project_command(
                args,
                "build_track",
                lambda project: build_parametric_track(project, args.segment, args.overwrite),
            )
        elif args.command == "track-graph-build":
            code = _project_command(
                args,
                "track_graph_build",
                lambda project: build_track_graph(
                    project,
                    _segment_sources(args.source),
                    args.overwrite,
                    args.output,
                    args.audit,
                ),
                required_status={"pass"},
            )
        elif args.command == "track-graph-validate":
            code = _project_command(
                args,
                "track_graph_validate",
                lambda project: validate_track_graph(project, args.graph, args.output),
                required_status={"pass"},
            )
        elif args.command == "track-continuity-recover":
            path = evaluate_targeted_rail_recovery(
                args.graph,
                args.output,
                observation_ids=args.observation_id,
                longitudinal_bin_m=args.longitudinal_bin_m,
                minimum_bin_points=args.minimum_bin_points,
                cross_half_width_m=args.cross_half_width_m,
                vertical_below_m=args.vertical_below_m,
                vertical_above_m=args.vertical_above_m,
                minimum_joint_support_ratio=args.minimum_joint_support_ratio,
                maximum_asymmetric_support_ratio=(args.maximum_asymmetric_support_ratio),
                maximum_internal_joint_gap_m=args.maximum_internal_joint_gap_m,
            )
            report = load_json(path)
            updated_graph = (
                None
                if args.updated_graph is None
                else str(apply_targeted_rail_recovery(args.graph, path, args.updated_graph))
            )
            _print(
                {
                    "report": str(path),
                    "updated_graph": updated_graph,
                    **report["summary"],
                }
            )
            code = 0
        elif args.command == "track-recovery-select":
            code = _project_command(
                args,
                "track_recovery_select",
                lambda project: select_corridor_rail_recoveries(
                    project,
                    args.baseline_graph,
                    args.baseline_audit,
                    args.settings,
                    output_name=args.output_name,
                    overwrite=args.overwrite,
                ),
            )
        elif args.command == "track-recovery-bind-scene":
            path = bind_targeted_recovery_to_scene(
                args.source_glb,
                args.registry,
                args.recovered_graph,
                args.recovery,
                args.output_glb,
                args.output_registry,
                args.report,
                profile=args.profile,
            )
            report = load_json(path)
            _print(
                {
                    "report": str(path),
                    "status": report["status"],
                    "output_glb": report["output_glb"],
                    "geometry_changed": report["geometry_changed"],
                    "observation_assets": report["observation_assets"],
                }
            )
            code = 0
        elif args.command == "build-track-graph-mesh":
            code = _project_command(
                args,
                "build_track_graph_mesh",
                lambda project: build_track_graph_mesh(
                    project,
                    args.graph,
                    args.graph_audit,
                    args.overwrite,
                    args.output_dir,
                    args.report_dir,
                    not args.skip_registry
                    and not args.candidate_only_nonpassing_audit
                    and not args.include_inferred_gap_hypotheses,
                    args.candidate_only_nonpassing_audit,
                    args.include_inferred_gap_hypotheses,
                    args.maximum_inferred_gap_m,
                ),
            )
        elif args.command == "track-review-create":
            code = _project_command(
                args,
                "track_review_create",
                lambda project: {
                    "status": "pending_independent_review",
                    "package": str(
                        create_rail_review_package(
                            project,
                            _segment_sources(args.source),
                            args.output_name,
                        )
                    ),
                },
            )
        elif args.command == "track-review-apply":
            code = _project_command(
                args,
                "track_review_apply",
                lambda project: apply_rail_review_package(project, args.path),
                required_status={"accepted"},
            )
        elif args.command == "track-review-owner-override":
            code = _project_command(
                args,
                "track_review_owner_override",
                lambda project: apply_rail_owner_override(
                    project,
                    args.path,
                    args.approved_by,
                    args.reason,
                ),
                required_status={"accepted"},
            )
        elif args.command == "track-pair-curate":
            code = _project_command(
                args,
                "track_pair_curate",
                lambda project: curate_rail_pairs(
                    project,
                    args.selection,
                    args.output_name,
                ),
                required_status={"accepted"},
            )
        elif args.command == "build-reviewed-scene":
            code = _project_command(
                args,
                "build_reviewed_scene",
                lambda project: build_reviewed_scene(
                    project, Path(args.layout).resolve(), args.overwrite
                ),
            )
        elif args.command == "calibrate-projection":
            code = _project_command(
                args,
                "calibrate_projection",
                lambda project: calibrate_projection(
                    project, args.segment, args.camera_index, args.overwrite
                ),
            )
        elif args.command == "calibrate-projection-consensus":
            code = _project_command(
                args,
                "calibrate_projection_consensus",
                lambda project: calibrate_projection_consensus(
                    project,
                    _projection_samples(args.sample),
                    args.output_name,
                    args.overwrite,
                ),
            )
        elif args.command == "seam-rail-compare":
            code = _project_command(
                args,
                "seam_rail_compare",
                lambda project: compare_seam_rail_reports(
                    project,
                    Path(args.left_report),
                    Path(args.right_report),
                    Path(args.output),
                ),
            )
        elif args.command == "registry-init":
            code = _project_command(
                args,
                "registry_init",
                lambda project: initialize_registry(project, args.overwrite),
            )
        elif args.command == "registry-check":
            project = load_project(args.project)
            registry, errors = validate_registry_file(project.workspace_path("asset_registry"))
            _print({"valid": not errors, "errors": errors, "summary": summarize_registry(registry)})
            code = 0 if not errors else 2
        elif args.command == "registry-import":
            code = _project_command(
                args,
                "registry_import",
                lambda project: import_registry_records(
                    project, Path(args.source).resolve(), args.replace_existing_ids
                ),
            )
        elif args.command == "registry-freeze":
            code = _project_command(
                args,
                "registry_freeze",
                lambda project: freeze_release_registry(project, args.release_id, args.output),
            )
        elif args.command == "qa":
            code = _project_command(args, "qa", quality_report)
        elif args.command == "gate-evaluate":
            project = load_project(args.project)
            result_path, result = evaluate_quality_gate(
                project,
                args.release_id,
                args.gate_id,
                _assignments(args.check, "--check"),
                _assignments(args.artifact, "--artifact"),
                previous_gate=args.previous_gate,
                registry=args.registry,
                waiver_sources=args.waiver,
                approval_sources=args.approval,
                allow_waiver=set(args.allow_waiver),
                required_approval_roles=set(args.require_approval),
            )
            manifest = write_run_manifest(
                project,
                "gate_evaluate",
                "completed",
                outputs=[str(result_path)],
                metrics={"gate_id": args.gate_id, "gate_status": result["status"]},
            )
            _print({**result, "gate_result": str(result_path), "run_manifest": str(manifest)})
            code = 0 if result["status"] in {"PASS", "PASS_WITH_WAIVER"} else 5
        elif args.command == "gate-validate":
            result = validate_gate_result(args.path)
            _print(result)
            code = 0 if result["valid"] else 5
        elif args.command == "web-release-config":
            code = _project_command(
                args,
                "web_release_config",
                lambda project: build_web_acceptance_config(
                    project,
                    args.release_id,
                    args.title,
                    args.model,
                    args.registry,
                    args.output,
                    model_url=args.model_url,
                    registry_url=args.registry_url,
                ),
            )
        elif args.command == "benchmark-init":
            root = initialize_benchmark(args.target, args.dataset_id, args.scene_id)
            _print(
                {
                    "created": str(root),
                    "next": (
                        "railway-recon benchmark-validate --path "
                        f"{root / 'manifests' / 'dataset-manifest.json'}"
                    ),
                }
            )
            code = 0
        elif args.command == "benchmark-validate":
            value, errors = validate_benchmark_file(args.path)
            _print(
                {
                    "path": str(Path(args.path).resolve()),
                    "schema_version": value.get("schema_version"),
                    "valid": not errors,
                    "errors": errors,
                }
            )
            code = 0 if not errors else 2
        elif args.command == "benchmark-check":
            values, errors = validate_benchmark_root(args.root)
            _print(
                {
                    "root": str(Path(args.root).resolve()),
                    "valid": not errors,
                    "documents": sorted(values),
                    "errors": errors,
                }
            )
            code = 0 if not errors else 2
        elif args.command == "benchmark-freeze":
            path = freeze_benchmark(args.root, args.full_hash)
            value = load_json(path)
            _print(
                {
                    "manifest": str(path),
                    "status": value["status"],
                    "input_count": len(value["inputs"]),
                    "warning_count": len(value["warnings"]),
                }
            )
            code = 0 if value["status"] == "completed" else 2
        elif args.command == "benchmark-evaluate":
            path = evaluate_benchmark_file(args.input, args.output)
            report = load_json(path)
            _print(
                {
                    "report": str(path),
                    "experiment_id": report["experiment_id"],
                    "sample_counts": report["sample_counts"],
                }
            )
            code = 0
        elif args.command == "benchmark-split-point-holdout":
            path = split_point_cloud_holdout(
                load_project(args.project),
                args.root,
                set(args.segment),
                output_name=args.output_name,
                holdout_fraction=args.fraction,
                voxel_size_m=args.voxel_size_m,
                seed=args.seed,
            )
            manifest = load_json(path)
            _print(
                {
                    "holdout_manifest": str(path),
                    "segment_count": len(manifest["segments"]),
                    "strategy": manifest["strategy"],
                    "voxel_size_m": manifest["voxel_size_m"],
                    "requested_holdout_fraction": manifest["requested_holdout_fraction"],
                }
            )
            code = 0
        elif args.command == "benchmark-check-point-holdout":
            report = validate_point_cloud_holdout(args.manifest)
            _print(report)
            code = 0 if report["passed"] else 2
        elif args.command == "benchmark-split-point-sources":
            path = split_existing_point_cloud_holdout(
                _assignments(args.source, "--source"),
                args.output_dir,
                holdout_id=args.holdout_id,
                holdout_fraction=args.fraction,
                voxel_size_m=args.voxel_size_m,
                seed=args.seed,
            )
            manifest = load_json(path)
            _print(
                {
                    "holdout_manifest": str(path),
                    "segment_count": len(manifest["segments"]),
                    "strategy": manifest["strategy"],
                }
            )
            code = 0
        elif args.command == "benchmark-evaluate-rail-holdout":
            path = evaluate_rail_holdout(
                _assignments(args.train_report, "--train-report"),
                _assignments(args.holdout_report, "--holdout-report"),
                _assignments(args.holdout_cloud, "--holdout-cloud"),
                args.holdout_manifest,
                args.output,
                maximum_match_distance_m=args.maximum_match_distance_m,
                core_length_m=args.core_length_m,
                sample_step_m=args.sample_step_m,
            )
            report = load_json(path)
            _print({"report": str(path), **report["summary"]})
            code = 0
        elif args.command == "benchmark-evaluate-topology-ablation":
            path = evaluate_topology_ablation(args.graph, args.audit, args.output)
            report = load_json(path)
            _print({"report": str(path), **report["delta"]})
            code = 0
        elif args.command == "benchmark-evaluate-defect-injection":
            path = evaluate_defect_injection(args.graph, args.settings, args.output)
            report = load_json(path)
            _print({"report": str(path), **report["summary"]})
            code = 0
        elif args.command == "benchmark-detect-open3d-rails":
            path = detect_open3d_rail_baseline(
                args.source,
                args.reference_report,
                args.segment,
                args.output,
                voxel_size_m=args.voxel_size_m,
                plane_distance_m=args.plane_distance_m,
                dbscan_eps_m=args.dbscan_eps_m,
                dbscan_min_points=args.dbscan_min_points,
                longitudinal_scale=args.longitudinal_scale,
                minimum_cluster_length_m=args.minimum_cluster_length_m,
                maximum_cluster_width_m=args.maximum_cluster_width_m,
                maximum_input_points=args.maximum_input_points,
                source_coordinate_space=args.source_coordinate_space,
                seed=args.seed,
            )
            report = load_json(path)
            _print(
                {
                    "report": str(path),
                    "rail_pair_count": report["rail_pair_count"],
                    "eligible_cluster_count": report["eligible_cluster_count"],
                    "downsampled_point_count": report["downsampled_point_count"],
                }
            )
            code = 0
        elif args.command == "benchmark-normalize-gislab-rails":
            path = normalize_gislab_railtrack_points(
                args.source,
                args.reference_report,
                args.segment,
                args.output,
                cross_bin_m=args.cross_bin_m,
                smoothing_sigma_m=args.smoothing_sigma_m,
                minimum_peak_spacing_m=args.minimum_peak_spacing_m,
                assignment_radius_m=args.assignment_radius_m,
                minimum_cluster_length_m=args.minimum_cluster_length_m,
                minimum_cluster_points=args.minimum_cluster_points,
                maximum_input_points=args.maximum_input_points,
            )
            report = load_json(path)
            _print(
                {
                    "report": str(path),
                    "rail_pair_count": report["rail_pair_count"],
                    "eligible_cluster_count": report["eligible_cluster_count"],
                    "normalized_point_count": report["normalized_point_count"],
                }
            )
            code = 0
        elif args.command == "benchmark-prepare-gislab-input":
            path, manifest = prepare_gislab_railtrack_input(
                args.source,
                args.reference_report,
                args.segment,
                args.output,
                args.manifest,
                coordinate_space=args.coordinate_space,
            )
            report = load_json(manifest)
            _print(
                {
                    "output": str(path),
                    "manifest": str(manifest),
                    "selected_point_count": report["selected_point_count"],
                    "source_point_count": report["source_point_count"],
                }
            )
            code = 0
        elif args.command == "benchmark-record-gislab-no-detection":
            path = record_gislab_no_detection(
                args.run_manifest,
                args.reference_report,
                args.segment,
                args.output,
                source_coordinate_space=args.source_coordinate_space,
            )
            _print({"report": str(path), "rail_pair_count": 0, "status": "no_detection"})
            code = 0
        elif args.command == "benchmark-record-gislab-timeout":
            path = record_gislab_timeout(
                args.run_manifest,
                args.reference_report,
                args.segment,
                args.output,
                source_coordinate_space=args.source_coordinate_space,
            )
            _print({"report": str(path), "rail_pair_count": 0, "status": "timeout"})
            code = 0
        elif args.command == "benchmark-compare-rail-holdout":
            path = compare_rail_holdout_reports(
                args.baseline,
                args.method,
                args.output,
                baseline_name=args.baseline_name,
                method_name=args.method_name,
            )
            report = load_json(path)
            _print({"report": str(path), "win_counts": report["win_counts"]})
            code = 0
        elif args.command == "benchmark-bootstrap-rail-holdout":
            path = bootstrap_rail_holdout_comparison(
                args.baseline,
                args.method,
                args.output,
                iterations=args.iterations,
                seed=args.seed,
                minimum_confirmatory_blocks=args.minimum_confirmatory_blocks,
            )
            report = load_json(path)
            _print(
                {
                    "report": str(path),
                    "independent_block_count": report["independent_block_count"],
                    "overall_status": report["overall_status"],
                }
            )
            code = 0
        elif args.command == "benchmark-select-rail-vertical-fit":
            path = select_rail_vertical_fit(
                _vertical_fit_variants(args.variant),
                args.baseline,
                args.output,
            )
            report = load_json(path)
            _print(
                {
                    "report": str(path),
                    "selected_variant": report["selected_variant"],
                    "selected_settings": report["selected_settings"],
                    "selection_scope": report["selection_scope"],
                }
            )
            code = 0
        elif args.command == "benchmark-audit-rail-vertical-followup":
            path = audit_rail_vertical_followup(
                args.baseline,
                args.initial_selection,
                args.trigger_followup,
                args.nested_correction,
                args.final_followup,
                args.corrected_settings,
                args.output,
            )
            report = load_json(path)
            _print(
                {
                    "report": str(path),
                    "status": report["status"],
                    "evaluation_scope": report["evaluation_scope"],
                }
            )
            code = 0 if report["status"] == "passes_development_guardrails" else 5
        elif args.command == "rail-regression-audit":
            path = audit_rail_production_regression(
                _assignments(args.baseline_report, "--baseline-report"),
                _assignments(args.candidate_report, "--candidate-report"),
                args.output,
                support_clouds=(
                    _assignments(args.support_cloud, "--support-cloud")
                    if args.support_cloud
                    else None
                ),
                allow_additional_lines=args.allow_additional_lines,
                core_length_m=args.core_length_m,
                sample_step_m=args.sample_step_m,
                maximum_cross_change_m=args.maximum_cross_change_m,
                per_segment_support_p90_tolerance_m=(args.per_segment_support_p90_tolerance_m),
                maximum_additional_line_support_p90_m=(args.maximum_additional_line_support_p90_m),
                maximum_matched_line_support_p90_m=(args.maximum_matched_line_support_p90_m),
            )
            report = load_json(path)
            _print({"report": str(path), "status": report["status"], **report["aggregate"]})
            code = 0 if report["status"] == "pass" else 5
        elif args.command == "benchmark-render-rail-comparison":
            path = render_rail_method_comparison(
                args.evidence_manifest,
                _assignments(args.baseline_report, "--baseline-report"),
                _assignments(args.method_report, "--method-report"),
                args.output_dir,
            )
            manifest = load_json(path)
            _print(
                {
                    "manifest": str(path),
                    "contact_sheet": manifest["contact_sheet"],
                    "station_count": len(manifest["stations"]),
                    "visualization_role": manifest["visualization_role"],
                }
            )
            code = 0
        elif args.command == "benchmark-lock-experiment":
            path = lock_benchmark_experiment(
                args.root,
                args.experiment,
                _assignments(args.binding, "--binding"),
                output_path=args.output,
                freeze_manifest_path=args.freeze_manifest,
            )
            locked = load_json(path)
            _print(
                {
                    "locked_experiment": str(path),
                    "experiment_id": locked["experiment_id"],
                    "execution_mode": locked["execution_mode"],
                    "binding_count": len(locked["lock"]["bindings"]),
                    "locked_before_test": locked["locked_before_test"],
                }
            )
            code = 0
        elif args.command == "benchmark-make-vertical-tasks":
            root = create_vertical_annotation_package(
                args.root,
                _segment_sources(args.source),
                output_name=args.output_name,
                seed=args.seed,
            )
            manifest = load_json(root / "manifest.json")
            _print(
                {
                    "package": str(root),
                    "task_count": manifest["task_count"],
                    "reviewer_files": manifest["reviewer_files"],
                    "blind_to_existing_classification": manifest[
                        "blind_to_existing_classification"
                    ],
                }
            )
            code = 0
        elif args.command == "benchmark-check-vertical-tasks":
            report = validate_vertical_annotation_package(args.path)
            _print(report)
            code = 0 if report["passed"] else 2
        elif args.command == "benchmark-render-rail-truth-evidence":
            project = load_project(args.project)
            path = render_neutral_rail_evidence(
                project,
                args.root,
                args.scene_id,
                args.start_m,
                args.end_m,
                spacing_m=args.spacing_m,
                slice_half_width_m=args.slice_half_width_m,
                cross_min_m=args.cross_min_m,
                cross_max_m=args.cross_max_m,
                z_min_m=args.z_min_m,
                z_max_m=args.z_max_m,
                output_name=args.output_name,
            )
            manifest = load_json(path)
            _print(
                {
                    "neutral_evidence_manifest": str(path),
                    "station_count": len(manifest["stations"]),
                    "selected_point_count": manifest["selected_point_count"],
                    "model_overlay": manifest["model_overlay"],
                    "candidate_overlay": manifest["candidate_overlay"],
                }
            )
            code = 0
        elif args.command == "benchmark-make-rail-truth-tasks":
            root = create_rail_truth_annotation_package(
                args.root,
                args.evidence_manifest,
                output_name=args.output_name,
                seed=args.seed,
            )
            manifest = load_json(root / "manifest.json")
            _print(
                {
                    "package": str(root),
                    "task_count": manifest["task_count"],
                    "reviewer_files": manifest["reviewer_files"],
                    "blind_to_existing_model": manifest["blind_to_existing_model"],
                }
            )
            code = 0
        elif args.command == "benchmark-check-rail-truth-tasks":
            report = validate_rail_truth_annotation_package(args.path)
            _print(report)
            code = 0 if report["passed"] else 2
        elif args.command == "benchmark-lock-vertical-predictions":
            root = lock_vertical_predictions(
                args.root,
                args.annotation_package,
                _segment_sources(args.source),
                output_name=args.output_name,
            )
            manifest = load_json(root / "manifest.json")
            _print(
                {
                    "prediction_set": str(root),
                    "prediction_count": manifest["prediction_count"],
                    "class_distribution": manifest["class_distribution"],
                    "execution_mode": manifest["execution_mode"],
                    "eligible_as_fully_automatic_baseline": manifest[
                        "eligible_as_fully_automatic_baseline"
                    ],
                }
            )
            code = 0
        elif args.command == "benchmark-compare-vertical-reviews":
            root = compare_vertical_reviews(args.annotation_package, args.output_name)
            report = load_json(root / "agreement-report.json")
            _print(
                {
                    "comparison": str(root),
                    "task_count": report["task_count"],
                    "joint_decision_agreement": report["joint_decision_agreement"],
                    "consensus_count": report["consensus_count"],
                    "disagreement_count": report["disagreement_count"],
                    "ground_truth_status": report["ground_truth_status"],
                }
            )
            code = 0
        elif args.command == "audit-support-relationships":
            output = Path(args.output).resolve()
            if output.exists():
                raise FileExistsError(output)
            report = validate_support_relationship_patch(
                load_json(Path(args.registry)),
                load_json(Path(args.patch)),
                policy=SupportRelationshipPolicy(
                    maximum_absolute_contact_residual_m=args.maximum_contact_residual_m,
                    minimum_direct_evidence_kinds=args.minimum_direct_evidence_kinds,
                    require_modeled_owner=not args.allow_unmodeled_owner,
                ),
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            _print({"report": str(output), **report["summary"], "status": report["status"]})
            code = 0 if report["relationship_patch_allowed"] else 2
        elif args.command == "audit-rapid-candidate":
            report = write_rapid_candidate_audit(
                args.output,
                args.release_directory,
                args.manifest,
                args.delivery_gate,
                lineage_path=args.lineage,
                expected_release_id=args.expected_release_id,
                expected_parent_release_id=args.expected_parent_release_id,
            )
            _print(
                {
                    "report": str(Path(args.output).resolve()),
                    "release_id": report["release_id"],
                    "iteration_ready": report["iteration_ready"],
                    "hard_blocker_count": report["hard_blocker_count"],
                    "warning_count": report["warning_count"],
                }
            )
            code = 0 if report["passed"] else 2
        elif args.command == "auto-review-fixed-views":
            report = write_fixed_view_auto_review(
                args.output,
                args.manifest,
                maximum_spot_checks=args.maximum_spot_checks,
            )
            _print(
                {
                    "report": str(Path(args.output).resolve()),
                    "release_id": report["release_id"],
                    "status": report["status"],
                    **report["summary"],
                }
            )
            code = 0 if report["passed"] else 2
        elif args.command == "mesh-audit":
            path = Path(args.path)
            if path.suffix.lower() != ".obj":
                raise ValueError("The core mesh audit currently accepts OBJ files only")
            report = audit_obj(path, args.area_tolerance)
            _print(report)
            code = 0 if report["passed"] else 4
        elif args.command == "safety-check":
            report = safety_check(Path(args.root), int(args.max_mb * 1024 * 1024))
            _print(report)
            code = 0 if report["passed"] else 3
        else:
            raise RuntimeError(f"Unhandled command: {args.command}")
        if code:
            raise SystemExit(code)
    except (FileNotFoundError, FileExistsError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
