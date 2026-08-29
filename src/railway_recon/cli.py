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
from .config import initialize_project, load_project, validate_project_value
from .defect_injection import evaluate_defect_injection
from .experiment_lock import lock_benchmark_experiment
from .gislab_baseline import (
    normalize_gislab_railtrack_points,
    prepare_gislab_railtrack_input,
    record_gislab_no_detection,
    record_gislab_timeout,
)
from .holdout_bootstrap import bootstrap_rail_holdout_comparison
from .holdout_comparison import compare_rail_holdout_reports
from .io import load_json
from .manifest import write_run_manifest
from .mesh_audit import audit_obj
from .open3d_baseline import detect_open3d_rail_baseline
from .point_holdout import (
    split_existing_point_cloud_holdout,
    split_point_cloud_holdout,
    validate_point_cloud_holdout,
)
from .prediction_lock import lock_vertical_predictions
from .projection import calibrate_projection
from .qa import quality_report
from .quality_gate import evaluate_quality_gate, validate_gate_result
from .rail_comparison_figure import render_rail_method_comparison
from .rail_holdout_metrics import evaluate_rail_holdout
from .rail_pair_curation import curate_rail_pairs
from .rail_production_regression import audit_rail_production_regression
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
from .segments import crop_segments, plan_segments
from .topology_ablation import evaluate_topology_ablation
from .track_graph import build_track_graph, validate_track_graph
from .vertical_fit_selection import select_rail_vertical_fit
from .vertical_followup import audit_rail_vertical_followup


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


def _vertical_fit_variants(values: list[str]) -> list[tuple[str, str, str]]:
    result: list[tuple[str, str, str]] = []
    for value in values:
        parts = value.split("=", 2)
        if len(parts) != 3 or not all(parts):
            raise ValueError(
                f"Invalid --variant {value!r}; expected NAME=EVALUATION=SETTINGS"
            )
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

    plan = commands.add_parser("plan-segments", help="Plan corridor segments from camera poses")
    plan.add_argument("--project", required=True)

    segment = commands.add_parser("segment", help="Crop planned LAS/LAZ corridor segments")
    segment.add_argument("--project", required=True)
    segment.add_argument("--ids", nargs="*")
    segment.add_argument("--overwrite", action="store_true")

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
    targeted_recovery.add_argument(
        "--minimum-joint-support-ratio", type=float, default=0.60
    )
    targeted_recovery.add_argument(
        "--maximum-asymmetric-support-ratio", type=float, default=0.25
    )
    targeted_recovery.add_argument(
        "--maximum-internal-joint-gap-m", type=float, default=5.0
    )

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
    build_graph_mesh.add_argument("--overwrite", action="store_true")

    create_track_review = commands.add_parser(
        "track-review-create",
        help="Create a hash-bound independent review queue for rail candidates",
    )
    create_track_review.add_argument("--project", required=True)
    create_track_review.add_argument(
        "--source", action="append", required=True, help="Repeat SEGMENT=RAIL_REPORT"
    )
    create_track_review.add_argument(
        "--output-name", default="track_graph_rail_review_v1"
    )

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
    benchmark_holdout_bootstrap.add_argument(
        "--minimum-confirmatory-blocks", type=int, default=10
    )

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
    rail_regression.add_argument(
        "--per-segment-support-p90-tolerance-m", type=float, default=0.005
    )
    rail_regression.add_argument(
        "--maximum-additional-line-support-p90-m", type=float, default=0.075
    )
    rail_regression.add_argument(
        "--maximum-matched-line-support-p90-m", type=float, default=0.060
    )

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
    benchmark_annotation.add_argument(
        "--output-name", default="vertical_candidates_blind_v1"
    )
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
    benchmark_rail_evidence.add_argument(
        "--output-name", default="rail_neutral_evidence_v1"
    )

    benchmark_rail_annotation = commands.add_parser(
        "benchmark-make-rail-truth-tasks",
        help="Build double-blind rail geometry tasks from neutral raw-point evidence",
    )
    benchmark_rail_annotation.add_argument("--root", required=True)
    benchmark_rail_annotation.add_argument(
        "--evidence-manifest", action="append", required=True
    )
    benchmark_rail_annotation.add_argument(
        "--output-name", default="rail_geometry_blind_v1"
    )
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
    benchmark_review_compare.add_argument(
        "--output-name", default="vertical_review_comparison_v1"
    )

    mesh_audit = commands.add_parser("mesh-audit", help="Audit an OBJ before Blender/UE import")
    mesh_audit.add_argument("--path", required=True)
    mesh_audit.add_argument("--area-tolerance", type=float, default=1e-12)

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
            _print({"project": str(Path(args.project).resolve()), "valid": not errors, "errors": errors})
            if errors:
                raise SystemExit(2)
            return
        if args.command == "audit":
            code = _project_command(
                args, "audit", lambda project: audit_project(project, full_hash=args.full_hash)
            )
        elif args.command == "plan-segments":
            code = _project_command(args, "plan_segments", plan_segments)
        elif args.command == "segment":
            code = _project_command(
                args,
                "segment",
                lambda project: crop_segments(project, set(args.ids or []), args.overwrite),
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
                maximum_asymmetric_support_ratio=(
                    args.maximum_asymmetric_support_ratio
                ),
                maximum_internal_joint_gap_m=args.maximum_internal_joint_gap_m,
            )
            report = load_json(path)
            updated_graph = (
                None
                if args.updated_graph is None
                else str(
                    apply_targeted_rail_recovery(
                        args.graph, path, args.updated_graph
                    )
                )
            )
            _print(
                {
                    "report": str(path),
                    "updated_graph": updated_graph,
                    **report["summary"],
                }
            )
            code = 0
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
                    not args.skip_registry,
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
                lambda project: freeze_release_registry(
                    project, args.release_id, args.output
                ),
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
                    "requested_holdout_fraction": manifest[
                        "requested_holdout_fraction"
                    ],
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
                per_segment_support_p90_tolerance_m=(
                    args.per_segment_support_p90_tolerance_m
                ),
                maximum_additional_line_support_p90_m=(
                    args.maximum_additional_line_support_p90_m
                ),
                maximum_matched_line_support_p90_m=(
                    args.maximum_matched_line_support_p90_m
                ),
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
