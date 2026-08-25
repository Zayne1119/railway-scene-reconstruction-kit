from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .audit import audit_project
from .algorithms.linear_candidates import detect_linear_candidates
from .algorithms.rail_candidates import detect_rail_candidates
from .algorithms.reviewed_scene import build_reviewed_scene
from .algorithms.track import build_parametric_track
from .config import initialize_project, load_project, validate_project_value
from .io import load_json
from .manifest import write_run_manifest
from .mesh_audit import audit_obj
from .projection import calibrate_projection
from .qa import quality_report
from .registry import (
    import_registry_records,
    initialize_registry,
    summarize_registry,
    validate_registry_file,
)
from .safety import safety_check
from .segments import crop_segments, plan_segments


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _project_command(args: argparse.Namespace, name: str, action: Callable[..., Any]) -> int:
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

    qa = commands.add_parser("qa", help="Build the current project quality report")
    qa.add_argument("--project", required=True)

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
                lambda project: detect_rail_candidates(project, args.segment, args.overwrite),
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
        elif args.command == "qa":
            code = _project_command(args, "qa", quality_report)
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
