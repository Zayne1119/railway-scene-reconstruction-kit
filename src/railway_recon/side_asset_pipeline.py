from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .corridor_column_grid import recover_corridor_column_grid
from .io import load_json, sha256_file, write_json
from .mesh_seam_reconciliation import reconcile_mesh_seams
from .opposite_platform_reconstruction import reconstruct_opposite_platform

SCHEMA_VERSION = "railway.side-asset-pipeline-plan.v1"


def validate_side_asset_pipeline_plan(plan: dict[str, Any]) -> list[str]:
    """Validate the small orchestration layer without hiding stage contracts."""
    errors: list[str] = []
    if plan.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")

    opposite = plan.get("opposite_platforms", [])
    grids = plan.get("column_grids", [])
    seams = plan.get("seam_reconciliations", [])
    for name, values in (
        ("opposite_platforms", opposite),
        ("column_grids", grids),
        ("seam_reconciliations", seams),
    ):
        if not isinstance(values, list):
            errors.append(f"{name} must be an array")
    if errors:
        return errors
    if not (opposite or grids or seams):
        errors.append("At least one pipeline stage must be configured")

    for index, item in enumerate(opposite):
        prefix = f"opposite_platforms[{index}]"
        required = (
            "id",
            "segment_id",
            "point_cloud",
            "vertical_report",
            "output_dir",
            "side",
        )
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be an object")
            continue
        errors.extend(f"{prefix}.{key} is required" for key in required if key not in item)
        interval = item.get("owned_interval_m")
        ownership_plan = item.get("ownership_plan")
        if (interval is None) == (ownership_plan is None):
            errors.append(
                f"{prefix} must configure exactly one of owned_interval_m or ownership_plan"
            )
        elif interval is not None and (not isinstance(interval, list) or len(interval) != 2):
            errors.append(f"{prefix}.owned_interval_m must contain [minimum, maximum]")
        if item.get("side") not in {"left", "right"}:
            errors.append(f"{prefix}.side must be 'left' or 'right'")

    for index, item in enumerate(grids):
        prefix = f"column_grids[{index}]"
        required = ("id", "ownership_plan", "vertical_reports", "output", "side")
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be an object")
            continue
        errors.extend(f"{prefix}.{key} is required" for key in required if key not in item)
        if not isinstance(item.get("vertical_reports"), dict) or not item.get("vertical_reports"):
            errors.append(f"{prefix}.vertical_reports must be a non-empty object")
        if item.get("side") not in {"left", "right"}:
            errors.append(f"{prefix}.side must be 'left' or 'right'")

    for index, item in enumerate(seams):
        prefix = f"seam_reconciliations[{index}]"
        required = ("id", "source_obj", "source_origin", "frame_report", "settings", "output_dir")
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be an object")
            continue
        errors.extend(f"{prefix}.{key} is required" for key in required if key not in item)

    identifiers = [
        str(item.get("id"))
        for values in (opposite, grids, seams)
        for item in values
        if isinstance(item, dict) and item.get("id") is not None
    ]
    if len(identifiers) != len(set(identifiers)):
        errors.append("Stage ids must be unique across the pipeline plan")
    return errors


def _stage_record(
    stage_id: str,
    stage_type: str,
    result: dict[str, Any],
    output_reference: str,
) -> dict[str, Any]:
    return {
        "id": stage_id,
        "type": stage_type,
        "status": "pass",
        "passed": True,
        "output_reference": output_reference,
        "result_schema_version": result.get("schema_version"),
    }


def run_side_asset_pipeline(
    project: ProjectConfig,
    plan_path: str | Path,
    output_report_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run reusable Site-B-derived stages from an explicit, auditable plan.

    The plan intentionally references concrete upstream artifacts.  It does not
    guess asset sides, seam pairs, or ownership boundaries; those decisions stay
    visible and reviewable in project configuration.
    """
    source = project.resolve(plan_path)
    output = project.resolve(output_report_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    plan = load_json(source)
    errors = validate_side_asset_pipeline_plan(plan)
    if errors:
        raise ValueError("Invalid side-asset pipeline plan:\n- " + "\n- ".join(errors))

    stages: list[dict[str, Any]] = []
    for item in plan.get("opposite_platforms", []):
        interval_value = item.get("owned_interval_m")
        interval = (
            tuple(float(value) for value in interval_value) if interval_value is not None else None
        )
        result = reconstruct_opposite_platform(
            project,
            str(item["segment_id"]),
            item["point_cloud"],
            item["vertical_report"],
            item["output_dir"],
            side=str(item["side"]),
            owned_interval_m=(interval[0], interval[1]) if interval else None,
            ownership_plan_path=item.get("ownership_plan"),
            settings_path=item.get("settings"),
            overwrite=overwrite,
        )
        stages.append(
            _stage_record(
                str(item["id"]),
                "opposite_platform_reconstruction",
                result,
                str(project.resolve(item["output_dir"]) / "opposite_platform_build.json"),
            )
        )

    for item in plan.get("column_grids", []):
        result = recover_corridor_column_grid(
            project,
            item["ownership_plan"],
            {str(key): value for key, value in item["vertical_reports"].items()},
            item["output"],
            side=str(item["side"]),
            settings_path=item.get("settings"),
            overwrite=overwrite,
        )
        stages.append(
            _stage_record(
                str(item["id"]),
                "corridor_column_grid_recovery",
                result,
                str(project.resolve(item["output"])),
            )
        )

    for item in plan.get("seam_reconciliations", []):
        result = reconcile_mesh_seams(
            project,
            item["source_obj"],
            item["source_origin"],
            item["frame_report"],
            item["settings"],
            item["output_dir"],
            registry_path=item.get("registry"),
            overwrite=overwrite,
        )
        stages.append(
            _stage_record(
                str(item["id"]),
                "mesh_seam_reconciliation",
                result,
                str(project.resolve(item["output_dir"]) / "seam_reconciliation.json"),
            )
        )

    report = {
        "schema_version": "railway.side-asset-pipeline-run.v1",
        "project_id": project.project_id,
        "plan": str(source),
        "plan_sha256": sha256_file(source),
        "stage_count": len(stages),
        "stages": stages,
        "passed": all(bool(item["passed"]) for item in stages),
        "status": "pass_all_configured_stages",
        "limitations": [
            "The pipeline fails closed when evidence gates, ownership, or seam tolerances fail.",
            "Inferred opposite-platform volume and missing column locations remain confidence-labelled.",
            "Mesh seam reconciliation must run before final normal/tangent generation for Blender or UE.",
        ],
    }
    write_json(output, report)
    return report
