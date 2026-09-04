from __future__ import annotations

from railway_recon.config import initialize_project, load_project
from railway_recon.io import load_json, write_json
from railway_recon.side_asset_pipeline import (
    run_side_asset_pipeline,
    validate_side_asset_pipeline_plan,
)


def _plan() -> dict:
    return {
        "schema_version": "railway.side-asset-pipeline-plan.v1",
        "opposite_platforms": [
            {
                "id": "platform-A",
                "segment_id": "A",
                "point_cloud": "input/A.las",
                "vertical_report": "reports/A.json",
                "output_dir": "derived/A-platform",
                "side": "left",
                "owned_interval_m": [0.0, 50.0],
            }
        ],
        "column_grids": [
            {
                "id": "grid-main",
                "ownership_plan": "reports/ownership.json",
                "vertical_reports": {"A": "reports/A.json"},
                "output": "reports/grid.json",
                "side": "left",
            }
        ],
        "seam_reconciliations": [
            {
                "id": "seams-main",
                "source_obj": "exports/source.obj",
                "source_origin": "exports/origin.json",
                "frame_report": "reports/A.json",
                "settings": "configs/seams.json",
                "output_dir": "exports/reconciled",
            }
        ],
    }


def test_plan_validation_rejects_duplicate_stage_ids_and_invalid_side() -> None:
    plan = _plan()
    plan["column_grids"][0]["id"] = "platform-A"
    plan["opposite_platforms"][0]["side"] = "centre"

    errors = validate_side_asset_pipeline_plan(plan)

    assert any("Stage ids must be unique" in item for item in errors)
    assert any("side must be" in item for item in errors)


def test_runner_executes_configured_stages_in_dependency_order(tmp_path, monkeypatch) -> None:
    project = load_project(initialize_project(tmp_path / "project", "sample", "Sample"))
    plan_path = project.root / "pipeline.json"
    write_json(plan_path, _plan())
    calls: list[tuple[str, object]] = []

    def platform(*args, **kwargs):
        calls.append(("platform", kwargs["owned_interval_m"]))
        return {"schema_version": "platform.result.v1"}

    def grid(*args, **kwargs):
        calls.append(("grid", kwargs["side"]))
        return {"schema_version": "grid.result.v1"}

    def seams(*args, **kwargs):
        calls.append(("seams", kwargs["registry_path"]))
        return {"schema_version": "seams.result.v1"}

    monkeypatch.setattr("railway_recon.side_asset_pipeline.reconstruct_opposite_platform", platform)
    monkeypatch.setattr("railway_recon.side_asset_pipeline.recover_corridor_column_grid", grid)
    monkeypatch.setattr("railway_recon.side_asset_pipeline.reconcile_mesh_seams", seams)

    report = run_side_asset_pipeline(
        project, plan_path, "workspace/reports/side_asset_pipeline_run.json"
    )

    assert calls == [
        ("platform", (0.0, 50.0)),
        ("grid", "left"),
        ("seams", None),
    ]
    assert report["passed"] is True
    assert [item["type"] for item in report["stages"]] == [
        "opposite_platform_reconstruction",
        "corridor_column_grid_recovery",
        "mesh_seam_reconciliation",
    ]
    stored = load_json(project.root / "workspace/reports/side_asset_pipeline_run.json")
    assert stored["stage_count"] == 3
