from __future__ import annotations

from railway_recon.cli import build_parser


def test_opposite_platform_cli_accepts_shared_ownership_plan() -> None:
    args = build_parser().parse_args(
        [
            "reconstruct-opposite-platform",
            "--project",
            "project.json",
            "--segment",
            "A",
            "--point-cloud",
            "A.laz",
            "--vertical-report",
            "A.json",
            "--output-dir",
            "derived/A",
            "--side",
            "left",
            "--ownership-plan",
            "ownership.json",
        ]
    )

    assert args.ownership_plan == "ownership.json"
    assert args.minimum_longitudinal_m is None


def test_side_asset_pipeline_cli_contract() -> None:
    args = build_parser().parse_args(
        [
            "run-side-asset-pipeline",
            "--project",
            "project.json",
            "--plan",
            "pipeline.json",
            "--output",
            "run.json",
        ]
    )

    assert args.command == "run-side-asset-pipeline"
    assert args.plan == "pipeline.json"
