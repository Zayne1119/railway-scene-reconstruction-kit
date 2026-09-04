from __future__ import annotations

import argparse
from pathlib import Path

from railway_recon.final_scene_qa import audit_final_scene


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final evidence-aware scene QA")
    parser.add_argument("--obj", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--integration-report", required=True, type=Path)
    parser.add_argument("--small-asset-report", required=True, type=Path)
    parser.add_argument("--support-comparison", required=True, type=Path)
    parser.add_argument("--track-closure", required=True, type=Path)
    parser.add_argument("--corridor-view-manifest", required=True, type=Path)
    parser.add_argument("--catenary-view-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--maximum-component-gap-m", type=float, default=0.15)
    args = parser.parse_args()
    report = audit_final_scene(
        obj_path=args.obj,
        registry_path=args.registry,
        integration_report_path=args.integration_report,
        small_asset_report_path=args.small_asset_report,
        support_comparison_path=args.support_comparison,
        track_closure_path=args.track_closure,
        corridor_view_manifest_path=args.corridor_view_manifest,
        catenary_view_manifest_path=args.catenary_view_manifest,
        output_path=args.output,
        maximum_component_gap_m=args.maximum_component_gap_m,
    )
    print(f"passed={report['summary']['passed']}")
    print(f"checks={report['summary']['check_count']}")
    print(f"failed={report['summary']['failed_check_count']}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()

