from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_recon.adjacent_boundary_conductors import (
    build_adjacent_boundary_conductors,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build evidence-owned conductor continuations across a model boundary"
    )
    parser.add_argument("--cloud", required=True, type=Path)
    parser.add_argument("--gap-report", required=True, type=Path)
    parser.add_argument("--frame-report", required=True, type=Path)
    parser.add_argument("--current-boundary-report", required=True, type=Path)
    parser.add_argument("--source-obj", required=True, type=Path)
    parser.add_argument("--source-origin", required=True, type=Path)
    parser.add_argument("--source-registry", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--render-radius-m", type=float, default=0.018)
    args = parser.parse_args()
    outputs = build_adjacent_boundary_conductors(
        cloud_path=args.cloud,
        gap_report_path=args.gap_report,
        frame_report_path=args.frame_report,
        current_boundary_report_path=args.current_boundary_report,
        source_obj=args.source_obj,
        source_origin=args.source_origin,
        source_registry=args.source_registry,
        output_directory=args.output_dir,
        render_radius_m=args.render_radius_m,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
