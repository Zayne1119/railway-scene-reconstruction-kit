from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_recon.supplemental_conductor_refinement import (
    build_supplemental_conductor_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build point-supported sagged conductor additions from reverse gap evidence."
    )
    parser.add_argument("--cloud", required=True, type=Path)
    parser.add_argument("--gap-report", required=True, type=Path)
    parser.add_argument("--frame-report", required=True, type=Path)
    parser.add_argument("--source-obj", required=True, type=Path)
    parser.add_argument("--source-origin", required=True, type=Path)
    parser.add_argument("--source-registry", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    outputs = build_supplemental_conductor_candidate(
        cloud_path=args.cloud,
        gap_report_path=args.gap_report,
        frame_report_path=args.frame_report,
        source_obj=args.source_obj,
        source_origin=args.source_origin,
        source_registry=args.source_registry,
        output_directory=args.output_dir,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
