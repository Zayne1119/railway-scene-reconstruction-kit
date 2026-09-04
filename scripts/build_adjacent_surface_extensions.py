from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_recon.adjacent_surface_extensions import build_adjacent_surface_extensions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build point-supported platform and canopy extensions at a scene boundary"
    )
    parser.add_argument("--source-obj", required=True, type=Path)
    parser.add_argument("--source-origin", required=True, type=Path)
    parser.add_argument("--cloud", required=True, type=Path)
    parser.add_argument("--frame-report", required=True, type=Path)
    parser.add_argument("--evidence-report", required=True, type=Path)
    parser.add_argument("--target-settings", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = build_adjacent_surface_extensions(
        source_obj=args.source_obj,
        source_origin=args.source_origin,
        cloud_path=args.cloud,
        frame_report=args.frame_report,
        evidence_report=args.evidence_report,
        target_settings=args.target_settings,
        output_directory=args.output_dir,
    )
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
