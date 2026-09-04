from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_recon.canopy_column_lateral_refit import (
    build_canopy_column_lateral_refit,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refit canopy-column assemblies from two observed shaft faces."
    )
    parser.add_argument("--source-obj", required=True, type=Path)
    parser.add_argument("--source-mtl", required=True, type=Path)
    parser.add_argument("--source-origin", required=True, type=Path)
    parser.add_argument("--source-registry", required=True, type=Path)
    parser.add_argument("--gap-report", required=True, type=Path)
    parser.add_argument("--frame-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    outputs = build_canopy_column_lateral_refit(
        source_obj=args.source_obj,
        source_mtl=args.source_mtl,
        source_origin=args.source_origin,
        source_registry=args.source_registry,
        gap_report_path=args.gap_report,
        frame_report_path=args.frame_report,
        output_directory=args.output_dir,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
