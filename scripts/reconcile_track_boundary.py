from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_recon.track_boundary_reconciliation import (
    build_track_boundary_reconciliation_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoothly reconcile reviewed track geometry to the adjacent track boundary"
    )
    parser.add_argument("--source-obj", required=True, type=Path)
    parser.add_argument("--source-origin", required=True, type=Path)
    parser.add_argument("--source-registry", required=True, type=Path)
    parser.add_argument("--frame-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seam-station-m", required=True, type=float)
    parser.add_argument("--transition-length-m", type=float, default=12.0)
    args = parser.parse_args()
    result = build_track_boundary_reconciliation_candidate(
        source_obj=args.source_obj,
        source_origin=args.source_origin,
        source_registry=args.source_registry,
        frame_report=args.frame_report,
        output_directory=args.output_dir,
        seam_station_m=args.seam_station_m,
        transition_length_m=args.transition_length_m,
    )
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
