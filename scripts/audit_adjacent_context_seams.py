from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_recon.adjacent_context_seam_audit import audit_adjacent_context_seams


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit track, platform and canopy cross-sections at an adjacent-scene seam"
    )
    parser.add_argument("--current-obj", required=True, type=Path)
    parser.add_argument("--current-origin", required=True, type=Path)
    parser.add_argument("--adjacent-obj", required=True, type=Path)
    parser.add_argument("--adjacent-origin", required=True, type=Path)
    parser.add_argument("--frame-report", required=True, type=Path)
    parser.add_argument("--seam-station-m", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit_adjacent_context_seams(
        current_obj=args.current_obj,
        current_origin=args.current_origin,
        adjacent_obj=args.adjacent_obj,
        adjacent_origin=args.adjacent_origin,
        frame_report=args.frame_report,
        seam_station_m=args.seam_station_m,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
