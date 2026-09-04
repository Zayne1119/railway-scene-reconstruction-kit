from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_recon.adjacent_platform_transition import build_adjacent_platform_transition


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a point-supported adjacent platform transition")
    parser.add_argument("--current-obj", required=True, type=Path)
    parser.add_argument("--current-origin", required=True, type=Path)
    parser.add_argument("--adjacent-obj", required=True, type=Path)
    parser.add_argument("--adjacent-origin", required=True, type=Path)
    parser.add_argument("--cloud", required=True, type=Path)
    parser.add_argument("--frame-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = build_adjacent_platform_transition(
        current_obj=args.current_obj,
        current_origin=args.current_origin,
        adjacent_obj=args.adjacent_obj,
        adjacent_origin=args.adjacent_origin,
        cloud_path=args.cloud,
        frame_report=args.frame_report,
        output_directory=args.output_dir,
        current_prefixes=("SEG2150-PLATFORM--",),
        adjacent_prefixes=("S2200_2250M-PLATFORM-",),
    )
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
