from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_recon.canopy_column_alignment_refinement import (
    build_canopy_column_alignment_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Align canopy column rows from supplemental point and photo evidence."
    )
    parser.add_argument("--obj", required=True, type=Path)
    parser.add_argument("--mtl", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--origin", required=True, type=Path)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--frame-report", required=True, type=Path)
    parser.add_argument("--photo-review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build_canopy_column_alignment_candidate(
        args.obj,
        args.mtl,
        args.registry,
        args.origin,
        args.comparison,
        args.frame_report,
        args.photo_review,
        args.output,
    )
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
