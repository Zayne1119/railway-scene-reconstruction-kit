from __future__ import annotations

import argparse
from pathlib import Path

from railway_recon.continuity_vertical_refinement import (
    build_continuity_vertical_refinement,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build point-continuity-confirmed vertical asset refinements."
    )
    parser.add_argument("--source-obj", required=True, type=Path)
    parser.add_argument("--source-mtl", required=True, type=Path)
    parser.add_argument("--source-registry", required=True, type=Path)
    parser.add_argument("--origin", required=True, type=Path)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--frame-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(
        build_continuity_vertical_refinement(
            args.source_obj,
            args.source_mtl,
            args.source_registry,
            args.origin,
            args.comparison,
            args.decisions,
            args.frame_report,
            args.output_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
