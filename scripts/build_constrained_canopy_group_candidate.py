from __future__ import annotations

import argparse
from pathlib import Path

from railway_recon.canopy_group_refinement import build_constrained_canopy_group_candidate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a seam-preserving multi-segment canopy roof-group candidate."
    )
    parser.add_argument("--source-obj", required=True, type=Path)
    parser.add_argument("--source-mtl", required=True, type=Path)
    parser.add_argument("--source-registry", required=True, type=Path)
    parser.add_argument("--origin", required=True, type=Path)
    parser.add_argument("--plane-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(
        build_constrained_canopy_group_candidate(
            source_obj=args.source_obj,
            source_mtl=args.source_mtl,
            source_registry=args.source_registry,
            model_origin=args.origin,
            plane_report_path=args.plane_report,
            output_directory=args.output_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
