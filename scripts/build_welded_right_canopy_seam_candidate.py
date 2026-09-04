from __future__ import annotations

import argparse
from pathlib import Path

from railway_recon.canopy_group_refinement import (
    build_welded_right_canopy_seam_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a locally welded right-canopy seam candidate."
    )
    parser.add_argument("--source-obj", required=True, type=Path)
    parser.add_argument("--source-mtl", required=True, type=Path)
    parser.add_argument("--source-registry", required=True, type=Path)
    parser.add_argument("--origin", required=True, type=Path)
    parser.add_argument("--frame-report", required=True, type=Path)
    parser.add_argument("--seam-evidence", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    outputs = build_welded_right_canopy_seam_candidate(
        source_obj=args.source_obj,
        source_mtl=args.source_mtl,
        source_registry=args.source_registry,
        model_origin=args.origin,
        frame_report_path=args.frame_report,
        seam_evidence_path=args.seam_evidence,
        output_directory=args.output_dir,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
