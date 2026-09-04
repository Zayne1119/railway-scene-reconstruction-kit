from __future__ import annotations

import argparse
from pathlib import Path

from railway_recon.canopy_group_refinement import audit_right_canopy_physical_seams


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit right-canopy physical seam cross-sections."
    )
    parser.add_argument("--obj", required=True, type=Path)
    parser.add_argument("--origin", required=True, type=Path)
    parser.add_argument("--frame-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        audit_right_canopy_physical_seams(
            obj_path=args.obj,
            model_origin_path=args.origin,
            frame_report_path=args.frame_report,
            output_path=args.output,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
