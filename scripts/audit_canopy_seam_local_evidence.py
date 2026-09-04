from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_recon.canopy_seam_local_evidence import audit_canopy_seam_local_evidence


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit dense point-cloud evidence around canopy segment seams."
    )
    parser.add_argument("--cloud", required=True, type=Path)
    parser.add_argument("--frame-report", required=True, type=Path)
    parser.add_argument("--union-seam", required=True, type=Path)
    parser.add_argument("--physical-seam", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    outputs = audit_canopy_seam_local_evidence(
        cloud_path=args.cloud,
        frame_report_path=args.frame_report,
        union_seam_path=args.union_seam,
        physical_seam_path=args.physical_seam,
        output_directory=args.output_dir,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
