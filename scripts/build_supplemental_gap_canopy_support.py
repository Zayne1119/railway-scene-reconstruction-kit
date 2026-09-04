from __future__ import annotations

import argparse
import json

from railway_recon.supplemental_gap_canopy_support import (
    build_supplemental_gap_canopy_support,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a photo-confirmed same-station outer canopy support candidate."
    )
    parser.add_argument("--cloud", required=True)
    parser.add_argument("--gap-report", required=True)
    parser.add_argument("--dispositions", required=True)
    parser.add_argument("--photo-evidence", required=True)
    parser.add_argument("--frame-report", required=True)
    parser.add_argument("--source-obj", required=True)
    parser.add_argument("--source-mtl", required=True)
    parser.add_argument("--source-origin", required=True)
    parser.add_argument("--source-registry", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    outputs = build_supplemental_gap_canopy_support(
        cloud_path=args.cloud,
        gap_report_path=args.gap_report,
        disposition_path=args.dispositions,
        photo_evidence_path=args.photo_evidence,
        frame_report_path=args.frame_report,
        source_obj=args.source_obj,
        source_mtl=args.source_mtl,
        source_origin=args.source_origin,
        source_registry=args.source_registry,
        output_directory=args.output,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
