from __future__ import annotations

import argparse

from railway_recon.supplemental_station_end_wall import build_supplemental_station_end_wall


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the observed station end-wall candidate")
    parser.add_argument("--cloud", required=True)
    parser.add_argument("--frame-report", required=True)
    parser.add_argument("--source-obj", required=True)
    parser.add_argument("--source-origin", required=True)
    parser.add_argument("--source-registry", required=True)
    parser.add_argument("--group-evidence", required=True)
    parser.add_argument("--photo-evidence", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    outputs = build_supplemental_station_end_wall(
        cloud_path=args.cloud,
        frame_report_path=args.frame_report,
        source_obj=args.source_obj,
        source_origin=args.source_origin,
        source_registry=args.source_registry,
        group_evidence_path=args.group_evidence,
        photo_evidence_path=args.photo_evidence,
        output_directory=args.output_directory,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
