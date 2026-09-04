from __future__ import annotations

import argparse

from railway_recon.supplemental_station_entry import build_supplemental_station_entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an evidence-aware station-entry candidate")
    parser.add_argument("--cloud", required=True)
    parser.add_argument("--frame-report", required=True)
    parser.add_argument("--surface-analysis", required=True)
    parser.add_argument("--source-obj", required=True)
    parser.add_argument("--source-origin", required=True)
    parser.add_argument("--source-registry", required=True)
    parser.add_argument("--photo-evidence-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    outputs = build_supplemental_station_entry(
        cloud_path=args.cloud,
        frame_report_path=args.frame_report,
        surface_analysis_path=args.surface_analysis,
        source_obj=args.source_obj,
        source_origin=args.source_origin,
        source_registry=args.source_registry,
        photo_evidence_directory=args.photo_evidence_directory,
        output_directory=args.output_directory,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
