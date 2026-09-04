from __future__ import annotations

import argparse

from railway_recon.group_point_evidence import render_group_point_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Render grouped point-cloud evidence for candidates")
    parser.add_argument("--cloud", required=True)
    parser.add_argument("--gap-report", required=True)
    parser.add_argument("--frame-report", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--candidate-id", action="append", required=True)
    parser.add_argument("--station-margin-m", type=float, default=4.0)
    parser.add_argument("--cross-margin-m", type=float, default=4.0)
    parser.add_argument("--z-margin-m", type=float, default=1.0)
    parser.add_argument("--maximum-points", type=int, default=500_000)
    args = parser.parse_args()
    outputs = render_group_point_evidence(
        cloud_path=args.cloud,
        gap_report_path=args.gap_report,
        frame_report_path=args.frame_report,
        output_directory=args.output_directory,
        candidate_ids=tuple(args.candidate_id),
        station_margin_m=args.station_margin_m,
        cross_margin_m=args.cross_margin_m,
        z_margin_m=args.z_margin_m,
        maximum_points=args.maximum_points,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
