from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_recon.boundary_surface_evidence import analyze_boundary_surface_evidence


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile platform and canopy point evidence near a model boundary"
    )
    parser.add_argument("--cloud", required=True, type=Path)
    parser.add_argument("--frame-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--station-minimum-m", required=True, type=float)
    parser.add_argument("--station-maximum-m", required=True, type=float)
    parser.add_argument("--station-bin-m", type=float, default=0.50)
    parser.add_argument("--minimum-points-per-bin", type=int, default=20)
    parser.add_argument(
        "--window",
        action="append",
        required=True,
        help="NAME:CROSS_MIN:CROSS_MAX:Z_MIN:Z_MAX",
    )
    args = parser.parse_args()
    windows = []
    for raw in args.window:
        name, cross_min, cross_max, z_min, z_max = raw.split(":")
        windows.append(
            {
                "name": name,
                "cross_range_m": [float(cross_min), float(cross_max)],
                "z_range_m": [float(z_min), float(z_max)],
            }
        )
    result = analyze_boundary_surface_evidence(
        cloud_path=args.cloud,
        frame_report_path=args.frame_report,
        output_path=args.output,
        station_minimum_m=args.station_minimum_m,
        station_maximum_m=args.station_maximum_m,
        windows=windows,
        station_bin_m=args.station_bin_m,
        minimum_points_per_bin=args.minimum_points_per_bin,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
