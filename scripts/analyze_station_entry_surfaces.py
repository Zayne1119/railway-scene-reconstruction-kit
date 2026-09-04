from __future__ import annotations

import argparse

from railway_recon.station_entry_surface_analysis import analyse_station_entry_surfaces


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse dense station-entry surface bands")
    parser.add_argument("--cloud", required=True)
    parser.add_argument("--frame-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = analyse_station_entry_surfaces(
        cloud_path=args.cloud,
        frame_report_path=args.frame_report,
        output_path=args.output,
    )
    print(output)


if __name__ == "__main__":
    main()
