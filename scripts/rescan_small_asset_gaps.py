from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_recon.small_asset_rescan import rescan_small_asset_gaps


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rescan supplemental point-cloud platform gaps for small assets."
    )
    parser.add_argument("--cloud", required=True, type=Path)
    parser.add_argument("--obj", required=True, type=Path)
    parser.add_argument("--origin", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--frame-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-cloud-points", type=int, default=6_000_000)
    args = parser.parse_args()
    outputs = rescan_small_asset_gaps(
        cloud_path=args.cloud,
        obj_path=args.obj,
        origin_path=args.origin,
        registry_path=args.registry,
        frame_report_path=args.frame_report,
        output_directory=args.output_dir,
        target_cloud_points=args.target_cloud_points,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
