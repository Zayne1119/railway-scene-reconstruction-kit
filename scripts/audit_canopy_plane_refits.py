from __future__ import annotations

import argparse
from pathlib import Path

from railway_recon.canopy_plane_refit import audit_canopy_plane_refits


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit segmented canopy roof planes to new evidence.")
    parser.add_argument("--obj", required=True, type=Path)
    parser.add_argument("--origin", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--cloud", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-cloud-points", type=int, default=4_000_000)
    args = parser.parse_args()
    print(
        audit_canopy_plane_refits(
            args.obj,
            args.origin,
            args.registry,
            args.cloud,
            args.output,
            target_cloud_points=args.target_cloud_points,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
