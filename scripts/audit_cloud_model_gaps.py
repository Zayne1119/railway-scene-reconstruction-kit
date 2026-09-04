from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_recon.cloud_model_gap_audit import audit_cloud_model_gaps


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit new point-cloud evidence that is not explained by the current model."
    )
    parser.add_argument("--cloud", required=True, type=Path)
    parser.add_argument("--obj", required=True, type=Path)
    parser.add_argument("--origin", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--frame-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-cloud-points", type=int, default=4_000_000)
    parser.add_argument("--model-spacing-m", type=float, default=0.15)
    parser.add_argument("--unexplained-distance-m", type=float, default=0.25)
    args = parser.parse_args()
    outputs = audit_cloud_model_gaps(
        cloud_path=args.cloud,
        obj_path=args.obj,
        model_origin_path=args.origin,
        frame_report_path=args.frame_report,
        asset_registry_path=args.registry,
        output_directory=args.output_dir,
        target_cloud_points=args.target_cloud_points,
        model_surface_spacing_m=args.model_spacing_m,
        unexplained_distance_m=args.unexplained_distance_m,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
