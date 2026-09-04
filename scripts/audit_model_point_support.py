from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_recon.model_point_support import audit_model_point_support


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit per-object mesh support against a supplemental point-cloud clip."
    )
    parser.add_argument("--obj", required=True, type=Path)
    parser.add_argument("--origin", required=True, type=Path)
    parser.add_argument("--cloud", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--target-cloud-points", type=int, default=3_000_000)
    parser.add_argument("--max-samples-per-object", type=int, default=5_000)
    args = parser.parse_args()
    outputs = audit_model_point_support(
        args.obj,
        args.origin,
        args.cloud,
        args.output,
        asset_registry_path=args.registry,
        target_cloud_points=args.target_cloud_points,
        maximum_samples_per_object=args.max_samples_per_object,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
