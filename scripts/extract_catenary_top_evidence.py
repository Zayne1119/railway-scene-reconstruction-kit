from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_recon.catenary_top_evidence import extract_catenary_top_evidence


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract per-mast cantilever, insulator and positioner point evidence."
    )
    parser.add_argument("--cloud", required=True, type=Path)
    parser.add_argument("--obj", required=True, type=Path)
    parser.add_argument("--origin", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--frame-report", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    outputs = extract_catenary_top_evidence(
        cloud_path=args.cloud,
        obj_path=args.obj,
        origin_path=args.origin,
        registry_path=args.registry,
        frame_report_path=args.frame_report,
        inventory_path=args.inventory,
        output_directory=args.output_dir,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
