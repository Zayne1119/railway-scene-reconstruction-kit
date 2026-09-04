from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_recon.catenary_top_integration import integrate_catenary_top_equipment


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Integrate point-supported catenary top-equipment nodes."
    )
    parser.add_argument("--evidence-report", required=True, type=Path)
    parser.add_argument("--local-points", required=True, type=Path)
    parser.add_argument("--source-obj", required=True, type=Path)
    parser.add_argument("--source-origin", required=True, type=Path)
    parser.add_argument("--source-registry", required=True, type=Path)
    parser.add_argument("--frame-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    outputs = integrate_catenary_top_equipment(
        evidence_report_path=args.evidence_report,
        local_points_path=args.local_points,
        source_obj=args.source_obj,
        source_origin=args.source_origin,
        source_registry=args.source_registry,
        frame_report_path=args.frame_report,
        output_directory=args.output_dir,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
