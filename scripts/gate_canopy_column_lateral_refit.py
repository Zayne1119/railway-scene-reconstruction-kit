from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_recon.canopy_column_lateral_refit_gate import (
    gate_canopy_column_lateral_refit,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate lateral canopy-column refits using placement and cloud evidence."
    )
    parser.add_argument("--refit-report", required=True, type=Path)
    parser.add_argument("--candidate-obj", required=True, type=Path)
    parser.add_argument("--origin", required=True, type=Path)
    parser.add_argument("--frame-report", required=True, type=Path)
    parser.add_argument("--baseline-gap-report", required=True, type=Path)
    parser.add_argument("--candidate-gap-report", required=True, type=Path)
    parser.add_argument("--mesh-audit", required=True, type=Path)
    parser.add_argument("--fixed-views", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = gate_canopy_column_lateral_refit(
        refit_report_path=args.refit_report,
        candidate_obj_path=args.candidate_obj,
        model_origin_path=args.origin,
        frame_report_path=args.frame_report,
        baseline_gap_report_path=args.baseline_gap_report,
        candidate_gap_report_path=args.candidate_gap_report,
        mesh_audit_path=args.mesh_audit,
        fixed_view_manifest_path=args.fixed_views,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
