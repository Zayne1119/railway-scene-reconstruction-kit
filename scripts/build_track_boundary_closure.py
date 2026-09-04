from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_recon.track_boundary_closure import build_track_boundary_closure


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Close mesh, seam, point-support and fixed-view gates for a track handoff"
    )
    parser.add_argument("--candidate-obj", required=True, type=Path)
    parser.add_argument("--candidate-build-report", required=True, type=Path)
    parser.add_argument("--mesh-audit", required=True, type=Path)
    parser.add_argument("--baseline-seam-audit", required=True, type=Path)
    parser.add_argument("--candidate-seam-audit", required=True, type=Path)
    parser.add_argument("--baseline-support-report", required=True, type=Path)
    parser.add_argument("--candidate-support-report", required=True, type=Path)
    parser.add_argument("--fixed-view-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build_track_boundary_closure(
        candidate_obj=args.candidate_obj,
        candidate_build_report=args.candidate_build_report,
        mesh_audit=args.mesh_audit,
        baseline_seam_audit=args.baseline_seam_audit,
        candidate_seam_audit=args.candidate_seam_audit,
        baseline_support_report=args.baseline_support_report,
        candidate_support_report=args.candidate_support_report,
        fixed_view_manifest=args.fixed_view_manifest,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
