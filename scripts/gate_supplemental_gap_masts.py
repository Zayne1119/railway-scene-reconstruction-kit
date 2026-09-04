from __future__ import annotations

import argparse
import json

from railway_recon.supplemental_gap_mast_gate import gate_supplemental_gap_masts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate photo-interpreted supplemental catenary mast candidates."
    )
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--obj", required=True)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--frame-report", required=True)
    parser.add_argument("--photo-evidence", required=True)
    parser.add_argument("--fixed-views", required=True)
    parser.add_argument("--mesh-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = gate_supplemental_gap_masts(
        candidate_report_path=args.candidate_report,
        candidate_obj_path=args.obj,
        candidate_origin_path=args.origin,
        frame_report_path=args.frame_report,
        photo_evidence_path=args.photo_evidence,
        fixed_view_manifest_path=args.fixed_views,
        mesh_audit_path=args.mesh_audit,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
