from __future__ import annotations

import argparse
import json

from railway_recon.canopy_support_roof_contact import audit_canopy_support_roof_contact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether a proposed canopy support has continuous roof evidence."
    )
    parser.add_argument("--cloud", required=True)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--obj", required=True)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--frame-report", required=True)
    parser.add_argument("--canopy-plane-audit", required=True)
    parser.add_argument("--roof-object", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit_canopy_support_roof_contact(
        cloud_path=args.cloud,
        candidate_report_path=args.candidate_report,
        candidate_obj_path=args.obj,
        candidate_origin_path=args.origin,
        frame_report_path=args.frame_report,
        canopy_plane_audit_path=args.canopy_plane_audit,
        roof_object_name=args.roof_object,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
