from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_recon.supplemental_candidate_gate import (
    gate_supplemental_conductor_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate supplemental conductor geometry using fit, mesh, support and views."
    )
    parser.add_argument("--refinement-report", required=True, type=Path)
    parser.add_argument("--mesh-audit", required=True, type=Path)
    parser.add_argument("--point-support", required=True, type=Path)
    parser.add_argument("--fixed-views", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = gate_supplemental_conductor_candidate(
        refinement_report_path=args.refinement_report,
        mesh_audit_path=args.mesh_audit,
        point_support_report_path=args.point_support,
        fixed_view_manifest_path=args.fixed_views,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
