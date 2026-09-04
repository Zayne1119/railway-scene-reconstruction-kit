from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_recon.adjacent_candidate_closure import build_adjacent_candidate_closure


def main() -> int:
    parser = argparse.ArgumentParser(description="Close all adjacent-boundary candidate gates")
    parser.add_argument("--final-obj", required=True, type=Path)
    parser.add_argument("--mesh-audit", required=True, type=Path)
    parser.add_argument("--seam-audit", required=True, type=Path)
    parser.add_argument("--column-report", required=True, type=Path)
    parser.add_argument("--conductor-report", required=True, type=Path)
    parser.add_argument("--surface-report", required=True, type=Path)
    parser.add_argument("--transition-report", required=True, type=Path)
    parser.add_argument("--normalized-registry", required=True, type=Path)
    parser.add_argument("--fixed-view-manifest", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build_adjacent_candidate_closure(
        final_obj=args.final_obj,
        mesh_audit=args.mesh_audit,
        seam_audit=args.seam_audit,
        column_report=args.column_report,
        conductor_report=args.conductor_report,
        surface_report=args.surface_report,
        transition_report=args.transition_report,
        normalized_registry=args.normalized_registry,
        fixed_view_manifests=args.fixed_view_manifest,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
