from __future__ import annotations

import argparse
from pathlib import Path

from railway_recon.canopy_group_refinement import evaluate_canopy_seam_candidate_gate
from railway_recon.io import load_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail closed when a canopy seam candidate loses evidence support."
    )
    parser.add_argument("--baseline-support", required=True, type=Path)
    parser.add_argument("--candidate-support", required=True, type=Path)
    parser.add_argument("--baseline-plane", required=True, type=Path)
    parser.add_argument("--candidate-plane", required=True, type=Path)
    parser.add_argument("--weld-report", required=True, type=Path)
    parser.add_argument("--physical-seam", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate_canopy_seam_candidate_gate(
        baseline_support=load_json(args.baseline_support),
        candidate_support=load_json(args.candidate_support),
        baseline_plane_audit=load_json(args.baseline_plane),
        candidate_plane_audit=load_json(args.candidate_plane),
        weld_report=load_json(args.weld_report),
        physical_seam_audit=load_json(args.physical_seam),
    )
    write_json(args.output, result)
    print(args.output.resolve())
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
