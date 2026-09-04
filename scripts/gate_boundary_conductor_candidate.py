from __future__ import annotations

import argparse
from pathlib import Path

from railway_recon.boundary_conductor_candidate_gate import (
    gate_boundary_conductor_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate outbound boundary conductor candidate")
    parser.add_argument("--candidate-report", required=True, type=Path)
    parser.add_argument("--baseline-gap-report", required=True, type=Path)
    parser.add_argument("--candidate-gap-report", required=True, type=Path)
    parser.add_argument("--fixed-view-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        gate_boundary_conductor_candidate(
            candidate_report_path=args.candidate_report,
            baseline_gap_report_path=args.baseline_gap_report,
            candidate_gap_report_path=args.candidate_gap_report,
            fixed_view_manifest_path=args.fixed_view_manifest,
            output_path=args.output,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
