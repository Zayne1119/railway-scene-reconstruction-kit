from __future__ import annotations

import argparse
from pathlib import Path

from railway_recon.vertical_gap_disposition_ledger import (
    build_vertical_gap_disposition_ledger,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a reconciled P1 gap disposition ledger")
    parser.add_argument("--baseline-gap-report", required=True, type=Path)
    parser.add_argument("--current-gap-report", required=True, type=Path)
    parser.add_argument("--candidate-gate", action="append", default=[], type=Path)
    parser.add_argument("--closure-report", action="append", default=[], type=Path)
    parser.add_argument("--position-tolerance-m", type=float, default=0.40)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        build_vertical_gap_disposition_ledger(
            baseline_gap_report_path=args.baseline_gap_report,
            current_gap_report_path=args.current_gap_report,
            candidate_gate_paths=tuple(args.candidate_gate),
            closure_report_paths=tuple(args.closure_report),
            output_path=args.output,
            position_tolerance_m=args.position_tolerance_m,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
