from __future__ import annotations

import argparse
from pathlib import Path

from railway_recon.high_priority_gap_completion import (
    build_high_priority_gap_completion_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build current-segment high-priority gap completion report")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--current-gap-report", required=True, type=Path)
    parser.add_argument("--vertical-ledger", required=True, type=Path)
    parser.add_argument("--candidate-gate", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        build_high_priority_gap_completion_report(
            model_path=args.model,
            registry_path=args.registry,
            current_gap_report_path=args.current_gap_report,
            vertical_disposition_ledger_path=args.vertical_ledger,
            candidate_gate_paths=tuple(args.candidate_gate),
            output_path=args.output,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
