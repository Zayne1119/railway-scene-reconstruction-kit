from __future__ import annotations

import argparse
from pathlib import Path

from railway_recon.vertical_candidate_attribute_evidence import (
    analyze_vertical_candidate_attributes,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize LAS classes, RGB and return evidence for vertical candidates."
    )
    parser.add_argument("--cloud", required=True, type=Path)
    parser.add_argument("--gap-report", required=True, type=Path)
    parser.add_argument("--candidate-id", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        analyze_vertical_candidate_attributes(
            cloud_path=args.cloud,
            gap_report_path=args.gap_report,
            candidate_ids=tuple(args.candidate_id),
            output_path=args.output,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
