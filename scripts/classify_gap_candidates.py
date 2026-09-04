from __future__ import annotations

import argparse
import json

from railway_recon.gap_candidate_disposition import write_gap_candidate_dispositions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile explicit point/photo semantic decisions for gap candidates."
    )
    parser.add_argument("--gap-report", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = write_gap_candidate_dispositions(
        gap_report_path=args.gap_report,
        review_path=args.review,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
