from __future__ import annotations

import argparse

from railway_recon.platform_clock_candidate_gate import gate_platform_clock_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate the observed platform-clock candidate")
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--baseline-gap-report", required=True)
    parser.add_argument("--candidate-gap-report", required=True)
    parser.add_argument("--fixed-view-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(
        gate_platform_clock_candidate(
            candidate_report_path=args.candidate_report,
            baseline_gap_report_path=args.baseline_gap_report,
            candidate_gap_report_path=args.candidate_gap_report,
            fixed_view_manifest_path=args.fixed_view_manifest,
            output_path=args.output,
        )
    )


if __name__ == "__main__":
    main()
