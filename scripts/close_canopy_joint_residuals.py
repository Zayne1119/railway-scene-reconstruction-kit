from __future__ import annotations

import argparse

from railway_recon.canopy_joint_residual_closure import close_canopy_joint_residuals


def main() -> None:
    parser = argparse.ArgumentParser(description="Close known canopy joint/fascia residuals")
    parser.add_argument("--gap-report", required=True)
    parser.add_argument("--rejected-weld-gate", required=True)
    parser.add_argument("--local-seam-evidence", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(
        close_canopy_joint_residuals(
            gap_report_path=args.gap_report,
            rejected_weld_gate_path=args.rejected_weld_gate,
            local_seam_evidence_path=args.local_seam_evidence,
            output_path=args.output,
        )
    )


if __name__ == "__main__":
    main()
