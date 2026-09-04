from __future__ import annotations

import argparse
from pathlib import Path

from railway_recon.external_context_post_closure import (
    close_external_context_post_row,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Close a photo-reviewed non-railway background post row."
    )
    parser.add_argument("--gap-report", required=True, type=Path)
    parser.add_argument("--detail-evidence", required=True, type=Path)
    parser.add_argument("--photo-evidence", required=True, type=Path)
    parser.add_argument("--attribute-evidence", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--candidate-id", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        close_external_context_post_row(
            gap_report_path=args.gap_report,
            detail_evidence_path=args.detail_evidence,
            photo_evidence_path=args.photo_evidence,
            attribute_evidence_path=args.attribute_evidence,
            model_path=args.model,
            registry_path=args.registry,
            candidate_ids=tuple(args.candidate_id),
            output_path=args.output,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
