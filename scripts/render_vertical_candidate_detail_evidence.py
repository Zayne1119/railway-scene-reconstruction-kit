from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_recon.vertical_candidate_evidence import (
    render_vertical_candidate_detail_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render dense coloured three-view evidence for standalone vertical gaps."
    )
    parser.add_argument("--cloud", required=True, type=Path)
    parser.add_argument("--gap-report", required=True, type=Path)
    parser.add_argument("--frame-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--priority", action="append", dest="priorities")
    parser.add_argument("--candidate-id", action="append", dest="candidate_ids")
    parser.add_argument(
        "--asset-relation",
        action="append",
        dest="asset_relations",
        help=(
            "Candidate asset relation to include. Defaults to "
            "new_standalone_vertical_candidate."
        ),
    )
    parser.add_argument("--current-segment-only", action="store_true")
    parser.add_argument(
        "--preserve-candidate-colour",
        action="store_true",
        help="Render candidate points with source RGB instead of lime highlighting.",
    )
    args = parser.parse_args()
    outputs = render_vertical_candidate_detail_evidence(
        cloud_path=args.cloud,
        gap_report_path=args.gap_report,
        frame_report_path=args.frame_report,
        output_directory=args.output_dir,
        priorities=tuple(args.priorities or ("P0",)),
        candidate_ids=tuple(args.candidate_ids or ()),
        asset_relations=tuple(
            args.asset_relations or ("new_standalone_vertical_candidate",)
        ),
        current_segment_only=args.current_segment_only,
        preserve_candidate_colour=args.preserve_candidate_colour,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
