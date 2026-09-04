from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_recon.gap_candidate_photo_evidence import (
    render_gap_candidate_photo_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Project cloud-model gap candidates into calibrated panoramas."
    )
    parser.add_argument("--gap-report", required=True, type=Path)
    parser.add_argument("--frame-report", required=True, type=Path)
    parser.add_argument("--projection-consensus", required=True, type=Path)
    parser.add_argument("--candidate-id", action="append", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--settings", type=Path)
    args = parser.parse_args()
    outputs = render_gap_candidate_photo_evidence(
        gap_report_path=args.gap_report,
        frame_report_path=args.frame_report,
        projection_consensus_path=args.projection_consensus,
        candidate_ids=args.candidate_id,
        output_directory=args.output_dir,
        settings_path=args.settings,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
