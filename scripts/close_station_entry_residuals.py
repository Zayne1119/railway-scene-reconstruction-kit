from __future__ import annotations

import argparse
from pathlib import Path

from railway_recon.station_entry_residual_closure import (
    close_station_entry_surface_residuals,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Close sparse station-entry surface residuals without fake assets"
    )
    parser.add_argument("--gap-report", required=True, type=Path)
    parser.add_argument("--detail-evidence", required=True, type=Path)
    parser.add_argument("--photo-evidence", required=True, type=Path)
    parser.add_argument("--attribute-evidence", required=True, type=Path)
    parser.add_argument("--surface-analysis", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument(
        "--decision",
        action="append",
        required=True,
        help="candidate_id=existing_station_entry_blue_surface_edge or "
        "candidate_id=sparse_station_entry_context_residual",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    decisions: dict[str, str] = {}
    for raw in args.decision:
        candidate_id, separator, decision = raw.partition("=")
        if not separator or not candidate_id or not decision:
            parser.error(f"Invalid --decision value: {raw}")
        decisions[candidate_id] = decision
    print(
        close_station_entry_surface_residuals(
            gap_report_path=args.gap_report,
            detail_evidence_path=args.detail_evidence,
            photo_evidence_path=args.photo_evidence,
            attribute_evidence_path=args.attribute_evidence,
            surface_analysis_path=args.surface_analysis,
            model_path=args.model,
            registry_path=args.registry,
            decisions=decisions,
            output_path=args.output,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
