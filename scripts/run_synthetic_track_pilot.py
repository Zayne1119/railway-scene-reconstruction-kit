"""Run the customer-independent P1 track pilot in a new output directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from railway_recon.synthetic_track_pilot import run_synthetic_track_pilot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory; existing paths are refused"
    )
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    try:
        report = run_synthetic_track_pilot(args.output, seed=args.seed)
    except (FileExistsError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(
        json.dumps(
            {
                "phase": report["phase"],
                "case_count": report["case_count"],
                "base_layout_count": report["base_layout_count"],
                "output": str(args.output),
                "summary_by_mode": report["summary_by_mode"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
