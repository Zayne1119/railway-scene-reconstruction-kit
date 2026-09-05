"""Run constructed development-only observation probes; never registered test cases."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from railway_recon.synthetic_track_adaptive_stress_run import run_adaptive_stress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    report = run_adaptive_stress(args.output, args.seed)
    print(json.dumps({key: report[key] for key in ("role", "case_count", "base_layout_count")}, indent=2))


if __name__ == "__main__":
    main()
