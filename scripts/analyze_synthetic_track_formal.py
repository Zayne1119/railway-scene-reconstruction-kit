"""Register a fixed analysis plan or analyze an existing complete guarded run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from railway_recon.synthetic_track_formal_analysis import (
    analyze_guarded_run,
    create_formal_analysis_plan,
)
from railway_recon.synthetic_track_pilot import _write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register")
    register.add_argument("--protocol", type=Path, required=True)
    register.add_argument("--output", type=Path, required=True)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--run-directory", type=Path, required=True)
    analyze.add_argument("--plan", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "register":
            _write_json(args.output, create_formal_analysis_plan(args.protocol))
            print(args.output)
        else:
            result = analyze_guarded_run(args.run_directory, args.plan, args.output)
            print(json.dumps({key: result[key] for key in ("split", "layout_count", "comparisons")},
                             indent=2))
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
