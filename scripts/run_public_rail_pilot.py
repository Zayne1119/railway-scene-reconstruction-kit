"""Initialize or execute the bounded public TRAIN rail-candidate pilot."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from railway_recon.public_rail_pilot import create_public_rail_protocol, run_public_rail_pilot
from railway_recon.synthetic_track_pilot import _write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init")
    initialize.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    for name in ("input", "metadata-directory", "protocol", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "init":
            _write_json(args.output, create_public_rail_protocol())
            print(args.output)
        else:
            report = run_public_rail_pilot(args.input, args.metadata_directory, args.protocol, args.output)
            print(json.dumps({key: report[key] for key in ("status", "point_count", "candidate_summary")}, indent=2))
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
