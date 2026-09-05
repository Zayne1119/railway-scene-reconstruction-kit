"""Initialize, run, or freeze a synthetic diagnosis study. Never runs test by default."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from railway_recon.synthetic_track_pilot import _write_json
from railway_recon.synthetic_track_study import run_study_split
from railway_recon.synthetic_track_study_protocol import (
    create_study_protocol,
    freeze_study_protocol,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init")
    initialize.add_argument("--output", required=True, type=Path)
    initialize.add_argument("--seed", type=int, default=20260906)
    run = commands.add_parser("run")
    run.add_argument("--protocol", required=True, type=Path)
    run.add_argument("--split", required=True, choices=("development", "validation", "test"))
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--freeze", type=Path)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--protocol", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "init":
        _write_json(args.output, create_study_protocol(seed=args.seed))
        print(args.output)
    elif args.command == "freeze":
        print(freeze_study_protocol(args.protocol, args.output))
    else:
        result = run_study_split(args.protocol, args.split, args.output, args.freeze)
        print(json.dumps({key: result[key] for key in ("split", "base_layout_count", "case_count")}, indent=2))


if __name__ == "__main__":
    main()
