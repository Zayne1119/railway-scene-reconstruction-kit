"""Initialize, run or freeze v2 adaptive diagnosis with unchanged v1 case splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from railway_recon.synthetic_track_adaptive_protocol import (
    create_adaptive_protocol,
    freeze_adaptive_protocol,
)
from railway_recon.synthetic_track_adaptive_study import run_adaptive_split
from railway_recon.synthetic_track_pilot import _write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init")
    initialize.add_argument(
        "--base-protocol",
        type=Path,
        help="Existing v1 protocol for identical seeds/groups; otherwise use its default protocol",
    )
    initialize.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--protocol", required=True, type=Path)
    run.add_argument("--split", required=True, choices=("development", "validation", "test"))
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--freeze", type=Path)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--protocol", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "init":
            base = (
                json.loads(args.base_protocol.read_text(encoding="utf-8"))
                if args.base_protocol
                else None
            )
            _write_json(args.output, create_adaptive_protocol(base_protocol=base))
            print(args.output)
        elif args.command == "freeze":
            print(freeze_adaptive_protocol(args.protocol, args.output))
        else:
            report = run_adaptive_split(args.protocol, args.split, args.output, args.freeze)
            print(
                json.dumps(
                    {
                        key: report[key]
                        for key in (
                            "split",
                            "base_layout_count",
                            "case_count",
                            "detector_run_count",
                        )
                    },
                    indent=2,
                )
            )
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
