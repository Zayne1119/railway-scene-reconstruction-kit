"""Initialize, freeze or run v3 six-way diagnosis; keep pressure suites separate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from railway_recon.synthetic_track_guarded_protocol import (
    create_guarded_protocol,
    freeze_guarded_protocol,
)
from railway_recon.synthetic_track_guarded_study import run_guarded_split, run_guarded_stress
from railway_recon.synthetic_track_pilot import _write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init")
    initialize.add_argument("--base-protocol", type=Path,
                            help="Existing v1 protocol for unchanged seeds/groups; otherwise v1 defaults")
    initialize.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--split", choices=("development", "validation", "test"), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--freeze", type=Path)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    stress = commands.add_parser("stress")
    stress.add_argument("--suite", choices=("legacy", "challenge"), default="legacy")
    stress.add_argument("--seed", type=int, help="Defaults: legacy 20260907; challenge 20260908")
    stress.add_argument("--policy", type=Path, help="Optional JSON StudyAuditPolicy object; actual policy is saved")
    stress.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "init":
            base = json.loads(args.base_protocol.read_text(encoding="utf-8")) if args.base_protocol else None
            _write_json(args.output, create_guarded_protocol(base_protocol=base))
            print(args.output)
        elif args.command == "freeze":
            print(freeze_guarded_protocol(args.protocol, args.output))
        else:
            if args.command == "stress":
                policy = json.loads(args.policy.read_text(encoding="utf-8")) if args.policy else None
                report = run_guarded_stress(args.output, args.suite, args.seed, policy)
            else:
                report = run_guarded_split(args.protocol, args.split, args.output, args.freeze)
            print(json.dumps({key: report[key] for key in (
                "split", "base_layout_count", "case_count", "detector_run_count")}, indent=2))
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
