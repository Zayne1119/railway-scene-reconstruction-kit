"""Initialize or execute the additive public TRAIN spacing sensitivity study."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from railway_recon.public_rail_spacing_study import (
    create_public_spacing_protocol,
    run_public_spacing_study,
)
from railway_recon.synthetic_track_pilot import _write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init")
    initialize.add_argument("--samples", type=Path, help="Explicit JSON array; original pilot sample must remain first")
    initialize.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    for name in ("input-directory", "metadata-directory", "receipt-directory", "protocol", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "init":
            samples = json.loads(args.samples.read_text(encoding="utf-8")) if args.samples else None
            _write_json(args.output, create_public_spacing_protocol(samples))
            print(args.output)
        else:
            report = run_public_spacing_study(args.input_directory, args.metadata_directory,
                                              args.receipt_directory, args.protocol, args.output)
            print(json.dumps({key: report[key] for key in ("status", "sample_count", "parent_cloud_count",
                                                         "pooled_descriptive_evaluation")}, indent=2))
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
