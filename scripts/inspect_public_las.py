"""Inspect one hash-pinned public LAS/LAZ file offline, without creating model truth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from railway_recon.public_las_intake import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_POINTS,
    inspect_public_las,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="input_path", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New JSON report; existing paths are refused"
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_MAX_POINTS,
        help="Reject larger files; never truncate",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()
    try:
        report = inspect_public_las(
            args.input_path,
            args.expected_sha256,
            args.output,
            max_points=args.max_points,
            chunk_size=args.chunk_size,
        )
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": report["source"]["sha256"],
                "point_count": report["decoded"]["point_count"],
                "las_version": report["header"]["las_version"],
                "point_format_id": report["header"]["point_format_id"],
                "model_evaluation_performed": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
