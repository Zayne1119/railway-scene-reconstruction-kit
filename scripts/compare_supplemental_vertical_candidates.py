from __future__ import annotations

import argparse
from pathlib import Path

from railway_recon.supplemental_vertical_comparison import (
    build_supplemental_vertical_comparison,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Match supplemental vertical candidates to existing model assets."
    )
    parser.add_argument("--linear-report", action="append", required=True, type=Path)
    parser.add_argument("--obj", required=True, type=Path)
    parser.add_argument("--origin", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--diagnostic", required=True, type=Path)
    args = parser.parse_args()
    print(
        build_supplemental_vertical_comparison(
            args.linear_report,
            args.obj,
            args.origin,
            args.registry,
            args.output,
            args.diagnostic,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
