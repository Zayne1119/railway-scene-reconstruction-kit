from __future__ import annotations

import argparse
from pathlib import Path

from railway_recon.model_support_comparison import compare_support_reports


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two per-object point-support reports.")
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(compare_support_reports(args.before, args.after, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
