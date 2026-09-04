from __future__ import annotations

import argparse
from pathlib import Path

from railway_recon.supplemental_vertical_comparison import (
    build_photo_review_vertical_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare supplemental vertical candidates for panorama review."
    )
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--frame-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(build_photo_review_vertical_report(args.comparison, args.frame_report, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
