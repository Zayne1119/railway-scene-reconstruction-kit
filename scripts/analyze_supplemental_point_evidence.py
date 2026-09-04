from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_recon.supplemental_point_evidence import compare_supplemental_density


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare old and supplemental point density inside an incremental review clip."
    )
    parser.add_argument("--old", required=True, type=Path)
    parser.add_argument("--new-clip", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cell-size", type=float, default=0.25)
    args = parser.parse_args()
    outputs = compare_supplemental_density(
        args.old,
        args.new_clip,
        args.output,
        cell_size_m=args.cell_size,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
