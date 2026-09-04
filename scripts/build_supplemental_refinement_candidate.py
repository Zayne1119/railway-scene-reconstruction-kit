from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_recon.supplemental_mesh_refinement import (
    build_supplemental_refinement_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a non-destructive supplemental mesh candidate.")
    parser.add_argument("--obj", required=True, type=Path)
    parser.add_argument("--mtl", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--origin", required=True, type=Path)
    parser.add_argument("--support-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    outputs = build_supplemental_refinement_candidate(
        args.obj,
        args.mtl,
        args.registry,
        args.origin,
        args.support_report,
        args.output,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
