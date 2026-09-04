from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_recon.obj_interchange import prepare_obj_interchange


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare an OBJ for deterministic Web/UE import with explicit normals."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crease-angle", type=float, default=35.0)
    args = parser.parse_args()
    report = prepare_obj_interchange(
        args.input,
        args.output,
        crease_angle_degrees=args.crease_angle,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

