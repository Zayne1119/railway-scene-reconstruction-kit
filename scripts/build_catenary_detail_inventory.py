from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_recon.catenary_detail_inventory import build_catenary_detail_inventory


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inventory per-mast detail completeness before point-supported refinement"
    )
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--obj", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build_catenary_detail_inventory(
        registry_path=args.registry,
        obj_path=args.obj,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
