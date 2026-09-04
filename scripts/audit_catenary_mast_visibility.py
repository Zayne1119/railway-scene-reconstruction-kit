from __future__ import annotations

import argparse
from pathlib import Path

from railway_recon.vertical_visibility import audit_catenary_mast_visibility


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit catenary mast visibility by height.")
    parser.add_argument("--obj", required=True, type=Path)
    parser.add_argument("--origin", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--cloud", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        audit_catenary_mast_visibility(
            args.obj, args.origin, args.registry, args.cloud, args.output
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
