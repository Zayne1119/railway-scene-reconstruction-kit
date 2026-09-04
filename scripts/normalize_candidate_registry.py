from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_recon.candidate_registry_normalization import normalize_candidate_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize a copied candidate asset registry")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    result = normalize_candidate_registry(args.source, args.output, args.report)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
