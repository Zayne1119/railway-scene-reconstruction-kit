from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_recon.adjacent_context_candidate import build_adjacent_context_candidate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assemble a refined scene with one ownership-clipped adjacent context"
    )
    parser.add_argument("--source-obj", required=True, type=Path)
    parser.add_argument("--source-origin", required=True, type=Path)
    parser.add_argument("--source-registry", required=True, type=Path)
    parser.add_argument("--context-obj", required=True, type=Path)
    parser.add_argument("--context-origin", required=True, type=Path)
    parser.add_argument("--context-clip-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    outputs = build_adjacent_context_candidate(
        source_obj=args.source_obj,
        source_origin=args.source_origin,
        source_registry=args.source_registry,
        context_obj=args.context_obj,
        context_origin=args.context_origin,
        context_clip_report=args.context_clip_report,
        output_directory=args.output_dir,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
