from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_recon.glb_node_translation import translate_named_glb_nodes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply reviewed rigid translations to named GLB nodes."
    )
    parser.add_argument("--source-glb", required=True, type=Path)
    parser.add_argument("--output-glb", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--candidate-obj", required=True, type=Path)
    parser.add_argument("--translations", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--release-id", required=True)
    args = parser.parse_args()
    raw = json.loads(args.translations.read_text(encoding="utf-8-sig"))
    translations = {
        str(name): tuple(float(value) for value in delta)
        for name, delta in raw.items()
    }
    print(
        translate_named_glb_nodes(
            source_glb=args.source_glb,
            output_glb=args.output_glb,
            translations=translations,
            registry_path=args.registry,
            candidate_obj_path=args.candidate_obj,
            report_path=args.report,
            release_id=args.release_id,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
