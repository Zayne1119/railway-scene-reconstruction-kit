from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_recon.web_glb_export import export_obj_to_web_glb


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a named OBJ/MTL scene to a dependency-free Web GLB"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = export_obj_to_web_glb(
        args.input,
        args.output,
        report_path=args.report,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
