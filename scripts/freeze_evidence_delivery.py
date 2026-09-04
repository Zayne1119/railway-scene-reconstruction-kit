from __future__ import annotations

import argparse
from pathlib import Path

from railway_recon.delivery_freeze import freeze_delivery


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze an evidence-aware candidate snapshot (not a formal release)"
    )
    parser.add_argument("--obj", required=True, type=Path)
    parser.add_argument("--mtl", required=True, type=Path)
    parser.add_argument("--glb", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--origin", required=True, type=Path)
    parser.add_argument("--final-qa", required=True, type=Path)
    parser.add_argument("--glb-conversion", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--report", action="append", default=[], type=Path)
    parser.add_argument("--view-directory", action="append", default=[], type=Path)
    args = parser.parse_args()
    result = freeze_delivery(
        source_obj=args.obj,
        source_mtl=args.mtl,
        source_glb=args.glb,
        source_registry=args.registry,
        source_origin=args.origin,
        final_qa_report=args.final_qa,
        glb_conversion_report=args.glb_conversion,
        output_directory=args.output_directory,
        report_files=args.report,
        view_directories=args.view_directory,
    )
    for name, path in result.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
