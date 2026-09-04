from __future__ import annotations

import argparse

from railway_recon.supplemental_remote_context_facade import (
    build_supplemental_remote_context_facade,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the observed remote context facade")
    parser.add_argument("--cloud", required=True)
    parser.add_argument("--frame-report", required=True)
    parser.add_argument("--source-obj", required=True)
    parser.add_argument("--source-origin", required=True)
    parser.add_argument("--source-registry", required=True)
    parser.add_argument("--group-evidence", required=True)
    parser.add_argument("--photo-evidence", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = build_supplemental_remote_context_facade(
        cloud_path=args.cloud,
        frame_report_path=args.frame_report,
        source_obj=args.source_obj,
        source_origin=args.source_origin,
        source_registry=args.source_registry,
        group_evidence_path=args.group_evidence,
        photo_evidence_paths=tuple(args.photo_evidence),
        output_directory=args.output_dir,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
