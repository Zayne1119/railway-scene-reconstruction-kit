from __future__ import annotations

import argparse
from pathlib import Path

from railway_recon.evidence_closure import build_no_geometry_evidence_closure


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hash-bind a no-geometry evidence review to an accepted model."
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--mesh-audit", required=True, type=Path)
    parser.add_argument("--vertical-decisions", required=True, type=Path)
    parser.add_argument("--canopy-audit", required=True, type=Path)
    parser.add_argument("--platform-support", required=True, type=Path)
    parser.add_argument("--platform-seams", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        build_no_geometry_evidence_closure(
            model_path=args.model,
            registry_path=args.registry,
            mesh_audit_path=args.mesh_audit,
            vertical_decisions_path=args.vertical_decisions,
            canopy_audit_path=args.canopy_audit,
            platform_support_path=args.platform_support,
            platform_seam_audit_path=args.platform_seams,
            output_path=args.output,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
