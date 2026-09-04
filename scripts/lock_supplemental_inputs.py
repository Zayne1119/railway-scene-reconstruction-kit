from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import laspy

from railway_recon.io import sha256_file, write_json


def _record(path: Path, role: str) -> dict[str, object]:
    source = path.resolve()
    with laspy.open(source) as reader:
        header = reader.header
        return {
            "role": role,
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "sha256": sha256_file(source),
            "point_count": int(header.point_count),
            "las_version": str(header.version),
            "point_format": int(header.point_format.id),
            "minimum_xyz": [float(value) for value in header.mins],
            "maximum_xyz": [float(value) for value in header.maxs],
            "scales": [float(value) for value in header.scales],
            "offsets": [float(value) for value in header.offsets],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze supplemental point-cloud roles and hashes.")
    parser.add_argument("--old", required=True, type=Path)
    parser.add_argument("--replacement", required=True, type=Path)
    parser.add_argument("--clip", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = {
        "schema_version": "railway.supplemental-input-lock.v1",
        "locked_at": datetime.now(UTC).isoformat(),
        "policy": {
            "replacement_rule": "replacement replaces old_tile; it is not appended",
            "clip_rule": "clip is a subset used for incremental refinement and QA only",
            "duplicate_guard": "old_tile + replacement and clip + replacement are prohibited",
        },
        "inputs": [
            _record(args.old, "old_tile_5_reference_only"),
            _record(args.replacement, "new_tile_5_canonical_replacement"),
            _record(args.clip, "s2050_2200_incremental_review_clip"),
        ],
    }
    write_json(args.output.resolve(), manifest)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
