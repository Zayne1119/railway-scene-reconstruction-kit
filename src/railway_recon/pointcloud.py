from __future__ import annotations

from pathlib import Path
from typing import Any

import laspy


def inspect_point_cloud(path: Path) -> dict[str, Any]:
    if path.suffix.lower() not in {".las", ".laz"}:
        raise ValueError(f"Only LAS/LAZ is supported by the core CLI: {path}")
    with laspy.open(path) as reader:
        header = reader.header
        dimensions = list(header.point_format.dimension_names)
        return {
            "path": str(path),
            "file_size_bytes": path.stat().st_size,
            "las_version": str(header.version),
            "point_format": int(header.point_format.id),
            "point_count": int(header.point_count),
            "scales": [float(value) for value in header.scales],
            "offsets": [float(value) for value in header.offsets],
            "mins": [float(value) for value in header.mins],
            "maxs": [float(value) for value in header.maxs],
            "dimensions": dimensions,
            "has_rgb": all(name in dimensions for name in ("red", "green", "blue")),
            "has_gps_time": "gps_time" in dimensions,
            "note": (
                "This is a LAS/LAZ point observation audit. It does not prove that native "
                "3D Gaussian covariance, scale, opacity or visibility attributes are present."
            ),
        }

