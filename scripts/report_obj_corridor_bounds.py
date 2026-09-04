from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from railway_recon.model_point_support import object_vertex_indices, parse_obj_model


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report OBJ object bounds in an existing corridor frame."
    )
    parser.add_argument("--obj", required=True, type=Path)
    parser.add_argument("--origin", required=True, type=Path)
    parser.add_argument("--frame-report", required=True, type=Path)
    parser.add_argument("--name-pattern", default=".*")
    args = parser.parse_args()

    model = parse_obj_model(args.obj)
    model_origin = np.asarray(
        json.loads(args.origin.read_text(encoding="utf-8"))["origin_xyz"],
        dtype=np.float64,
    )
    frame = json.loads(args.frame_report.read_text(encoding="utf-8"))["frame"]
    frame_origin = np.asarray(frame["origin_xy"], dtype=np.float64)
    along = np.asarray(frame["along_xy"], dtype=np.float64)
    cross = np.asarray(frame["cross_xy"], dtype=np.float64)
    pattern = re.compile(args.name_pattern, flags=re.IGNORECASE)

    print("object\tstation_min_m\tstation_max_m\tcross_min_m\tcross_max_m\tz_min_m\tz_max_m")
    for name in sorted(model.faces_by_object):
        if not pattern.search(name):
            continue
        vertices = model.vertices[object_vertex_indices(model, name)] + model_origin
        delta = vertices[:, :2] - frame_origin
        station = delta @ along
        lateral = delta @ cross
        print(
            f"{name}\t{np.min(station):.3f}\t{np.max(station):.3f}"
            f"\t{np.min(lateral):.3f}\t{np.max(lateral):.3f}"
            f"\t{np.min(vertices[:, 2]):.3f}\t{np.max(vertices[:, 2]):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
