from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .camera import audit_camera_rows, load_camera_rows
from .config import ProjectConfig
from .io import sha256_file, sha256_json, write_json
from .pointcloud import inspect_point_cloud


def audit_project(project: ProjectConfig, full_hash: bool = False) -> dict[str, Any]:
    point_cloud = project.input_path("point_cloud")
    camera_csv = project.input_path("camera_csv")
    panorama_root = project.input_path("panorama_root")
    required = {"point_cloud": point_cloud, "camera_csv": camera_csv}
    missing = [name for name, path in required.items() if path is None or not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required project inputs: {missing}")

    assert point_cloud is not None
    assert camera_csv is not None
    camera_rows = load_camera_rows(camera_csv)
    input_hashes: dict[str, str] = {"camera_csv": sha256_file(camera_csv)}
    if full_hash:
        input_hashes["point_cloud"] = sha256_file(point_cloud)
    else:
        input_hashes["point_cloud"] = "not_computed_use_--full-hash_for_release"

    referenced_photos = [str(row["file"]) for row in camera_rows]
    photo_check: dict[str, Any] = {
        "configured": panorama_root is not None,
        "root_exists": bool(panorama_root and panorama_root.is_dir()),
        "referenced_count": len(referenced_photos),
    }
    if panorama_root and panorama_root.is_dir():
        missing_photos = [
            name
            for name in referenced_photos
            if not (panorama_root / name).is_file()
            and not (panorama_root / Path(name).name).is_file()
        ]
        photo_check.update(
            {
                "missing_count": len(missing_photos),
                "missing_examples": missing_photos[:20],
            }
        )

    result = {
        "schema_version": "railway.project-audit.v1",
        "project_id": project.project_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "config_path": str(project.path),
        "config_sha256": sha256_json(project.value),
        "point_cloud": inspect_point_cloud(point_cloud),
        "camera_csv": audit_camera_rows(camera_rows),
        "panoramas": photo_check,
        "input_hashes": input_hashes,
        "precision_statement": (
            "Absolute survey accuracy is not claimable unless CRS, vertical datum and control "
            "points are all confirmed. Internal point-to-model fit may still be reported."
        ),
    }
    write_json(project.workspace_path("reports") / "input_audit.json", result)
    return result

