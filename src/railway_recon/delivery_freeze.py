from __future__ import annotations

import copy
import shutil
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .io import load_json, sha256_file, write_json
from .registry import summarize_registry, validate_registry_value


def _copy_obj_with_mtl(source: Path, destination: Path, mtl_name: str) -> None:
    lines = source.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    replaced = False
    output: list[str] = []
    for line in lines:
        if line.lstrip().startswith("mtllib ") and not replaced:
            output.append(f"mtllib {mtl_name}")
            replaced = True
        else:
            output.append(line)
    if not replaced:
        output.insert(0, f"mtllib {mtl_name}")
    destination.write_text("\n".join(output) + "\n", encoding="utf-8", newline="\n")


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def freeze_delivery(
    *,
    source_obj: str | Path,
    source_mtl: str | Path,
    source_glb: str | Path,
    source_registry: str | Path,
    source_origin: str | Path,
    final_qa_report: str | Path,
    glb_conversion_report: str | Path,
    output_directory: str | Path,
    report_files: Iterable[str | Path] = (),
    view_directories: Iterable[str | Path] = (),
) -> dict[str, Path]:
    """Freeze an immutable *candidate* snapshot.

    This function only sees local mesh/conversion QA and therefore cannot make
    a formal release decision.  Formal/final naming belongs to the project
    release packager after its preflight and delivery gates pass.
    """
    source_obj_file = Path(source_obj).resolve()
    source_mtl_file = Path(source_mtl).resolve()
    source_glb_file = Path(source_glb).resolve()
    source_registry_file = Path(source_registry).resolve()
    source_origin_file = Path(source_origin).resolve()
    final_qa_file = Path(final_qa_report).resolve()
    conversion_file = Path(glb_conversion_report).resolve()
    required = (
        source_obj_file,
        source_mtl_file,
        source_glb_file,
        source_registry_file,
        source_origin_file,
        final_qa_file,
        conversion_file,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    qa = load_json(final_qa_file)
    conversion = load_json(conversion_file)
    if not qa.get("summary", {}).get("passed"):
        raise ValueError("Final QA report is not passing")
    if not conversion.get("passed"):
        raise ValueError("GLB conversion report is not passing")
    if qa.get("candidate_obj_sha256") != sha256_file(source_obj_file):
        raise ValueError("Final QA report does not bind the source OBJ hash")
    if conversion.get("source_sha256") != sha256_file(source_obj_file):
        raise ValueError("GLB conversion report does not bind the source OBJ hash")

    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite frozen delivery: {output}")
    output.mkdir(parents=True, exist_ok=True)
    reports_output = output / "reports"
    views_output = output / "views"
    reports_output.mkdir()
    views_output.mkdir()

    candidate_obj = output / "site_b_200m_candidate.obj"
    candidate_mtl = output / "site_b_200m_candidate.mtl"
    candidate_glb = output / "site_b_200m_candidate.glb"
    candidate_origin = output / "model_origin.json"
    candidate_registry = output / "candidate_asset_registry.json"
    _copy_obj_with_mtl(source_obj_file, candidate_obj, candidate_mtl.name)
    shutil.copy2(source_mtl_file, candidate_mtl)
    shutil.copy2(source_glb_file, candidate_glb)
    shutil.copy2(source_origin_file, candidate_origin)

    registry = copy.deepcopy(load_json(source_registry_file))
    for asset in registry.get("assets", []):
        geometry = asset.get("geometry")
        if isinstance(geometry, dict) and geometry.get("file"):
            geometry["file"] = candidate_obj.name
    registry.pop("release_id", None)
    registry["candidate_id"] = output.name
    registry["updated_at"] = datetime.now(UTC).isoformat()
    registry["model_sha256"] = sha256_file(candidate_obj)
    registry["summary"] = summarize_registry(registry)
    errors = validate_registry_value(registry)
    if errors:
        raise ValueError("Frozen registry is invalid: " + "; ".join(errors))
    write_json(candidate_registry, registry)

    report_sources = [final_qa_file, conversion_file]
    report_sources.extend(Path(value).resolve() for value in report_files)
    copied_reports: list[Path] = []
    seen_report_names: set[str] = set()
    for source in report_sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.name in seen_report_names:
            continue
        destination = reports_output / source.name
        shutil.copy2(source, destination)
        seen_report_names.add(source.name)
        copied_reports.append(destination)

    copied_views: list[Path] = []
    for directory_value in view_directories:
        directory = Path(directory_value).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        destination = views_output / directory.name
        shutil.copytree(directory, destination)
        copied_views.extend(path for path in destination.rglob("*") if path.is_file())

    primary = [candidate_obj, candidate_mtl, candidate_glb, candidate_origin, candidate_registry]
    manifest = {
        "schema_version": "railway.frozen-candidate-snapshot.v2",
        "candidate_id": output.name,
        "formal_release": False,
        "frozen_at": datetime.now(UTC).isoformat(),
        "source": {
            "obj": str(source_obj_file),
            "obj_sha256": sha256_file(source_obj_file),
            "registry": str(source_registry_file),
        },
        "primary_artifacts": [_artifact(path, output) for path in primary],
        "report_artifacts": [_artifact(path, output) for path in copied_reports],
        "view_artifacts": [_artifact(path, output) for path in copied_views],
        "qa": {
            "passed": True,
            "final_scene_check_count": qa["summary"]["check_count"],
            "final_scene_failed_check_count": qa["summary"]["failed_check_count"],
            "glb_structure_passed": True,
        },
        "scope": {
            "supported_catenary_assembly_count": qa["summary"][
                "supported_catenary_assembly_count"
            ],
            "added_catenary_component_count": qa["summary"][
                "added_catenary_component_count"
            ],
            "withheld_low_confidence_mast_count": qa["summary"][
                "withheld_low_confidence_mast_count"
            ],
            "new_unresolved_small_asset_count": qa["summary"][
                "new_unresolved_small_asset_count"
            ],
        },
        "status": "frozen_candidate_not_formal_release",
        "limitations": qa.get("limitations", []),
    }
    manifest_path = output / "candidate_snapshot_manifest.json"
    write_json(manifest_path, manifest)
    return {
        "directory": output,
        "obj": candidate_obj,
        "mtl": candidate_mtl,
        "glb": candidate_glb,
        "origin": candidate_origin,
        "registry": candidate_registry,
        "manifest": manifest_path,
    }
