from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .io import sha256_file, write_json
from .mesh_audit import audit_obj
from .targeted_canopy_integration import merge_candidate_objs


def parse_mesh_source_specs(values: list[str]) -> list[tuple[str, Path]]:
    sources = []
    for value in values:
        namespace, separator, path = value.partition("=")
        if not separator or not namespace or not path:
            raise ValueError("Mesh source must use NAMESPACE=OBJ_PATH")
        sources.append((namespace, Path(path)))
    if len({namespace for namespace, _ in sources}) != len(sources):
        raise ValueError("Mesh source namespaces must be unique")
    return sources


def assemble_candidate_meshes(
    project: ProjectConfig,
    output_dir_value: str | Path,
    output_name: str,
    sources: list[tuple[str, Path]],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Merge audited candidate OBJs while preserving a namespace per source."""

    if len(sources) < 2:
        raise ValueError("At least two mesh sources are required")
    if not output_name or Path(output_name).name != output_name:
        raise ValueError("output_name must be one safe path component")
    namespaces = [namespace for namespace, _ in sources]
    if len(set(namespaces)) != len(namespaces):
        raise ValueError("Mesh source namespaces must be unique")
    resolved = []
    for namespace, value in sources:
        obj = project.resolve(value)
        origin = obj.parent / "model_origin.json"
        if not obj.is_file():
            raise FileNotFoundError(obj)
        if not origin.is_file():
            raise FileNotFoundError(origin)
        resolved.append((namespace, obj, origin))
    output_dir = project.resolve(output_dir_value)
    output_obj = output_dir / f"{output_name}.obj"
    output_mtl = output_dir / f"{output_name}.mtl"
    output_origin = output_dir / "model_origin.json"
    output_audit = output_dir / "mesh_audit.json"
    output_report = output_dir / "assembly_report.json"
    outputs = (output_obj, output_mtl, output_origin, output_audit, output_report)
    if not overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite mesh assembly outputs: {existing}")
    merge = merge_candidate_objs(
        [(obj, origin) for _, obj, origin in resolved],
        output_obj,
        output_mtl,
        output_origin,
        source_namespaces=namespaces,
    )
    audit = audit_obj(output_obj)
    audit["status"] = "pass" if audit["passed"] else "fail"
    write_json(output_audit, audit)
    if not audit["passed"]:
        raise ValueError("Candidate mesh assembly failed mesh audit")
    report = {
        "schema_version": "railway.candidate-mesh-assembly.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "project_id": project.project_id,
        "output_name": output_name,
        "component_count": len(resolved),
        "components": [
            {
                "namespace": namespace,
                "obj": str(obj),
                "obj_sha256": sha256_file(obj),
                "origin": str(origin),
            }
            for namespace, obj, origin in resolved
        ],
        "merge": merge,
        "output_obj": str(output_obj),
        "output_mtl": str(output_mtl),
        "output_origin": str(output_origin),
        "output_mesh_audit": str(output_audit),
        "output_obj_sha256": sha256_file(output_obj),
        "passed": True,
        "status": "candidate_mesh_assembly_written_not_formal_release",
    }
    write_json(output_report, report)
    return report
