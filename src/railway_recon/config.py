from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator
except ImportError:  # The built-in validation still supports offline field work.
    Draft202012Validator = None  # type: ignore[assignment]

from .io import load_json, write_json

PROJECT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{2,63}$")


@dataclass(frozen=True)
class ProjectConfig:
    path: Path
    value: dict[str, Any]

    @property
    def root(self) -> Path:
        return self.path.parent

    @property
    def project_id(self) -> str:
        return str(self.value["project"]["id"])

    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.root / path).resolve()

    def input_path(self, key: str) -> Path | None:
        value = self.value.get("inputs", {}).get(key)
        return self.resolve(value) if value else None

    def workspace_path(self, key: str) -> Path:
        value = self.value.get("workspace", {}).get(key)
        if not value:
            raise KeyError(f"workspace.{key} is not configured")
        path = self.resolve(value)
        root = self.root.resolve()
        if path != root and root not in path.parents:
            raise ValueError(f"workspace.{key} must stay inside the project directory: {path}")
        return path


def _resource_json(name: str) -> dict[str, Any]:
    resource = resources.files("railway_recon.resources").joinpath(name)
    with resource.open("r", encoding="utf-8") as stream:
        import json

        return json.load(stream)


def validate_project_value(value: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if Draft202012Validator is not None:
        validator = Draft202012Validator(_resource_json("project.schema.json"))
        errors.extend(
            f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
            for error in sorted(validator.iter_errors(value), key=lambda item: list(item.path))
        )
    else:
        required = (
            "schema_version",
            "project",
            "inputs",
            "workspace",
            "segmentation",
            "reconstruction",
            "quality",
            "algorithms",
            "rules",
        )
        errors.extend(
            f"<root>: missing required property '{key}'" for key in required if key not in value
        )
        if value.get("schema_version") != "railway.toolkit.project.v1":
            errors.append("schema_version must be 'railway.toolkit.project.v1'")
        nested_required = {
            "project": ("id", "name", "units", "axis", "crs"),
            "inputs": ("camera_csv",),
            "workspace": (
                "manifests",
                "segment_manifest",
                "segments",
                "derived",
                "asset_registry",
                "reports",
                "exports",
            ),
            "segmentation": (
                "length_m",
                "corridor_half_width_m",
                "z_below_camera_m",
                "z_above_camera_m",
                "chunk_size_points",
            ),
            "algorithms": (
                "rail_detection",
                "linear_detection",
                "track_build",
                "projection_calibration",
            ),
        }
        for section, keys in nested_required.items():
            section_value = value.get(section)
            if not isinstance(section_value, dict):
                errors.append(f"{section}: expected an object")
                continue
            errors.extend(
                f"{section}: missing required property '{key}'"
                for key in keys
                if key not in section_value
            )
    project_section = value.get("project")
    project_values = project_section if isinstance(project_section, dict) else {}
    project_id = str(project_values.get("id", ""))
    if project_id and not PROJECT_ID_PATTERN.fullmatch(project_id):
        errors.append(
            "project.id must start with a lowercase letter and contain only lowercase "
            "letters, digits, '-' or '_' (3-64 characters)"
        )
    if project_values.get("axis") != "Z-up":
        errors.append("project.axis must be 'Z-up' in toolkit v0.1")
    inputs = value.get("inputs")
    input_values = inputs if isinstance(inputs, dict) else {}
    has_single = bool(input_values.get("point_cloud"))
    has_multiple = "point_clouds" in input_values
    if has_single == has_multiple:
        errors.append("inputs must configure exactly one of point_cloud or point_clouds")
    return errors


def load_project(path: str | Path) -> ProjectConfig:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    value = load_json(config_path)
    errors = validate_project_value(value)
    if errors:
        raise ValueError("Invalid project config:\n- " + "\n- ".join(errors))
    return ProjectConfig(config_path, value)


def initialize_project(target: str | Path, project_id: str, name: str) -> Path:
    root = Path(target).resolve()
    if root.exists() and not root.is_dir():
        raise FileExistsError(f"Target exists and is not a directory: {root}")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Target directory is not empty: {root}")
    if not PROJECT_ID_PATTERN.fullmatch(project_id):
        raise ValueError("Invalid project id; use lowercase letters, digits, '-' or '_'")

    root.mkdir(parents=True, exist_ok=True)
    for relative in (
        "input/pointcloud",
        "input/panoramas",
        "workspace/manifests",
        "workspace/gates",
        "workspace/segments",
        "workspace/derived",
        "workspace/registry",
        "workspace/reports",
        "workspace/exports",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)

    project = _resource_json("project.template.json")
    project["project"]["id"] = project_id
    project["project"]["name"] = name
    write_json(root / "project.json", project)
    write_json(root / "rules.json", _resource_json("rules.default.json"))
    write_json(root / "rail_detection.json", _resource_json("rail-detection.default.json"))
    write_json(root / "linear_detection.json", _resource_json("linear-detection.default.json"))
    write_json(
        root / "vertical_hypotheses.json",
        _resource_json("vertical-hypotheses.default.json"),
    )
    write_json(
        root / "canopy_structure.json",
        _resource_json("canopy-structure.default.json"),
    )
    write_json(
        root / "canopy_photo_evidence.json",
        _resource_json("canopy-photo-evidence.default.json"),
    )
    write_json(
        root / "platform_surface.json",
        _resource_json("platform-surface.default.json"),
    )
    write_json(
        root / "platform_photo_evidence.json",
        _resource_json("platform-photo-evidence.default.json"),
    )
    write_json(
        root / "platform_mesh.json",
        _resource_json("platform-mesh.default.json"),
    )
    write_json(
        root / "platform_interface_audit.json",
        _resource_json("platform-interface-audit.default.json"),
    )
    write_json(
        root / "vertical_conflict_photo_evidence.json",
        _resource_json("vertical-conflict-photo-evidence.default.json"),
    )
    write_json(
        root / "targeted_canopy_recovery.json",
        _resource_json("targeted-canopy-recovery.default.json"),
    )
    write_json(root / "track_build.json", _resource_json("track-build.default.json"))
    write_json(root / "track_graph.json", _resource_json("track-graph.default.json"))
    write_json(
        root / "projection_calibration.json",
        _resource_json("projection-calibration.default.json"),
    )

    note = root / "input" / "README.txt"
    note.write_text(
        "Production inputs stay in this directory and are excluded from Git.\n"
        "Expected: pointcloud/site.laz, cameras.csv, panoramas/.\n",
        encoding="utf-8",
    )
    return root / "project.json"
