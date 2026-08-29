from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .io import load_json, sha256_file, write_json
from .registry import validate_registry_value

GATE_STATUSES = {"PASS", "FAIL", "PASS_WITH_WAIVER", "NOT_APPLICABLE"}
PASS_SOURCE_STATUSES = {"pass", "passed"}
GATE_ID_PATTERN = re.compile(r"^QG[0-9]+$")
RELEASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,95}$")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_project_path(project: ProjectConfig, value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (project.root / path).resolve()
    root = project.root.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Quality Gate input must stay inside the project directory: {resolved}")
    return resolved


def _relative_to_project(project: ProjectConfig, path: Path) -> str:
    return path.resolve().relative_to(project.root.resolve()).as_posix()


def _file_record(project: ProjectConfig, role: str, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "role": role,
        "path": _relative_to_project(project, path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _source_report_passed(value: dict[str, Any]) -> tuple[bool, str, str]:
    raw_status = value.get("status")
    if raw_status is not None:
        normalized = str(raw_status).strip()
        if normalized in {"PASS", "PASS_WITH_WAIVER"}:
            return True, normalized, "Quality Gate source status is passing"
        if normalized.lower() in PASS_SOURCE_STATUSES:
            return True, normalized, "Legacy source status is explicitly passing"
        return (
            False,
            normalized,
            (
                "Only explicit pass/PASS/PASS_WITH_WAIVER is accepted; candidate, "
                "conditional, pending, review and unknown states are blocking"
            ),
        )
    passed = value.get("passed")
    if isinstance(passed, bool):
        return passed, str(passed).lower(), "Boolean report result"
    return False, "missing", "Report contains neither an explicit status nor a boolean passed"


def _report_check(
    project: ProjectConfig,
    check_id: str,
    source: str | Path,
) -> dict[str, Any]:
    path = _safe_project_path(project, source)
    if not path.is_file():
        return {
            "id": check_id,
            "passed": False,
            "waivable": False,
            "source": _relative_to_project(project, path),
            "reason": "Required check report does not exist",
        }
    try:
        value = load_json(path)
        passed, raw_status, reason = _source_report_passed(value)
    except (OSError, ValueError) as exc:
        return {
            "id": check_id,
            "passed": False,
            "waivable": False,
            "source": _relative_to_project(project, path),
            "source_sha256": sha256_file(path),
            "reason": f"Cannot read check report: {exc}",
        }
    return {
        "id": check_id,
        "passed": passed,
        "waivable": True,
        "source": _relative_to_project(project, path),
        "source_sha256": sha256_file(path),
        "raw_status": raw_status,
        "reason": reason,
    }


def _registry_checks(
    project: ProjectConfig,
    registry_source: str | Path | None,
    required: bool,
    release_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if registry_source is None:
        return [
            {
                "id": "registry.required",
                "passed": not required,
                "waivable": False,
                "reason": "A clean release registry is required" if required else "Not requested",
            }
        ], None
    path = _safe_project_path(project, registry_source)
    if not path.is_file():
        return [
            {
                "id": "registry.exists",
                "passed": False,
                "waivable": False,
                "source": _relative_to_project(project, path),
                "reason": "Registry file does not exist",
            }
        ], None
    registry = load_json(path)
    validation_errors = validate_registry_value(registry)
    assets = list(registry.get("assets", []))
    statuses = Counter(str(asset.get("status", "missing")) for asset in assets)
    blocking = sorted(
        str(asset.get("id")) for asset in assets if asset.get("status") != "accepted"
    )
    asset_ids = sorted(str(asset.get("id")) for asset in assets)
    asset_set_sha256 = hashlib.sha256("\n".join(asset_ids).encode("utf-8")).hexdigest()
    record = {
        "source": _relative_to_project(project, path),
        "source_sha256": sha256_file(path),
    }
    checks = [
        {
            "id": "registry.valid",
            "passed": not validation_errors,
            "waivable": False,
            "errors": validation_errors,
            "reason": "Registry schema and references must be valid",
            **record,
        },
        {
            "id": "registry.project_id_matches",
            "passed": registry.get("project_id") == project.project_id,
            "waivable": False,
            "actual": registry.get("project_id"),
            "expected": project.project_id,
            "reason": "Registry and project identities must match",
            **record,
        },
        {
            "id": "registry.clean_states",
            "passed": bool(assets) and not blocking,
            "waivable": False,
            "status_counts": dict(sorted(statuses.items())),
            "blocking_asset_ids": blocking[:100],
            "blocking_asset_count": len(blocking),
            "reason": "Clean release registries may contain accepted assets only",
            **record,
        },
    ]
    if required:
        checks.append(
            {
                "id": "registry.release_id_matches",
                "passed": registry.get("release_id") == release_id,
                "waivable": False,
                "actual": registry.get("release_id"),
                "expected": release_id,
                "reason": "Release registry must bind the current release_id",
                **record,
            }
        )
    return checks, {
        "path": record["source"],
        "sha256": record["source_sha256"],
        "release_id": registry.get("release_id"),
        "asset_count": len(asset_ids),
        "asset_set_sha256": asset_set_sha256,
    }


def _previous_gate_check(
    project: ProjectConfig,
    previous_source: str | Path | None,
    release_id: str,
    gate_id: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if previous_source is None:
        if gate_id == "QG0":
            return None, None
        return None, {
            "id": "previous_gate.required",
            "passed": False,
            "waivable": False,
            "reason": f"{gate_id} requires the immediately preceding Gate result",
        }
    path = _safe_project_path(project, previous_source)
    if not path.is_file():
        return None, {
            "id": "previous_gate.exists",
            "passed": False,
            "waivable": False,
            "reason": "Previous gate result does not exist",
        }
    value = load_json(path)
    expected_previous_id = f"QG{int(gate_id[2:]) - 1}" if gate_id != "QG0" else None
    passed = (
        value.get("status") in {"PASS", "PASS_WITH_WAIVER"}
        and value.get("project_id") == project.project_id
        and value.get("release_id") == release_id
        and value.get("gate_id") == expected_previous_id
    )
    record = {
        "path": _relative_to_project(project, path),
        "sha256": sha256_file(path),
        "gate_id": value.get("gate_id"),
        "status": value.get("status"),
    }
    check = {
        "id": "previous_gate.passed_and_matches",
        "passed": passed,
        "waivable": False,
        "actual": {
            "project_id": value.get("project_id"),
            "release_id": value.get("release_id"),
            "status": value.get("status"),
        },
        "expected": {
            "project_id": project.project_id,
            "release_id": release_id,
            "gate_id": expected_previous_id,
            "status": ["PASS", "PASS_WITH_WAIVER"],
        },
        "reason": "Gate chain cannot continue from a failed or mismatched previous gate",
    }
    return record, check


def _load_objects(project: ProjectConfig, sources: list[str | Path]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in sources:
        path = _safe_project_path(project, source)
        value = load_json(path)
        value["source_path"] = _relative_to_project(project, path)
        value["source_sha256"] = sha256_file(path)
        result.append(value)
    return result


def _approval_checks(
    approvals: list[dict[str, Any]],
    required_roles: set[str],
    artifact_hashes: set[str],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for role in sorted(required_roles):
        matching = [
            approval
            for approval in approvals
            if approval.get("role") == role and approval.get("decision") == "approved"
        ]
        valid = [
            approval
            for approval in matching
            if set(approval.get("artifact_sha256", []))
            and set(approval.get("artifact_sha256", [])).issubset(artifact_hashes)
        ]
        checks.append(
            {
                "id": f"approval.{role}",
                "passed": bool(valid),
                "waivable": False,
                "matching_approval_ids": [item.get("approval_id") for item in valid],
                "reason": "Required approval must bind the current artifact hashes",
            }
        )
    return checks


def _release_artifact_checks(
    project: ProjectConfig,
    gate_id: str,
    release_id: str,
    records: list[dict[str, Any]],
    registry_identity: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if gate_id not in {"QG7", "QG8", "QG9"}:
        return []
    by_role = {str(record["role"]): record for record in records}
    required_roles = {"model", "registry"}
    if gate_id in {"QG8", "QG9"}:
        required_roles.add("web_config")
    checks: list[dict[str, Any]] = []
    for role in sorted(required_roles):
        checks.append(
            {
                "id": f"release_artifact.{role}.required",
                "passed": role in by_role,
                "waivable": False,
                "reason": f"{gate_id} requires an artifact with role={role}",
            }
        )
    registry_record = by_role.get("registry")
    if registry_record and registry_identity:
        checks.append(
            {
                "id": "release_artifact.registry_matches_checked_registry",
                "passed": registry_record["sha256"] == registry_identity["sha256"],
                "waivable": False,
                "actual": registry_record["sha256"],
                "expected": registry_identity["sha256"],
                "reason": "The packaged registry must be the registry evaluated by the Gate",
            }
        )
    web_record = by_role.get("web_config")
    if web_record:
        config = load_json(project.root / web_record["path"])
        expected = {
            "release_id": release_id,
            "model_sha256": by_role.get("model", {}).get("sha256"),
            "registry_sha256": by_role.get("registry", {}).get("sha256"),
            "asset_set_sha256": (
                registry_identity.get("asset_set_sha256") if registry_identity else None
            ),
        }
        actual = {key: config.get(key) for key in expected}
        checks.append(
            {
                "id": "release_artifact.web_config_bindings",
                "passed": actual == expected and config.get("acceptance_mode") is True,
                "waivable": False,
                "actual": {**actual, "acceptance_mode": config.get("acceptance_mode")},
                "expected": {**expected, "acceptance_mode": True},
                "reason": "Web config must bind the exact release, model, registry and asset set",
            }
        )
    return checks


def _valid_waiver(
    waiver: dict[str, Any],
    check_id: str,
    artifact_hashes: set[str],
) -> tuple[bool, str]:
    if waiver.get("check_id") != check_id:
        return False, "check_id does not match"
    hashes = waiver.get("artifact_sha256", [])
    if isinstance(hashes, str):
        hashes = [hashes]
    if not hashes or not set(hashes).issubset(artifact_hashes):
        return False, "waiver is not bound to the current artifacts"
    approvers = waiver.get("approved_by", [])
    names = {
        str(item.get("reviewer") if isinstance(item, dict) else item).strip()
        for item in approvers
    }
    names.discard("")
    if len(names) < 2:
        return False, "waiver requires at least two distinct approvers"
    expires_at = waiver.get("expires_at")
    try:
        expires = datetime.fromisoformat(str(expires_at))
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
    except (TypeError, ValueError):
        return False, "waiver expires_at is invalid"
    if expires <= datetime.now(UTC):
        return False, "waiver is expired"
    for field in ("waiver_id", "reason", "risk", "mitigation"):
        if not str(waiver.get(field, "")).strip():
            return False, f"waiver is missing {field}"
    return True, "valid"


def evaluate_quality_gate(
    project: ProjectConfig,
    release_id: str,
    gate_id: str,
    report_sources: list[tuple[str, str | Path]],
    artifacts: list[tuple[str, str | Path]],
    *,
    previous_gate: str | Path | None = None,
    registry: str | Path | None = None,
    waiver_sources: list[str | Path] | None = None,
    approval_sources: list[str | Path] | None = None,
    allow_waiver: set[str] | None = None,
    required_approval_roles: set[str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    if not RELEASE_ID_PATTERN.fullmatch(release_id):
        raise ValueError("release_id must use 3-96 letters, digits, '.', '_' or '-'")
    if not GATE_ID_PATTERN.fullmatch(gate_id):
        raise ValueError("gate_id must look like QG0, QG1, ...")
    duplicate_check_ids = [
        value for value, count in Counter(item[0] for item in report_sources).items() if count > 1
    ]
    if duplicate_check_ids:
        raise ValueError(f"Duplicate Gate check ids: {sorted(duplicate_check_ids)}")
    reserved_prefixes = (
        "approval.",
        "artifact.",
        "artifacts.",
        "previous_gate.",
        "registry.",
        "release_artifact.",
    )
    reserved_check_ids = sorted(
        check_id
        for check_id, _ in report_sources
        if check_id.startswith(reserved_prefixes)
    )
    if reserved_check_ids:
        raise ValueError(f"Gate check ids use reserved prefixes: {reserved_check_ids}")
    duplicate_roles = [
        value for value, count in Counter(item[0] for item in artifacts).items() if count > 1
    ]
    if duplicate_roles:
        raise ValueError(f"Duplicate artifact roles: {sorted(duplicate_roles)}")

    configured_gate_root = project.value.get("workspace", {}).get("gates")
    gate_root = (
        project.workspace_path("gates")
        if configured_gate_root
        else (project.root / "workspace" / "gates").resolve()
    )
    gate_dir = gate_root / release_id / gate_id
    if gate_dir.exists():
        raise FileExistsError(f"Gate evidence is immutable; use a new release id: {gate_dir}")

    artifact_records: list[dict[str, Any]] = []
    checks = [_report_check(project, check_id, source) for check_id, source in report_sources]
    for role, source in artifacts:
        path = _safe_project_path(project, source)
        try:
            artifact_records.append(_file_record(project, role, path))
        except FileNotFoundError:
            checks.append(
                {
                    "id": f"artifact.{role}.exists",
                    "passed": False,
                    "waivable": False,
                    "source": _relative_to_project(project, path),
                    "reason": "Required artifact does not exist",
                }
            )
    if gate_id in {"QG7", "QG8", "QG9"} and not artifact_records:
        checks.append(
            {
                "id": "artifacts.required",
                "passed": False,
                "waivable": False,
                "reason": "Release, runtime and final gates require hashed artifacts",
            }
        )

    previous_record, previous_check = _previous_gate_check(
        project, previous_gate, release_id, gate_id
    )
    if previous_check is not None:
        checks.append(previous_check)
    registry_checks, registry_identity = _registry_checks(
        project,
        registry,
        required=gate_id in {"QG7", "QG8", "QG9"},
        release_id=release_id,
    )
    checks.extend(registry_checks)
    checks.extend(
        _release_artifact_checks(
            project, gate_id, release_id, artifact_records, registry_identity
        )
    )

    waivers = _load_objects(project, waiver_sources or [])
    approvals = _load_objects(project, approval_sources or [])
    artifact_hashes = {record["sha256"] for record in artifact_records}
    checks.extend(
        _approval_checks(approvals, required_approval_roles or set(), artifact_hashes)
    )

    allowed = allow_waiver or set()
    valid_waiver_by_check: dict[str, dict[str, Any]] = {}
    invalid_waivers: list[dict[str, str]] = []
    for waiver in waivers:
        check_id = str(waiver.get("check_id", ""))
        valid, reason = _valid_waiver(waiver, check_id, artifact_hashes)
        if valid and check_id in allowed:
            valid_waiver_by_check[check_id] = waiver
        else:
            if valid and check_id not in allowed:
                reason = "check_id was not explicitly enabled by --allow-waiver"
            invalid_waivers.append(
                {"waiver_id": str(waiver.get("waiver_id")), "check_id": check_id, "reason": reason}
            )

    failed_checks = [check for check in checks if not check["passed"]]
    failed_check_ids = {str(check["id"]) for check in failed_checks}
    effective_waiver_by_check = {
        check_id: waiver
        for check_id, waiver in valid_waiver_by_check.items()
        if check_id in failed_check_ids
    }
    if not failed_checks:
        status = "PASS"
    elif all(
        check["id"] in effective_waiver_by_check
        and check.get("waivable", False)
        and check["id"] in allowed
        for check in failed_checks
    ):
        status = "PASS_WITH_WAIVER"
    else:
        status = "FAIL"

    generated_at = _now()
    gate_dir.mkdir(parents=True, exist_ok=False)
    artifact_manifest = {
        "schema_version": "railway.artifact-manifest.v2",
        "project_id": project.project_id,
        "release_id": release_id,
        "gate_id": gate_id,
        "generated_at": generated_at,
        "artifacts": artifact_records,
    }
    checks_document = {
        "schema_version": "railway.gate-checks.v2",
        "project_id": project.project_id,
        "release_id": release_id,
        "gate_id": gate_id,
        "checks": checks,
    }
    approvals_document = {
        "schema_version": "railway.gate-approvals.v2",
        "approvals": approvals,
    }
    waivers_document = {
        "schema_version": "railway.gate-waivers.v2",
        "allowed_check_ids": sorted(allowed),
        "waivers": waivers,
        "invalid_waivers": invalid_waivers,
    }
    supporting = {
        "artifact_manifest": artifact_manifest,
        "checks": checks_document,
        "approvals": approvals_document,
        "waivers": waivers_document,
    }
    supporting_records: dict[str, dict[str, Any]] = {}
    for name, value in supporting.items():
        path = gate_dir / f"{name.replace('_', '-')}.json"
        write_json(path, value)
        supporting_records[name] = {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    result = {
        "schema_version": "railway.quality-gate.v2",
        "project_id": project.project_id,
        "release_id": release_id,
        "gate_id": gate_id,
        "status": status,
        "generated_at": generated_at,
        "project_root_from_gate": Path(
            os.path.relpath(project.root.resolve(), gate_dir.resolve())
        ).as_posix(),
        "previous_gate": previous_record,
        "input_artifacts": artifact_records,
        "registry_identity": registry_identity,
        "supporting_documents": supporting_records,
        "check_count": len(checks),
        "failed_check_ids": [check["id"] for check in failed_checks],
        "waived_check_ids": sorted(effective_waiver_by_check),
        "required_approval_roles": sorted(required_approval_roles or set()),
    }
    result_path = gate_dir / "gate-result.json"
    write_json(result_path, result)
    return result_path, result


def validate_gate_result(path: str | Path) -> dict[str, Any]:
    result_path = Path(path).resolve()
    result = load_json(result_path)
    errors: list[str] = []
    if result.get("schema_version") != "railway.quality-gate.v2":
        errors.append("schema_version must be railway.quality-gate.v2")
    if result.get("status") not in GATE_STATUSES:
        errors.append("status is not a Quality Gate v2 status")
    gate_dir = result_path.parent
    project_root_value = result.get("project_root_from_gate")
    project_root = (gate_dir / str(project_root_value)).resolve()
    if not project_root_value or not project_root.is_dir():
        errors.append("project_root_from_gate does not resolve to a directory")

    for name, record in result.get("supporting_documents", {}).items():
        source = gate_dir / str(record.get("path", ""))
        if not source.is_file():
            errors.append(f"supporting document is missing: {name}")
            continue
        if source.stat().st_size != record.get("bytes"):
            errors.append(f"supporting document size changed: {name}")
        if sha256_file(source) != record.get("sha256"):
            errors.append(f"supporting document hash changed: {name}")

    for record in result.get("input_artifacts", []):
        source = project_root / str(record.get("path", ""))
        role = record.get("role", "unknown")
        if not source.is_file():
            errors.append(f"artifact is missing: {role}")
            continue
        if source.stat().st_size != record.get("bytes"):
            errors.append(f"artifact size changed: {role}")
        if sha256_file(source) != record.get("sha256"):
            errors.append(f"artifact hash changed: {role}")

    previous = result.get("previous_gate")
    if previous:
        previous_path = project_root / str(previous.get("path", ""))
        if not previous_path.is_file():
            errors.append("previous gate result is missing")
        elif sha256_file(previous_path) != previous.get("sha256"):
            errors.append("previous gate result hash changed")

    checks_record = result.get("supporting_documents", {}).get("checks", {})
    checks_path = gate_dir / str(checks_record.get("path", ""))
    if checks_path.is_file():
        checks_value = load_json(checks_path)
        checks = checks_value.get("checks", [])
        actual_failed = [check.get("id") for check in checks if not check.get("passed")]
        if actual_failed != result.get("failed_check_ids", []):
            errors.append("failed_check_ids does not match checks.json")
        waived = set(result.get("waived_check_ids", []))
        if not actual_failed:
            expected_status = "PASS"
        elif all(
            check.get("id") in waived and check.get("waivable", False)
            for check in checks
            if not check.get("passed")
        ):
            expected_status = "PASS_WITH_WAIVER"
        else:
            expected_status = "FAIL"
        if result.get("status") != expected_status:
            errors.append(
                f"gate status is inconsistent with checks: expected {expected_status}"
            )
        for check in checks:
            source_value = check.get("source")
            source_hash = check.get("source_sha256")
            if not source_value or not source_hash:
                continue
            source = project_root / str(source_value)
            if not source.is_file() or sha256_file(source) != source_hash:
                errors.append(f"check source changed: {check.get('id')}")

    registry_identity = result.get("registry_identity")
    if registry_identity:
        registry_path = project_root / str(registry_identity.get("path", ""))
        if registry_path.is_file():
            registry_value = load_json(registry_path)
            asset_ids = sorted(str(asset.get("id")) for asset in registry_value.get("assets", []))
            asset_set_sha256 = hashlib.sha256(
                "\n".join(asset_ids).encode("utf-8")
            ).hexdigest()
            if asset_set_sha256 != registry_identity.get("asset_set_sha256"):
                errors.append("registry asset_set_sha256 changed")

    return {
        "schema_version": "railway.quality-gate-validation.v2",
        "path": str(result_path),
        "valid": not errors,
        "status": result.get("status"),
        "errors": errors,
    }
