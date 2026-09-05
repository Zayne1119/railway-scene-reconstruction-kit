"""Versioned protocol for adaptive diagnosis after the observed P2 v1 pilot.

The embedded v1 allocation is preserved exactly. No cases are generated here,
and no v1 freeze is rewritten or accepted as authorization for the new method.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from .io import load_json, sha256_json
from .research_protocol import freeze_research_baseline, verify_research_baseline
from .synthetic_track_study_protocol import (
    REQUIRED_EXECUTION_FILES as BASE_REQUIRED_EXECUTION_FILES,
)
from .synthetic_track_study_protocol import (
    _freeze_inputs as _base_freeze_inputs,
)
from .synthetic_track_study_protocol import (
    _integer,
    _relative,
    _repository_root,
    create_study_protocol,
    validate_study_protocol,
)

PROTOCOL_SCHEMA = "railway.synthetic-track-adaptive-protocol.v1"
ADAPTIVE_MODES = ("local_geometry", "topology_only", "topology_evidence", "adaptive_evidence")
ADAPTATION_DISCLOSURE = {
    "v1_development_results_observed": True,
    "v1_validation_results_observed": True,
    "revision_role": "method_development_after_v1_development_and_validation",
    "confidence_semantics": "qualitative_not_calibrated_probability",
    "fallback_semantics": "topology_inference_not_observation_verified",
    "numeric_thresholds": "unchanged_from_embedded_base_protocol",
    "group_allocation": "unchanged_from_embedded_base_protocol",
    "test_result_use_for_this_revision": "none",
    "freeze_semantics": "new_source_binding_required_not_external_preregistration",
}
ADAPTIVE_EXECUTION_FILES = (
    "src/railway_recon/synthetic_track_adaptive_audit.py",
    "src/railway_recon/synthetic_track_adaptive_protocol.py",
    "src/railway_recon/synthetic_track_adaptive_study.py",
    "scripts/run_synthetic_track_adaptive_study.py",
)
REQUIRED_EXECUTION_FILES = (*BASE_REQUIRED_EXECUTION_FILES, *ADAPTIVE_EXECUTION_FILES)


def _protocol_id(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "protocol_id"}
    return "p2-adaptive-" + sha256_json(payload)[:24]


def create_adaptive_protocol(base_protocol: dict | None = None) -> dict[str, Any]:
    """Embed an unchanged v1 protocol and declare the fourth, adaptive mode."""
    base = create_study_protocol() if base_protocol is None else copy.deepcopy(base_protocol)
    validate_study_protocol(base)
    value = {
        "schema_version": PROTOCOL_SCHEMA,
        "base_protocol": base,
        "base_protocol_sha256": sha256_json(base),
        "modes": list(ADAPTIVE_MODES),
        "policy": copy.deepcopy(base["policy"]),
        "evaluation": copy.deepcopy(base["evaluation"]),
        "adaptation_disclosure": copy.deepcopy(ADAPTATION_DISCLOSURE),
    }
    value["protocol_id"] = _protocol_id(value)
    validate_adaptive_protocol(value)
    return value


def validate_adaptive_protocol(value: Any) -> None:
    """Validate v1 allocations and the transparent, threshold-preserving revision."""
    if not isinstance(value, dict) or value.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("Unsupported adaptive protocol schema_version")
    base = value.get("base_protocol")
    validate_study_protocol(base)
    if value.get("base_protocol_sha256") != sha256_json(base):
        raise ValueError("Embedded base_protocol SHA-256 mismatch")
    if value.get("modes") != list(ADAPTIVE_MODES):
        raise ValueError("modes must exactly match the four adaptive study ablations")
    if sha256_json(value.get("policy")) != sha256_json(base["policy"]):
        raise ValueError("Adaptive policy must exactly preserve the base protocol thresholds")
    if sha256_json(value.get("evaluation")) != sha256_json(base["evaluation"]):
        raise ValueError("Adaptive evaluation must preserve the abstention-aware diagnosis endpoint")
    if sha256_json(value.get("adaptation_disclosure")) != sha256_json(ADAPTATION_DISCLOSURE):
        raise ValueError("Adaptive protocol must retain the development and confidence disclosures")
    if value.get("protocol_id") != _protocol_id(value):
        raise ValueError("Adaptive protocol_id does not match its complete payload")


def _freeze_inputs(root: Path, protocol: Path) -> tuple[list[str], set[str]]:
    includes, required = _base_freeze_inputs(root, protocol)
    for relative in ADAPTIVE_EXECUTION_FILES:
        path = root / relative
        _relative(path, root)
        if not path.is_file():
            raise FileNotFoundError(f"Missing required adaptive execution file: {relative}")
        required.add(relative)
    includes.append("scripts/run_synthetic_track_adaptive_study.py")
    return includes, required


def freeze_adaptive_protocol(protocol_path: str | Path, output_path: str | Path) -> Path:
    """Create a new binding of the v2 method, unchanged dependencies and protocol."""
    protocol = Path(protocol_path).resolve()
    value = load_json(protocol)
    validate_adaptive_protocol(value)
    root = _repository_root(protocol)
    includes, _ = _freeze_inputs(root, protocol)
    output = Path(output_path).resolve()
    _relative(output, root)
    if output == protocol or output.is_relative_to(root / "src"):
        raise ValueError("Freeze output must not replace the protocol or be inside source code")
    manifest = freeze_research_baseline(
        root, output, freeze_id=value["protocol_id"], includes=includes
    )
    verification = verify_adaptive_freeze(protocol, manifest)
    if verification["status"] != "pass":
        raise ValueError("Adaptive freeze verification failed: " + "; ".join(verification["issues"]))
    return manifest


def verify_adaptive_freeze(protocol_path: str | Path, freeze_path: str | Path) -> dict[str, Any]:
    """Check complete v2 bindings; a valid v1 content manifest is insufficient.

    The runner must additionally match this repository to its executing package
    root, refuse partial test execution, and preserve first-run results.
    """
    protocol, source = Path(protocol_path).resolve(), Path(freeze_path).resolve()
    issues: list[str] = []
    checked_files = 0
    root: Path | None = None
    try:
        value = load_json(protocol)
        validate_adaptive_protocol(value)
        root = _repository_root(protocol)
        _relative(source, root)
        _, required = _freeze_inputs(root, protocol)
        manifest = load_json(source)
        if manifest.get("status") != "completed":
            issues.append("Freeze status must be completed")
        if manifest.get("freeze_id") != value["protocol_id"]:
            issues.append("Freeze ID does not bind this adaptive protocol_id")
        records = manifest.get("files")
        if not isinstance(records, list) or not records:
            raise ValueError("Freeze files must be a nonempty list")
        if type(manifest.get("file_count")) is not int or manifest["file_count"] != len(records):
            issues.append("Freeze file_count does not match its bindings")
        bound: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                raise TypeError("Malformed freeze file binding")
            relative = record.get("path")
            if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
                raise ValueError("Freeze binding path must be relative and nonempty")
            canonical = _relative(root / relative, root)
            if canonical != relative or ".." in Path(relative).parts:
                raise ValueError("Freeze binding paths must be canonical repository-relative paths")
            if canonical in bound:
                raise ValueError(f"Duplicate freeze binding: {canonical}")
            bound.add(canonical)
            _integer(record.get("bytes"), f"Frozen byte size for {canonical}")
            digest = record.get("sha256")
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError(f"Invalid SHA-256 binding: {canonical}")
        for relative in sorted(required - bound):
            issues.append(f"Missing required hash binding: {relative}")
        for relative in sorted(bound - required):
            issues.append(f"Out-of-scope adaptive freeze binding: {relative}")
        baseline = verify_research_baseline(root, source)
        checked_files = baseline["checked_files"]
        issues.extend(baseline["issues"])
    except (OSError, ValueError, TypeError, KeyError) as error:
        issues.append(str(error))
    return {
        "schema_version": "railway.synthetic-track-adaptive-freeze-verification.v1",
        "protocol": str(protocol),
        "manifest": str(source),
        "repository_root": str(root) if root is not None else None,
        "status": "pass" if not issues else "fail",
        "checked_files": checked_files,
        "issues": issues,
        "limitations": [
            "This revision follows observed v1 development and validation results.",
            "Local hashes are not external preregistration or proof of unobserved test data.",
            "A v1 freeze may retain content integrity but does not authorize the adaptive method.",
            "Verification does not generate or run any split.",
        ],
    }
