"""Transparent P2 v3 revision and complete, local source bindings.

The original v1 allocation is embedded unchanged. Protocol operations declare
seeds only: they never construct candidate geometry, observations or truth.
Prior development, validation and legacy stress results informed this revision;
a local freeze neither makes those results blind nor proves test non-access.
"""

from __future__ import annotations

import copy
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .io import load_json, sha256_json
from .research_protocol import freeze_research_baseline, verify_research_baseline
from .synthetic_track_adaptive_protocol import (
    ADAPTIVE_MODES,
)
from .synthetic_track_adaptive_protocol import (
    REQUIRED_EXECUTION_FILES as ADAPTIVE_REQUIRED_EXECUTION_FILES,
)
from .synthetic_track_adaptive_protocol import (
    _freeze_inputs as _adaptive_freeze_inputs,
)
from .synthetic_track_study_audit import StudyAuditPolicy
from .synthetic_track_study_protocol import (
    _integer,
    _relative,
    _repository_root,
    create_study_protocol,
    validate_study_protocol,
)

PROTOCOL_SCHEMA = "railway.synthetic-track-guarded-protocol.v1"
GUARDED_MODES = (*ADAPTIVE_MODES, "direction_geometry", "guarded_evidence")
ADAPTATION_DISCLOSURE = {
    "v1_development_results_observed": True,
    "v1_validation_results_observed": True,
    "v2_development_results_observed": True,
    "v2_validation_results_observed": True,
    "legacy_stress_results_observed": True,
    "legacy_stress_case_count": 18,
    "revision_role": "method_development_after_v1_v2_and_legacy_stress",
    "development_replay_is_blind_test": False,
    "confidence_semantics": "qualitative_not_calibrated_probability",
    "fallback_semantics": "direction_geometry_inference_not_observation_verified",
    "evidence_verification_semantics": "internal_support_checks_not_truth_or_authenticity",
    "numeric_thresholds": "unchanged_original_StudyAuditPolicy_defaults",
    "group_allocation": "unchanged_from_embedded_base_protocol",
    "test_result_use_for_this_revision": "none",
    "production_defaults_modified": False,
    "freeze_semantics": "new_source_binding_required_not_external_preregistration",
    "base_provenance": "embedded_content_hash_not_external_lineage_attestation",
}
METHOD_CONTRACT = {
    "direction_measurement": "current_chord_against_source_end_and_target_start_tangents",
    "direction_threshold": "minimum_direction_cosine_from_embedded_policy",
    "direction_geometry": "prefer_smooth_current_gap_over_nearer_aligned_alternative",
    "guard_condition": (
        "current_path_supported_but_endpoint_direction_inconsistent_with_"
        "nearby_aligned_alternative"
    ),
    "guard_action": "fall_back_to_direction_geometry_with_unverified_inference_label",
    "guard_scope": "specific_direction_inconsistent_bridge_not_all_observation_contamination",
    "geometry_alarms": "identical_shared_v1_geometry_alarms_not_detection_improvement",
}
GUARDED_EXECUTION_FILES = (
    "src/railway_recon/synthetic_track_guarded_audit.py",
    "src/railway_recon/synthetic_track_guarded_protocol.py",
    "src/railway_recon/synthetic_track_guarded_stress.py",
    "src/railway_recon/synthetic_track_guarded_study.py",
    "scripts/run_synthetic_track_guarded_study.py",
)
GUARDED_DEPENDENCY_FILES = ("src/railway_recon/synthetic_track_adaptive_stress.py",)
REQUIRED_EXECUTION_FILES = (
    *ADAPTIVE_REQUIRED_EXECUTION_FILES, *GUARDED_EXECUTION_FILES, *GUARDED_DEPENDENCY_FILES,
)
_PROTOCOL_FIELDS = {
    "schema_version", "method_revision", "base_protocol", "base_protocol_sha256",
    "modes", "policy", "evaluation", "adaptation_disclosure", "method_contract", "protocol_id",
}


def _protocol_id(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "protocol_id"}
    return "p2-guarded-v3-" + sha256_json(payload)[:24]


def create_guarded_protocol(base_protocol: dict | None = None) -> dict[str, Any]:
    """Declare six diagnosis modes without changing or generating the v1 allocation."""
    base = create_study_protocol() if base_protocol is None else copy.deepcopy(base_protocol)
    validate_study_protocol(base)
    value = {
        "schema_version": PROTOCOL_SCHEMA,
        "method_revision": "v3",
        "base_protocol": base,
        "base_protocol_sha256": sha256_json(base),
        "modes": list(GUARDED_MODES),
        "policy": copy.deepcopy(base["policy"]),
        "evaluation": copy.deepcopy(base["evaluation"]),
        "adaptation_disclosure": copy.deepcopy(ADAPTATION_DISCLOSURE),
        "method_contract": copy.deepcopy(METHOD_CONTRACT),
    }
    value["protocol_id"] = _protocol_id(value)
    validate_guarded_protocol(value)
    return value


def validate_guarded_protocol(value: Any) -> None:
    """Reject altered allocations, thresholds, revision disclosures or method claims."""
    if not isinstance(value, dict) or value.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("Unsupported guarded protocol schema_version")
    if set(value) != _PROTOCOL_FIELDS:
        raise ValueError("Guarded protocol must contain exactly its declared fields")
    if value["method_revision"] != "v3":
        raise ValueError("Guarded protocol method_revision must be v3")
    base = value["base_protocol"]
    validate_study_protocol(base)
    if value["base_protocol_sha256"] != sha256_json(base):
        raise ValueError("Embedded base_protocol SHA-256 mismatch")
    if value["modes"] != list(GUARDED_MODES):
        raise ValueError("modes must exactly match the six guarded study ablations")
    original_policy_hash = sha256_json(asdict(StudyAuditPolicy()))
    if sha256_json(base["policy"]) != original_policy_hash:
        raise ValueError("Guarded revision must preserve original StudyAuditPolicy thresholds")
    if sha256_json(value["policy"]) != sha256_json(base["policy"]):
        raise ValueError("Guarded policy must exactly preserve the embedded base thresholds")
    if sha256_json(value["evaluation"]) != sha256_json(base["evaluation"]):
        raise ValueError("Guarded evaluation must preserve the abstention-aware diagnosis endpoint")
    if sha256_json(value["adaptation_disclosure"]) != sha256_json(ADAPTATION_DISCLOSURE):
        raise ValueError("Guarded protocol must retain observed-data and confidence disclosures")
    if sha256_json(value["method_contract"]) != sha256_json(METHOD_CONTRACT):
        raise ValueError("Guarded protocol must retain its limited direction-guard contract")
    if value["protocol_id"] != _protocol_id(value):
        raise ValueError("Guarded protocol_id does not match its complete payload")


def _freeze_inputs(root: Path, protocol: Path) -> tuple[list[str], set[str]]:
    includes, required = _adaptive_freeze_inputs(root, protocol)
    for relative in (*GUARDED_EXECUTION_FILES, *GUARDED_DEPENDENCY_FILES):
        path = root / relative
        _relative(path, root)
        if not path.is_file():
            raise FileNotFoundError(f"Missing required guarded execution file: {relative}")
        required.add(relative)
    includes.append("scripts/run_synthetic_track_guarded_study.py")
    return includes, required


def freeze_guarded_protocol(protocol_path: str | Path, output_path: str | Path) -> Path:
    """Bind v3 execution, all package dependencies/resources and embedded v1 content."""
    protocol = Path(protocol_path).resolve()
    value = load_json(protocol)
    validate_guarded_protocol(value)
    root = _repository_root(protocol)
    includes, _ = _freeze_inputs(root, protocol)
    output = Path(output_path).resolve()
    _relative(output, root)
    if output == protocol or output.is_relative_to(root / "src"):
        raise ValueError("Freeze output must not replace the protocol or be inside source code")
    manifest = freeze_research_baseline(
        root, output, freeze_id=value["protocol_id"], includes=includes
    )
    verification = verify_guarded_freeze(protocol, manifest)
    if verification["status"] != "pass":
        raise ValueError("Guarded freeze verification failed: " + "; ".join(verification["issues"]))
    return manifest


def verify_guarded_freeze(protocol_path: str | Path, freeze_path: str | Path) -> dict[str, Any]:
    """Check complete v3 bindings; prior version freezes cannot authorize this method.

    The runner must separately check its executing package root, reject partial
    test runs and preserve first-run results. This operation executes no cases.
    """
    protocol, source = Path(protocol_path).resolve(), Path(freeze_path).resolve()
    issues: list[str] = []
    checked_files = 0
    root: Path | None = None
    try:
        value = load_json(protocol)
        validate_guarded_protocol(value)
        root = _repository_root(protocol)
        _relative(source, root)
        _, required = _freeze_inputs(root, protocol)
        manifest = load_json(source)
        if manifest.get("status") != "completed":
            issues.append("Freeze status must be completed")
        if manifest.get("freeze_id") != value["protocol_id"]:
            issues.append("Freeze ID does not bind this guarded protocol_id")
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
            issues.append(f"Out-of-scope guarded freeze binding: {relative}")
        baseline = verify_research_baseline(root, source)
        checked_files = baseline["checked_files"]
        issues.extend(baseline["issues"])
    except (OSError, ValueError, TypeError, KeyError) as error:
        issues.append(str(error))
    return {
        "schema_version": "railway.synthetic-track-guarded-freeze-verification.v1",
        "protocol": str(protocol),
        "manifest": str(source),
        "repository_root": str(root) if root is not None else None,
        "status": "pass" if not issues else "fail",
        "checked_files": checked_files,
        "issues": issues,
        "limitations": [
            "This revision follows observed v1/v2 development, validation and 18 stress cases.",
            "The directional guard does not detect all forms of contaminated observations.",
            "Local hashes are not external preregistration or proof of unobserved test data.",
            "A v1/v2 freeze may retain content integrity but cannot authorize the guarded method.",
            "The embedded v1 content hash does not establish external allocation provenance.",
            "Verification does not generate or run any split.",
        ],
    }
