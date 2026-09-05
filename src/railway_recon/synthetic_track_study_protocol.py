"""Grouped synthetic study declarations and source-bound test authorization.

Creating or validating a protocol never generates geometry, observations or
truth. A freeze is a local content-integrity record, not external registration
or proof that the reserved seeds have never been inspected.
"""

from __future__ import annotations

import copy
import random
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .io import load_json, sha256_json
from .research_protocol import (
    BASELINE_SUFFIXES,
    freeze_research_baseline,
    verify_research_baseline,
)
from .synthetic_track_study_audit import STUDY_MODES, StudyAuditPolicy
from .synthetic_track_study_scene import STUDY_VARIANTS

PROTOCOL_SCHEMA = "railway.synthetic-track-study-protocol.v1"
STUDY_SPLITS = ("development", "validation", "test")
DEFAULT_COUNTS = {"development": 40, "validation": 20, "test": 60}
EVALUATION_POLICY = {
    "primary_endpoint": "diagnosis_recall_including_abstentions",
    "abstention_rule": "unassigned_causes_are_missed_diagnoses_not_correct_gaps",
    "detection_comparison": "identical_shared_geometry_alarms_not_an_improvement_claim",
    "analysis_unit": "base_layout_group",
    "condition_pairing": "all_conditions_stay_with_their_base_layout",
    "test_requires_verified_freeze": True,
    "partial_test_allowed": False,
}
REQUIRED_EXECUTION_FILES = (
    "pyproject.toml",
    "uv.lock",
    "src/railway_recon/__init__.py",
    "src/railway_recon/synthetic_track_scene.py",
    "src/railway_recon/synthetic_track_audit.py",
    "src/railway_recon/synthetic_track_pilot.py",
    "src/railway_recon/relationships.py",
    "src/railway_recon/synthetic_track_study_scene.py",
    "src/railway_recon/synthetic_track_study_audit.py",
    "src/railway_recon/synthetic_track_study_protocol.py",
    "src/railway_recon/synthetic_track_study.py",
    "src/railway_recon/research_protocol.py",
    "src/railway_recon/io.py",
    "src/railway_recon/benchmark.py",
    "scripts/run_synthetic_track_study.py",
)
_SKIPPED_DIRECTORIES = {".git", ".pytest_cache", ".ruff_cache", "__pycache__"}


def _integer(value: Any, name: str, *, minimum: int = 0, maximum: int = 2**63 - 1) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict) or not value or set(value) - set(STUDY_SPLITS):
        raise ValueError("counts must be a nonempty mapping of valid study splits")
    return {
        split: _integer(value[split], f"counts.{split}", minimum=1)
        for split in STUDY_SPLITS if split in value
    }


def create_study_protocol(seed: int = 20260906, counts: dict | None = None) -> dict[str, Any]:
    """Declare independent seeded layout groups without constructing any cases."""
    _integer(seed, "seed")
    selected_counts = _counts(DEFAULT_COUNTS if counts is None else counts)
    groups: list[dict[str, Any]] = []
    used_seeds: set[int] = set()
    for split, count in selected_counts.items():
        # A development count change does not silently reassign the test seeds.
        rng = random.Random(int(sha256_json([PROTOCOL_SCHEMA, seed, split]), 16))
        for index in range(count):
            group_seed = rng.getrandbits(63)
            while group_seed in used_seeds:
                group_seed = rng.getrandbits(63)
            used_seeds.add(group_seed)
            groups.append({
                "group_id": sha256_json([PROTOCOL_SCHEMA, "group", seed, split, index])[:32],
                "split": split,
                "seed": group_seed,
                "family_index": index % 6,
            })
    value = {
        "schema_version": PROTOCOL_SCHEMA,
        "seed": seed,
        "counts": selected_counts,
        "modes": list(STUDY_MODES),
        "policy": asdict(StudyAuditPolicy()),
        "conditions": list(STUDY_VARIANTS),
        "groups": groups,
        "evaluation": copy.deepcopy(EVALUATION_POLICY),
    }
    value["protocol_id"] = "p2-" + sha256_json(value)[:24]
    validate_study_protocol(value)
    return value


def validate_study_protocol(value: Any) -> None:
    """Validate the entire allocation and fixed comparison contract, without generation."""
    if not isinstance(value, dict) or value.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("Unsupported study protocol schema_version")
    identity = value.get("protocol_id")
    if not isinstance(identity, str) or not identity.strip() or identity != identity.strip():
        raise ValueError("protocol_id must be a nonempty string without surrounding whitespace")
    _integer(value.get("seed"), "seed")
    counts = _counts(value.get("counts"))
    if value.get("modes") != list(STUDY_MODES):
        raise ValueError("modes must exactly match the three study ablations in order")
    if value.get("conditions") != list(STUDY_VARIANTS):
        raise ValueError("conditions must exactly match STUDY_VARIANTS in order")
    if value.get("evaluation") != EVALUATION_POLICY:
        raise ValueError("evaluation must retain grouped, abstention-aware shared-alarm semantics")
    policy = value.get("policy")
    if not isinstance(policy, dict) or set(policy) != set(asdict(StudyAuditPolicy())):
        raise ValueError("policy must explicitly contain every StudyAuditPolicy field")
    if any(type(item) not in (int, float) for item in policy.values()):
        raise ValueError("policy values must be numeric, not booleans or strings")
    try:
        StudyAuditPolicy(**policy)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid study policy: {error}") from error
    groups = value.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("groups must be a nonempty list")
    identities: set[str] = set()
    seeds: set[int] = set()
    actual_counts: Counter[str] = Counter()
    for index, group in enumerate(groups):
        if not isinstance(group, dict) or set(group) != {
            "group_id", "split", "seed", "family_index"
        }:
            raise ValueError(f"groups[{index}] must contain group_id, split, seed and family_index")
        identity = group["group_id"]
        if not isinstance(identity, str) or not identity.strip() or identity in identities:
            raise ValueError("group_id values must be nonempty and unique across all splits")
        identities.add(identity)
        split = group["split"]
        if not isinstance(split, str) or split not in counts:
            raise ValueError(f"Invalid or undeclared group split at groups[{index}]")
        group_seed = _integer(group["seed"], f"groups[{index}].seed")
        if group_seed in seeds:
            raise ValueError("Group seeds must be unique across all splits")
        seeds.add(group_seed)
        _integer(group["family_index"], f"groups[{index}].family_index", maximum=5)
        actual_counts[split] += 1
    if dict(actual_counts) != counts:
        raise ValueError("groups must exactly match the declared split counts")


def _repository_root(protocol_path: Path) -> Path:
    for candidate in protocol_path.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ValueError("Protocol must be inside a repository containing pyproject.toml")


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"Study freeze path escapes repository: {path}") from error


def _freeze_inputs(root: Path, protocol: Path) -> tuple[list[str], set[str]]:
    protocol_relative = _relative(protocol, root)
    if protocol.suffix.lower() != ".json" or not protocol.is_file():
        raise ValueError("Study protocol must be an existing .json file")
    required = set(REQUIRED_EXECUTION_FILES) | {protocol_relative}
    for relative in required:
        path = root / relative
        _relative(path, root)
        if not path.is_file():
            raise FileNotFoundError(f"Missing required study execution file: {relative}")
    # Freeze the complete package, including transitive imports and resource
    # schemas. Never select paper drafts, customer inputs or generated outputs.
    includes = [
        "src/railway_recon", "scripts/run_synthetic_track_study.py",
        "pyproject.toml", "uv.lock", protocol_relative,
    ]
    for path in (root / "src" / "railway_recon").rglob("*"):
        if any(part in _SKIPPED_DIRECTORIES for part in path.relative_to(root).parts):
            continue
        if path.is_file() and path.suffix.lower() in BASELINE_SUFFIXES:
            required.add(_relative(path, root))
    return includes, required


def freeze_study_protocol(protocol_path: str | Path, output_path: str | Path) -> Path:
    """Bind the validated protocol, all package sources and the real study CLI."""
    protocol = Path(protocol_path).resolve()
    value = load_json(protocol)
    validate_study_protocol(value)
    root = _repository_root(protocol)
    includes, _ = _freeze_inputs(root, protocol)
    output = Path(output_path).resolve()
    _relative(output, root)
    if output == protocol or output.is_relative_to(root / "src"):
        raise ValueError("Freeze output must not replace the protocol or be inside source code")
    manifest = freeze_research_baseline(
        root, output, freeze_id=value["protocol_id"], includes=includes
    )
    verification = verify_study_freeze(protocol, manifest)
    if verification["status"] != "pass":
        raise ValueError("Study freeze verification failed: " + "; ".join(verification["issues"]))
    return manifest


def verify_study_freeze(protocol_path: str | Path, freeze_path: str | Path) -> dict[str, Any]:
    """Reject altered or incomplete bindings, even a self-consistent empty manifest.

    This validates local content only. The runner separately enforces full test
    execution and preserves first-run outputs; this function never runs a split.
    """
    protocol, source = Path(protocol_path).resolve(), Path(freeze_path).resolve()
    issues: list[str] = []
    checked_files = 0
    try:
        value = load_json(protocol)
        validate_study_protocol(value)
        root = _repository_root(protocol)
        _relative(source, root)
        _, required = _freeze_inputs(root, protocol)
        manifest = load_json(source)
        if manifest.get("status") != "completed":
            issues.append("Freeze status must be completed")
        if manifest.get("freeze_id") != value["protocol_id"]:
            issues.append("Freeze ID does not bind this protocol_id")
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
            issues.append(f"Out-of-scope study freeze binding: {relative}")
        # The existing verifier checks the manifest checksum plus every bound
        # file's size and SHA-256. Required-set checks above close its empty or
        # partial manifest loophole without changing the P1 freeze mechanism.
        baseline = verify_research_baseline(root, source)
        checked_files = baseline["checked_files"]
        issues.extend(baseline["issues"])
    except (OSError, ValueError, TypeError, KeyError) as error:
        issues.append(str(error))
    return {
        "schema_version": "railway.synthetic-track-study-freeze-verification.v1",
        "protocol": str(protocol),
        "manifest": str(source),
        "status": "pass" if not issues else "fail",
        "checked_files": checked_files,
        "issues": issues,
        "limitations": [
            "Local content verification is not external preregistration or an access log.",
            "This check does not run any development, validation or test cases.",
        ],
    }
