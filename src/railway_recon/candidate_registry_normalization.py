from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .io import load_json, write_json
from .registry import summarize_registry, validate_registry_value

VALID_STATUSES = {"candidate", "reviewed", "accepted", "rejected", "superseded"}
VALID_EVIDENCE = {"observed", "photo_interpreted", "rule_inferred", "unsupported"}


def normalize_candidate_registry_value(registry: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    value = copy.deepcopy(registry)
    status_changes = []
    evidence_changes = []
    for asset in value.get("assets", []):
        status = str(asset.get("status", "candidate"))
        if status not in VALID_STATUSES:
            asset.setdefault("parameters", {})["legacy_status_before_normalization"] = status
            asset["status"] = "candidate"
            status_changes.append({"asset_id": asset.get("id"), "from": status, "to": "candidate"})
        evidence = str(asset.get("evidence_level", "unsupported"))
        if evidence not in VALID_EVIDENCE:
            lowered = evidence.lower()
            normalized = (
                "photo_interpreted"
                if "photo" in lowered
                else "observed"
                if "point" in lowered
                else "rule_inferred"
            )
            asset.setdefault("parameters", {})[
                "legacy_evidence_level_before_normalization"
            ] = evidence
            asset["evidence_level"] = normalized
            evidence_changes.append(
                {"asset_id": asset.get("id"), "from": evidence, "to": normalized}
            )
    known = {str(asset.get("id")) for asset in value.get("assets", [])}
    kept_relations = []
    dropped_relations = []
    for relation in value.get("relations", []):
        if relation.get("from") in known and relation.get("to") in known:
            kept_relations.append(relation)
        else:
            dropped_relations.append(copy.deepcopy(relation))
    value["relations"] = kept_relations
    value["summary"] = summarize_registry(value)
    errors = validate_registry_value(value)
    report = {
        "schema_version": "railway.candidate-registry-normalization.v1",
        "status_change_count": len(status_changes),
        "evidence_change_count": len(evidence_changes),
        "dropped_dangling_relation_count": len(dropped_relations),
        "status_changes": status_changes,
        "evidence_changes": evidence_changes,
        "dropped_dangling_relations": dropped_relations,
        "validation_errors": errors,
        "passed": not errors,
        "status": "pass" if not errors else "normalization_incomplete",
        "policy": (
            "Normalize only a candidate registry copy; preserve legacy values in asset parameters "
            "and preserve dropped dangling relation records in this report."
        ),
    }
    return value, report


def normalize_candidate_registry(
    source_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    source = Path(source_path).resolve()
    output = Path(output_path).resolve()
    report_output = Path(report_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    for path in (output, report_output):
        if path.exists():
            raise FileExistsError(path)
    normalized, report = normalize_candidate_registry_value(load_json(source))
    if not report["passed"]:
        raise ValueError("Candidate registry normalization did not produce a valid registry")
    write_json(output, normalized)
    write_json(
        report_output,
        {
            **report,
            "source_registry": str(source),
            "normalized_registry": str(output),
        },
    )
    return report
