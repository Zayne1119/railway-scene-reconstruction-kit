"""Create an internal, source-only research snapshot from a narrow allowlist.

This does not publish, anonymize a production project, or grant a code license.
Raw point clouds, customer projects, web payloads, repository history, credentials,
and historical internal documents are never selected. The original LICENSE stays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from railway_recon.safety import SENSITIVE_PATTERNS

SCRIPTS = (
    "run_synthetic_track_pilot.py", "run_synthetic_track_study.py",
    "run_synthetic_track_adaptive_study.py", "run_synthetic_track_adaptive_stress.py",
    "run_synthetic_track_guarded_study.py", "run_public_rail_pilot.py",
    "run_public_rail_spacing_study.py", "analyze_synthetic_track_formal.py",
    "prepare_paper_reproduction.py",
)
PROTOCOLS = (
    "synthetic-track-study-v1.json", "synthetic-track-adaptive-v2.json",
    "synthetic-track-guarded-v3.json", "public-rail-pilot-v1.json",
    "public-rail-spacing-v2.json", "synthetic-track-formal-analysis-v2.json",
)


def selected_files(root: Path) -> list[Path]:
    """Select source/dependencies and research-only entry points, never run data."""
    paths = {root / name for name in ("LICENSE", "pyproject.toml", "uv.lock")}
    paths.update(path for path in (root / "src" / "railway_recon").rglob("*")
                 if path.suffix in (".py", ".json") and "__pycache__" not in path.parts)
    paths.update(root / "scripts" / name for name in SCRIPTS)
    paths.update(root / "benchmarks" / "protocols" / name for name in PROTOCOLS)
    paths.update((root / "tests").glob("test_synthetic_track*.py"))
    paths.update((root / "tests").glob("test_public_rail*.py"))
    paths.update(root / "tests" / name for name in ("__init__.py", "test_public_las_intake.py",
                 "test_paper_reproduction.py", "test_synthetic_track_formal_analysis.py"))
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required research source absent: {path.name}")
        relative = path.relative_to(root)
        if path.resolve() != root.resolve() / relative:
            raise ValueError("Research snapshot refuses symlinks and redirected paths")
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def inspect_text(payload: bytes, relative: str) -> None:
    if len(payload) > 5 * 1024 * 1024:
        raise ValueError(f"Source exceeds text bound: {relative}")
    text = payload.decode("utf-8-sig")
    for kind, pattern in SENSITIVE_PATTERNS.items():
        if pattern.search(text):
            raise ValueError(f"Potential disclosure ({kind}): {relative}")


def prepare_snapshot(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    if not output.is_relative_to(root / "benchmarks" / "local" / "paper-reproduction"):
        raise ValueError("Snapshot output must be a new child of benchmarks/local/paper-reproduction")
    if output == root / "benchmarks" / "local" / "paper-reproduction" or output.exists():
        raise ValueError("Refusing existing output or snapshot container itself")
    selected = selected_files(root)
    contents = []
    for path in selected:
        relative = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        inspect_text(raw, relative)
        contents.append((path, relative, raw, hashlib.sha256(raw).hexdigest()))
    # Validate all selected source before creating the destination.
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for _, relative, raw, digest in contents:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(raw)
        records.append({"path": relative, "bytes": len(raw), "sha256": digest})
    readme = (
        b"# Internal research reproduction snapshot\n\n"
        b"This is a source-only snapshot, not a public release. The original LICENSE applies.\n"
        b"It includes package code because the existing study freeze binds the entire package; "
        b"it excludes customer projects, web payloads, raw data, run outputs, credentials, "
        b"Git history and historical internal documents.\n\n"
        b"Run the research tests, create a new local freeze, then replay the registered study "
        b"using scripts/run_synthetic_track_guarded_study.py. A replay is not a new held-out test. "
        b"Use separately pinned public data for the optional public experiment.\n\n"
        b"Automatic checks identify selected known sensitive patterns; they cannot establish "
        b"legal clearance or guarantee absence of unknown identifiers. Publication and a code "
        b"license require the owner's decision.\n"
    )
    with (output / "README.md").open("xb") as stream:
        stream.write(readme)
    records.append({"path": "README.md", "bytes": len(readme),
                    "sha256": hashlib.sha256(readme).hexdigest()})
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != digest
           for path, _, _, digest in contents):
        raise ValueError("Source changed during snapshot creation; no completed manifest written")
    report = {"schema_version": "railway.paper-source-snapshot.v1", "status": "completed",
              "created_at_utc": datetime.now(UTC).isoformat(), "role": "internal_reproduction_only",
              "published": False, "customer_data_selected": False,
              "raw_public_data_selected": False, "code_license": "original_all_rights_reserved",
              "automatic_disclosure_check": "known_patterns_and_source_allowlist_only",
              "fresh_dependency_environment_verified": False,
              "files": records, "file_count": len(records)}
    with (output / "snapshot_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_snapshot(ROOT, args.output)
    print(json.dumps({key: result[key] for key in ("status", "file_count", "role")}))


if __name__ == "__main__":
    main()
