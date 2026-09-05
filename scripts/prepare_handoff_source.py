"""Copy allowlisted source/configuration text for an internal newcomer handoff.

This creates a new directory, not an archive or public release. It never copies
environments, Git history, raw data, generated models, local viewer project
configuration or customer workspaces. Pattern checks do not prove anonymity,
legal clearance, or the absence of previously unknown identifiers in source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from railway_recon.safety import SENSITIVE_PATTERNS

MANIFEST_NAME = "handoff_source_manifest.json"
MAXIMUM_FILE_BYTES = 8 * 1024 * 1024
MAXIMUM_TOTAL_BYTES = 64 * 1024 * 1024
EXCLUDED_PARTS = frozenset({
    ".git", ".venv", ".runtime", "node_modules", "__pycache__", ".pytest_cache",
    ".ruff_cache", ".env", "private", "customer", "customer_data", "data", "datasets",
    "fixtures", "models", "generated", "dist", "local", "runs",
})
ROOT_FILES = ("README.md", "CHANGELOG.md", "LICENSE", "pyproject.toml", "uv.lock", "Railway.ps1", ".gitignore")
DOC_FILES = (
    "NEWCOMER_HANDOFF_CN.md", "AI_HANDOFF_CN.md", "QUICKSTART_CN.md", "NEW_PROJECT_CHECKLIST_CN.md",
    "HANDOFF_VERIFICATION_CN.md",
)
WEB_ROOT_FILES = (
    "package.json", "package-lock.json", "index.html", "vite.config.js", "server-config.js", "README.md", ".gitignore",
)
# Inspected text-only placeholders: paths refer to /models or /data, not a
# particular machine. Neither actual models nor those data directories follow.
WEB_PUBLIC_FILES = ("project.example.json", "project.track-boundary.example.json")
EXAMPLE_FILES = (
    "README.md", "camera.example.csv", "asset_registry.example.json",
    "manual_review.example.csv", "reviewed_scene.example.json",
)
PROTOCOL_FILES = (
    "synthetic-track-study-v1.json", "synthetic-track-adaptive-v2.json", "synthetic-track-guarded-v3.json",
    "public-rail-pilot-v1.json", "public-rail-spacing-v2.json",
)
SAFE_TEST_FIXTURE_FILES = (
    "README.md", "canopy_interfaces_fail.json", "mesh_requires_optimization.json",
    "release_local_pass.json", "segment_conditional.json", "track_full_length_fail.json",
)
CONFIG_TEMPLATE_FILES = (
    "camera.example.csv", "canopy_photo_evidence.default.json", "canopy_structure.default.json",
    "corridor_column_grid.default.json", "linear_detection.default.json", "mesh_seam_reconciliation.example.json",
    "opposite_platform_reconstruction.default.json", "platform_interface_audit.default.json",
    "platform_mesh.default.json", "platform_photo_evidence.default.json", "platform_surface.default.json",
    "project.example.json", "projection_calibration.default.json", "rail_detection.anchored_vertical.example.json",
    "rail_detection.default.json", "rules.default.json", "scene_relationships.default.json",
    "side_asset_pipeline.example.json", "targeted_canopy_recovery.default.json", "track_build.default.json",
    "track_graph.default.json", "vertical_conflict_photo_evidence.default.json", "vertical_hypotheses.default.json",
)
BENCHMARK_TEMPLATE_FILES = (
    "automation_boundary.example.csv", "dataset-manifest.example.json", "evaluation-input-v2.example.json",
    "evaluation-input.example.json", "experiment.example.json", "failure-case.example.json",
    "ground-truth.example.json", "human_review_log.example.csv", "metric-report.example.json",
    "rail-neutral-evidence.example.json", "split-manifest.example.json",
)
REQUIRED_ENTRY_FILES = (
    "src/railway_recon/__init__.py", "src/railway_recon/cli.py", "src/railway_recon/onboarding_demo.py",
    "scripts/bootstrap.ps1", "scripts/onboarding_common.ps1", "scripts/check_environment.py",
    "scripts/build_onboarding_demo.py", "scripts/new_project.ps1", "scripts/prepare_handoff_source.py",
    "tests/__init__.py", "web/src/main.js", "web/src/style.css", "web/src/project-loading.js",
)


def _explicit_paths(root: Path) -> set[Path]:
    paths = {root / name for name in (*ROOT_FILES, *REQUIRED_ENTRY_FILES)}
    for prefix, names in (
        ("docs", DOC_FILES), ("web", WEB_ROOT_FILES), ("web/public", WEB_PUBLIC_FILES),
        ("examples/minimal_project", EXAMPLE_FILES), ("benchmarks/protocols", PROTOCOL_FILES),
        ("configs/templates", CONFIG_TEMPLATE_FILES), ("benchmarks/templates", BENCHMARK_TEMPLATE_FILES),
        ("tests/fixtures/quality_gate", SAFE_TEST_FIXTURE_FILES),
    ):
        paths.update(root / prefix / name for name in names)
    return paths


def _check_source_path(root: Path, path: Path) -> None:
    relative = path.relative_to(root)
    safe_fixture = relative.as_posix() in {
        f"tests/fixtures/quality_gate/{name}" for name in SAFE_TEST_FIXTURE_FILES
    }
    if any(part.lower() in EXCLUDED_PARTS for part in relative.parts) and not safe_fixture:
        raise ValueError(f"Excluded directory in selected source path: {relative.as_posix()}")
    if path.resolve() != path or not path.resolve().is_relative_to(root):
        raise ValueError(f"Symlinks or redirected source paths are not allowed: {relative.as_posix()}")
    if not path.is_file():
        raise FileNotFoundError(f"Required handoff source is missing: {relative.as_posix()}")


def selected_source_files(root: Path) -> list[Path]:
    """Select only explicit text templates and source extensions in owned trees."""
    root = root.resolve()
    paths = _explicit_paths(root)
    for prefix, suffixes, recursive in (
        ("src/railway_recon", {".py"}, True),
        ("src/railway_recon/resources", {".json"}, False),
        ("scripts", {".py", ".ps1"}, False),
        ("tests", {".py"}, True),
        ("web/src", {".js", ".css"}, True),
        ("web/test", {".mjs"}, False),
    ):
        directory = root / prefix
        entries = directory.rglob("*") if recursive else directory.glob("*")
        for path in entries:
            relative = path.relative_to(root)
            if any(part.lower() in EXCLUDED_PARTS for part in relative.parts):
                continue
            if path.suffix in suffixes and (path.is_file() or path.is_symlink()):
                paths.add(path)
    for path in paths:
        _check_source_path(root, path)
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def inspect_source_text(payload: bytes, relative: str) -> None:
    if len(payload) > MAXIMUM_FILE_BYTES:
        raise ValueError(f"Selected text exceeds file-size bound: {relative}")
    text = payload.decode("utf-8-sig")
    if "\0" in text:
        raise ValueError(f"Selected source contains binary NUL bytes: {relative}")
    for name, pattern in SENSITIVE_PATTERNS.items():
        if pattern.search(text):
            raise ValueError(f"Known sensitive pattern ({name}) in selected source: {relative}")


def _destination(root: Path, output: Path) -> Path:
    if ".." in output.parts:
        raise ValueError("Output must not contain parent traversal")
    target = output.absolute()
    if target.resolve() != target:
        raise ValueError("Output must not use symlinks or redirected parents")
    if target == root or target in root.parents or target.parent == target:
        raise ValueError("Output must be a dedicated new directory, not source or an ancestor")
    if target.is_relative_to(root):
        allowed = any(target.is_relative_to(root / name) and target != root / name for name in ("projects", "tmp"))
        if not allowed:
            raise ValueError("Inside the repository, handoff output must be a new child of projects/ or tmp/")
    if any(part.lower() in {".git", ".venv", ".runtime", "node_modules"} for part in target.parts):
        raise ValueError("Output must not be inside protected runtime or repository directories")
    if target.exists():
        raise FileExistsError("Handoff output already exists; choose a new directory, never overwrite")
    return target


def prepare_handoff_source(root: str | Path, output: str | Path) -> dict:
    """Make a source-only local copy; no install, subprocess, upload or network."""
    source = Path(root).absolute()
    if source.resolve() != source:
        raise ValueError("Source root must not be symlinked or redirected")
    target = _destination(source, Path(output))
    paths = selected_source_files(source)
    if len(paths) > 2000:
        raise ValueError("Source count exceeds the bounded handoff allowance")
    snapshots = []
    total_bytes = 0
    for path in paths:
        relative = path.relative_to(source).as_posix()
        if path.stat().st_size > MAXIMUM_FILE_BYTES:
            raise ValueError(f"Selected source exceeds file-size bound: {relative}")
        payload = path.read_bytes()
        inspect_source_text(payload, relative)
        total_bytes += len(payload)
        if total_bytes > MAXIMUM_TOTAL_BYTES:
            raise ValueError("Handoff text exceeds total-size allowance")
        snapshots.append((path, relative, payload, hashlib.sha256(payload).hexdigest()))
    # Acquiring the new target fails atomically if another process created it.
    target.mkdir(parents=True, exist_ok=False)
    try:
        records = []
        for _, relative, payload, digest in snapshots:
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(payload)
            if destination.read_bytes() != payload:
                raise ValueError(f"Copied source bytes differ: {relative}")
            records.append({"path": relative, "bytes": len(payload), "sha256": digest})
        if selected_source_files(source) != paths:
            raise ValueError("Source inventory changed during copying; no completed manifest written")
        for path, relative, _, digest in snapshots:
            _check_source_path(source, path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Source changed during copying: {relative}")
        report = {
            "schema_version": "railway.newcomer-source-handoff.v1", "status": "completed",
            "role": "local_source_copy_for_internal_handoff_not_public_release",
            "network_used": False, "archive_created": False, "public_release": False,
            "selection_contract": "allowlisted_source_and_configuration_text_only",
            "excluded_categories": ["Git history", "runtime environments", "node_modules", "raw point clouds",
                                    "generated models", "customer workspaces", "private directories", "local web project.json",
                                    "environment files", "historical office documents"],
            "inspection_scope": "source_allowlist_UTF8_size_limits_and_known_sensitive_patterns",
            "limitations": ["Unknown identifiers embedded in source may not match known patterns.",
                            "This is not a privacy guarantee, legal clearance, or a publication decision.",
                            "The original LICENSE is preserved; no additional license grant is made.",
                            "Only selected onboarding documents are included, not every internal document.",
                            "Runtime installation, cold-start checks and model generation are separate steps."],
            "clean_start_verified_by_this_copy": False,
            "files": records, "file_count": len(records), "total_bytes": total_bytes,
            "manifest_self_hash_excluded": True,
        }
        with (target / MANIFEST_NAME).open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.write("\n")
        return report
    except Exception as error:
        with (target / "handoff_failure.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump({"status": "failed", "exception_type": type(error).__name__,
                       "recovery": "preserve_partial_copy_and_choose_a_new_directory"}, stream, indent=2)
            stream.write("\n")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="A new dedicated directory; existing directories are refused")
    args = parser.parse_args()
    try:
        report = prepare_handoff_source(ROOT, args.output)
        print(json.dumps({key: report[key] for key in ("status", "file_count", "total_bytes", "role")}, indent=2))
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"Source handoff stopped safely: {error}\nNo existing files were overwritten or deleted.\n")


if __name__ == "__main__":
    main()
