"""Fixture-only source-copy checks; never read or copy production project data."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_handoff_source.py"
SPEC = importlib.util.spec_from_file_location("handoff_source", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def fixture_source(root):
    for path in module._explicit_paths(root):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" if path.suffix == ".json" else "# Independent safe fixture\n", encoding="utf-8")
    for relative in ("src/railway_recon/safe.py", "src/railway_recon/resources/safe.json",
                     "scripts/extra.ps1", "scripts/extra.py", "tests/nested/test_fixture.py",
                     "web/src/sub/module.js", "web/src/sub/style.css", "web/test/fixture.test.mjs"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" if path.suffix == ".json" else "# Independent safe fixture\n", encoding="utf-8")
    return root


def test_copies_required_web_launcher_tests_templates_and_only_source_text(tmp_path):
    root = fixture_source(tmp_path / "source")
    excluded = (
        ".env", ".git/config", ".venv/secret.py", ".runtime/node.exe", "node_modules/client.js",
        "projects/customer/site.laz", "private/customer.json", "web/public/project.json",
        "web/public/demo/scene.glb", "web/public/unknown.json", "web/dist/index.html",
        "docs/history.docx", "docs/private_notes.md", "benchmarks/local/private/results.json",
        "tests/private/example.py", "src/railway_recon/private/secret.py", "scripts/private.json",
        "tests/fixtures/customer.laz", "examples/minimal_project/site.laz",
        "tests/fixtures/quality_gate/not-allowlisted.json",
        "configs/templates/not-allowlisted.json", "benchmarks/protocols/not-allowlisted.json",
    )
    for relative in excluded:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"private-source-marker-not-to-copy")
    output = tmp_path / "handoff"
    report = module.prepare_handoff_source(root, output)
    assert report["status"] == "completed"
    assert report["network_used"] is False
    assert report["public_release"] is False
    assert report["clean_start_verified_by_this_copy"] is False
    for relative in excluded:
        assert not (output / relative).exists(), relative
    for relative in ("Railway.ps1", "uv.lock", "web/package-lock.json", "web/server-config.js",
                     "web/src/project-loading.js", "web/test/fixture.test.mjs", "tests/__init__.py",
                     "tests/nested/test_fixture.py", "src/railway_recon/resources/safe.json",
                     "tests/fixtures/quality_gate/release_local_pass.json",
                     "examples/minimal_project/reviewed_scene.example.json"):
        assert (output / relative).is_file(), relative
    for record in report["files"]:
        payload = (output / record["path"]).read_bytes()
        assert b"private-source-marker-not-to-copy" not in payload
        assert len(payload) == record["bytes"]
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]
    assert report["file_count"] == len(report["files"])
    assert report["total_bytes"] == sum(item["bytes"] for item in report["files"])
    assert json.loads((output / module.MANIFEST_NAME).read_bytes()) == report


def test_missing_required_entry_fails_before_output_creation(tmp_path):
    root = fixture_source(tmp_path / "source")
    (root / "Railway.ps1").unlink()
    output = tmp_path / "handoff"
    with pytest.raises(FileNotFoundError, match="Railway.ps1"):
        module.prepare_handoff_source(root, output)
    assert not output.exists()


def test_existing_or_source_outputs_are_never_modified(tmp_path):
    root = fixture_source(tmp_path / "source")
    owner = tmp_path / "existing"
    owner.mkdir()
    note = owner / "keep.txt"
    note.write_text("keep", encoding="utf-8")
    for output in (owner, root, root / "src/new", root / "web/new", root / "projects", tmp_path):
        with pytest.raises((ValueError, FileExistsError)):
            module.prepare_handoff_source(root, output)
    assert note.read_text() == "keep"


def test_new_project_child_and_external_new_directory_are_allowed(tmp_path):
    root = fixture_source(tmp_path / "source")
    for output in (root / "projects/handoff-v1", root / "tmp/handoff-v2", tmp_path / "external-handoff"):
        report = module.prepare_handoff_source(root, output)
        assert report["status"] == "completed"
        assert not (output / "projects").exists()


def test_traversal_is_rejected(tmp_path):
    root = fixture_source(tmp_path / "source")
    with pytest.raises(ValueError, match="traversal"):
        module.prepare_handoff_source(root, tmp_path / "new/../escaped")


def test_sensitive_or_binary_selected_source_fails_before_copy(tmp_path):
    root = fixture_source(tmp_path / "source")
    path = root / "src/railway_recon/safe.py"
    for data, exception in ((b"\xff", UnicodeDecodeError), (b"a\0b", ValueError),
                            (("password = '" + "sensitive-value" + "'").encode(), ValueError)):
        path.write_bytes(data)
        output = tmp_path / "never-created"
        with pytest.raises(exception):
            module.prepare_handoff_source(root, output)
        assert not output.exists()


def test_source_symlink_is_rejected_or_platform_skip(tmp_path):
    root = fixture_source(tmp_path / "source")
    outside = tmp_path / "owner.py"
    outside.write_text("# Private fixture outside source\n", encoding="utf-8")
    linked = root / "src/railway_recon/linked.py"
    try:
        linked.symlink_to(outside)
    except OSError:
        pytest.skip("Creating symlinks requires unavailable OS privilege")
    with pytest.raises(ValueError, match="Symlinks"):
        module.prepare_handoff_source(root, tmp_path / "handoff")


def test_source_change_during_copy_leaves_failure_not_completed_manifest(tmp_path, monkeypatch):
    root = fixture_source(tmp_path / "source")
    original = module.selected_source_files
    calls = 0

    def changed(directory):
        nonlocal calls
        calls += 1
        result = original(directory)
        if calls == 2:
            (root / "README.md").write_text("# Changed fixture source\n", encoding="utf-8")
        return result

    monkeypatch.setattr(module, "selected_source_files", changed)
    output = tmp_path / "handoff"
    with pytest.raises(ValueError, match="Source changed"):
        module.prepare_handoff_source(root, output)
    assert not (output / module.MANIFEST_NAME).exists()
    assert (output / "handoff_failure.json").is_file()
    assert (output / "README.md").read_text() == "# Independent safe fixture\n"


def test_text_size_limit_is_checked(tmp_path, monkeypatch):
    root = fixture_source(tmp_path / "source")
    monkeypatch.setattr(module, "MAXIMUM_FILE_BYTES", 3)
    with pytest.raises(ValueError, match="file-size"):
        module.prepare_handoff_source(root, tmp_path / "handoff")
