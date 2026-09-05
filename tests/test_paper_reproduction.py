from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "prepare_paper_reproduction.py"
SPEC = importlib.util.spec_from_file_location("paper_snapshot", SOURCE)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_detects_known_path_without_printing_content():
    sensitive = ("D:" + "\\" + "Railway" + "\\" + "private.txt").encode()
    with pytest.raises(ValueError, match="workspace_absolute_path"):
        module.inspect_text(sensitive, "source.py")


def test_rejects_large_or_nontext():
    with pytest.raises(ValueError, match="bound"):
        module.inspect_text(b"x" * (5 * 1024 * 1024 + 1), "source.py")
    with pytest.raises(UnicodeDecodeError):
        module.inspect_text(b"\xff", "source.py")


def test_snapshot_is_source_only_and_hash_bound(tmp_path, monkeypatch):
    source = tmp_path / "public.py"
    source.write_text("# Pure algorithm\n", encoding="utf-8")
    private = tmp_path / "customer_data"
    private.mkdir()
    (private / "payload.laz").write_bytes(b"not selected")
    monkeypatch.setattr(module, "selected_files", lambda root: [source])
    output = tmp_path / "benchmarks/local/paper-reproduction/new"
    report = module.prepare_snapshot(tmp_path, output)
    assert report["file_count"] == 2
    assert not (output / "customer_data").exists()
    assert report["published"] is False
    assert str(tmp_path) not in (output / "snapshot_manifest.json").read_text()
    assert json.loads((output / "snapshot_manifest.json").read_text())["files"] == report["files"]
    with pytest.raises(ValueError, match="existing"):
        module.prepare_snapshot(tmp_path, output)


def test_refuses_broad_or_escaping_output(tmp_path):
    for output in (tmp_path, tmp_path / "benchmarks/local/paper-reproduction", tmp_path / "elsewhere"):
        with pytest.raises(ValueError):
            module.prepare_snapshot(tmp_path, output)


def test_sensitive_source_blocks_before_destination_creation(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_text("password = '" + "sensitive-value" + "'", encoding="utf-8")
    monkeypatch.setattr(module, "selected_files", lambda root: [source])
    output = tmp_path / "benchmarks/local/paper-reproduction/new"
    with pytest.raises(ValueError, match="generic_secret"):
        module.prepare_snapshot(tmp_path, output)
    assert not output.exists()
