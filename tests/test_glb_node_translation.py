from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from railway_recon.glb_node_translation import translate_named_glb_nodes
from railway_recon.recovery_binding import _read_glb


def _write_minimum_glb(path: Path, node_names: list[str]) -> bytes:
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(node_names)))}],
        "nodes": [{"name": name} for name in node_names],
        "extras": {
            "assetRegistry": [
                {
                    "id": name,
                    "index": index,
                    "evidenceScope": "observed",
                    "center": [1.0, 2.0, 3.0],
                    "bounds": {"min": [0.0, 1.0, 2.0], "max": [2.0, 3.0, 4.0]},
                    "parameters": {"existing": True},
                    "source": [],
                }
                for index, name in enumerate(node_names)
            ]
        },
    }
    json_bytes = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    binary = b"test"
    remainder = struct.pack("<II", len(binary), 0x004E4942) + binary
    total = 12 + 8 + len(json_bytes) + len(remainder)
    path.write_bytes(
        b"glTF"
        + struct.pack("<II", 2, total)
        + struct.pack("<II", len(json_bytes), 0x4E4F534A)
        + json_bytes
        + remainder
    )
    return remainder


def test_translate_nodes_preserves_binary_and_embeds_registry(tmp_path: Path) -> None:
    source = tmp_path / "source.glb"
    remainder = _write_minimum_glb(source, ["roof", "column"])
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps({"assets": [{"id": "roof"}, {"id": "column"}]}),
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.obj"
    candidate.write_text("o roof\nv 0 0 0\n", encoding="utf-8")
    output = tmp_path / "output.glb"
    report = tmp_path / "report.json"

    translate_named_glb_nodes(
        source_glb=source,
        output_glb=output,
        translations={"roof": (0.0, 0.0, -0.1)},
        registry_path=registry,
        candidate_obj_path=candidate,
        report_path=report,
        release_id="run09",
    )

    document, output_remainder, _ = _read_glb(output)
    assert output_remainder == remainder
    assert document["nodes"][0]["translation"] == [0.0, 0.0, -0.1]
    assert document["extras"]["assetRegistry"][0]["id"] == "roof"
    assert document["extras"]["assetRegistry"][0]["parameters"]["existing"] is True
    assert document["extras"]["assetRegistry"][0]["center"] == [1.0, 2.0, 2.9]
    assert document["extras"]["assetRegistry"][1]["center"] == [1.0, 2.0, 3.0]
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == "pass"


def test_translate_nodes_rejects_registry_node_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.glb"
    _write_minimum_glb(source, ["roof"])
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"assets": [{"id": "other"}]}), encoding="utf-8")
    candidate = tmp_path / "candidate.obj"
    candidate.write_text("o roof\n", encoding="utf-8")

    with pytest.raises(ValueError, match="do not exactly match"):
        translate_named_glb_nodes(
            source_glb=source,
            output_glb=tmp_path / "output.glb",
            translations={"roof": (0.0, 0.0, -0.1)},
            registry_path=registry,
            candidate_obj_path=candidate,
            report_path=tmp_path / "report.json",
            release_id="run09",
        )
