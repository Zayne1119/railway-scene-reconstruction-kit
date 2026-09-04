from __future__ import annotations

from pathlib import Path

import pytest

from railway_recon.blender_obj_labels import (
    temporary_obj_label_proxy,
    write_obj_label_proxy,
)


def source_obj(path: Path) -> tuple[str, str]:
    shared = "SITE--TRACK--CATENARY--VERY-LONG-SHARED-SEMANTIC-PREFIX--"
    first = shared + "POSITIONER-0005"
    second = shared + "POSITIONER-0006"
    path.write_text(
        "mtllib source.mtl\n"
        f"o {first}\n"
        "v 0 0 0\n"
        f"o {second}\n"
        "v 1 0 0\n",
        encoding="utf-8",
    )
    return first, second


def test_proxy_replaces_only_object_labels_and_keeps_them_blender_safe(tmp_path: Path) -> None:
    source = tmp_path / "source.obj"
    first, second = source_obj(source)
    proxy = tmp_path / ".source.import.obj"

    mapping = write_obj_label_proxy(source, proxy)
    lines = proxy.read_text(encoding="utf-8").splitlines()
    proxy_labels = [line[2:] for line in lines if line.startswith("o ")]

    assert list(mapping.values()) == [first, second]
    assert proxy_labels == list(mapping)
    assert all(len(label.encode("ascii")) <= 63 for label in proxy_labels)
    assert len(set(proxy_labels)) == 2
    assert lines[0] == "mtllib source.mtl"
    assert [line for line in lines if line.startswith("v ")] == ["v 0 0 0", "v 1 0 0"]


def test_temporary_proxy_is_removed_even_when_import_fails(tmp_path: Path) -> None:
    source = tmp_path / "source.obj"
    source_obj(source)
    proxy_path: Path | None = None

    with (
        pytest.raises(RuntimeError, match="simulated importer failure"),
        temporary_obj_label_proxy(source, tag="engine-import") as (proxy, mapping),
    ):
        proxy_path = proxy
        assert proxy.is_file()
        assert len(mapping) == 2
        raise RuntimeError("simulated importer failure")

    assert proxy_path is not None
    assert not proxy_path.exists()


def test_duplicate_semantic_labels_fail_without_leaving_proxy(tmp_path: Path) -> None:
    source = tmp_path / "source.obj"
    source.write_text("o DUPLICATE\nv 0 0 0\no DUPLICATE\nv 1 0 0\n", encoding="utf-8")
    proxy = tmp_path / ".source.import.obj"

    with pytest.raises(ValueError, match="duplicated"):
        write_obj_label_proxy(source, proxy)

    assert not proxy.exists()


def test_proxy_refuses_cross_directory_material_resolution(tmp_path: Path) -> None:
    source = tmp_path / "source.obj"
    source_obj(source)
    other = tmp_path / "other"
    other.mkdir()

    with pytest.raises(ValueError, match="beside"):
        write_obj_label_proxy(source, other / "proxy.obj")
