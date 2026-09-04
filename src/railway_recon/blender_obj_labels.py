from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

TOKEN_PREFIX_PATTERN = re.compile(r"^[A-Z][A-Z0-9]{1,7}$")
PROXY_TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$")


def _label_token(label: str, index: int, prefix: str) -> str:
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()[:16]
    token = f"{prefix}{index:06d}_{digest}"
    if len(token.encode("ascii")) > 63:
        raise ValueError("Generated Blender-safe label exceeds 63 bytes")
    return token


def write_obj_label_proxy(
    source: str | Path,
    proxy: str | Path,
    *,
    prefix: str = "RB",
) -> dict[str, str]:
    """Write an OBJ proxy with short, unique object labels.

    Blender ID names are limited to 63 bytes.  Long names with a shared prefix
    can therefore become ambiguous before registry metadata is attached.  This
    helper changes only ``o`` labels and returns ``short_label -> source_label``.
    The proxy must stay beside the source so relative material-library paths
    keep their original meaning.
    """
    source_path = Path(source).resolve()
    proxy_path = Path(proxy).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if proxy_path.exists():
        raise FileExistsError(proxy_path)
    if proxy_path.parent != source_path.parent:
        raise ValueError("OBJ label proxy must be written beside its source")
    if not TOKEN_PREFIX_PATTERN.fullmatch(prefix):
        raise ValueError("prefix must be 2-8 uppercase ASCII letters/digits")

    token_to_label: dict[str, str] = {}
    seen_labels: set[str] = set()
    try:
        with source_path.open("r", encoding="utf-8-sig") as input_stream, proxy_path.open(
            "x", encoding="utf-8", newline="\n"
        ) as output_stream:
            for line in input_stream:
                if not line.startswith("o "):
                    output_stream.write(line)
                    continue
                label = line[2:].strip()
                if not label:
                    raise ValueError("OBJ contains an empty object label")
                if label in seen_labels:
                    raise ValueError(f"OBJ object label is duplicated: {label}")
                token = _label_token(label, len(token_to_label), prefix)
                token_to_label[token] = label
                seen_labels.add(label)
                output_stream.write(f"o {token}\n")
    except Exception:
        proxy_path.unlink(missing_ok=True)
        raise
    if not token_to_label:
        proxy_path.unlink(missing_ok=True)
        raise ValueError(f"OBJ contains no object labels: {source_path}")
    return token_to_label


@contextmanager
def temporary_obj_label_proxy(
    source: str | Path,
    *,
    tag: str,
    prefix: str = "RB",
) -> Iterator[tuple[Path, dict[str, str]]]:
    """Yield a collision-safe import proxy and always remove it afterwards."""
    if not PROXY_TAG_PATTERN.fullmatch(tag):
        raise ValueError("tag contains unsupported characters")
    source_path = Path(source).resolve()
    proxy_path = source_path.with_name(f".{source_path.stem}.{tag}.obj")
    mapping = write_obj_label_proxy(source_path, proxy_path, prefix=prefix)
    try:
        yield proxy_path, mapping
    finally:
        proxy_path.unlink(missing_ok=True)

