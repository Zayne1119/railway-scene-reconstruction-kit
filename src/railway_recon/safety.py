from __future__ import annotations

import re
from pathlib import Path
from typing import Any

BANNED_EXTENSIONS = {
    ".las", ".laz", ".e57", ".pcd", ".ply", ".splat", ".ksplat",
    ".glb", ".gltf", ".fbx", ".obj", ".mtl", ".blend",
    ".7z", ".rar", ".zip", ".mp4", ".mov", ".avi", ".tif", ".tiff",
}
BANNED_NAMES = {"camera.csv", ".env", "hosting.json"}
IGNORED_PARTS = {
    ".git", ".venv", "node_modules", "dist", ".next", ".pytest_cache",
    ".ruff_cache", ".uv-cache", "__pycache__", "projects", "runs",
}
IGNORED_PREFIXES = {("benchmarks", "local")}
TEXT_EXTENSIONS = {
    ".py", ".md", ".json", ".yml", ".yaml", ".toml", ".txt", ".ps1",
    ".js", ".ts", ".tsx", ".css", ".html", ".csv",
}
SENSITIVE_PATTERNS = {
    "workspace_absolute_path": re.compile(
        "(?:[A-Za-z]:" + r"\\" + "(?:Users|Rail" + "way)" + r"\\" + "|/ho" + "me/)" ,
        re.IGNORECASE,
    ),
    "private_ipv4": re.compile(
        r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
        r"192\.168\.\d{1,3}\.\d{1,3}|"
        r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"
    ),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "generic_secret": re.compile(r"(?i)(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"][^'\"]{8,}"),
    "known_case_marker": re.compile(
        "(?i)(?:" + "TL" + "13|Gap" + "Candidate|全景" + r"照片\.7z|app" + "gprj_)"
    ),
}


def safety_check(root: Path, maximum_file_bytes: int = 10 * 1024 * 1024) -> dict[str, Any]:
    root = root.resolve()
    findings: list[dict[str, Any]] = []
    scanned = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(relative.parts[: len(prefix)] == prefix for prefix in IGNORED_PREFIXES):
            continue
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        if not path.is_file():
            continue
        scanned += 1
        suffix = path.suffix.lower()
        if suffix in BANNED_EXTENSIONS:
            findings.append({"severity": "error", "kind": "banned_extension", "path": str(relative)})
        if path.name.lower() in BANNED_NAMES:
            findings.append({"severity": "error", "kind": "banned_filename", "path": str(relative)})
        size = path.stat().st_size
        if size > maximum_file_bytes:
            findings.append(
                {"severity": "error", "kind": "large_file", "path": str(relative), "bytes": size}
            )
        if suffix in TEXT_EXTENSIONS and size <= 2 * 1024 * 1024:
            try:
                text = path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                findings.append({"severity": "warning", "kind": "non_utf8_text", "path": str(relative)})
                continue
            for name, pattern in SENSITIVE_PATTERNS.items():
                if pattern.search(text):
                    findings.append({"severity": "error", "kind": name, "path": str(relative)})
    return {
        "root": str(root),
        "scanned_file_count": scanned,
        "finding_count": len(findings),
        "error_count": sum(item["severity"] == "error" for item in findings),
        "findings": findings,
        "passed": not any(item["severity"] == "error" for item in findings),
    }
