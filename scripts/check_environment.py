"""只读环境体检：检查选定范围，不安装依赖、不联网、不读取客户数据。"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # The doctor can explain an older bootstrap interpreter.
    tomllib = None

ROOT = Path(__file__).resolve().parents[1]
SETUP_COMMAND = r"powershell -ExecutionPolicy Bypass -File .\Railway.ps1 setup"
DEPENDENCIES = {"numpy": "numpy", "scipy": "scipy.spatial", "pillow": "PIL.Image",
                "laspy": "laspy", "lazrs": "lazrs", "jsonschema": "jsonschema"}
PYTHON_PROBE = """
import importlib, importlib.metadata, json, sys
names = {'numpy':'numpy','scipy':'scipy.spatial','pillow':'PIL.Image',
         'laspy':'laspy','lazrs':'lazrs','jsonschema':'jsonschema'}
packages = {}
for name, module in names.items():
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = None
    try:
        importlib.import_module(module)
        imported, error = True, None
    except Exception as exception:
        imported, error = False, type(exception).__name__
    packages[name] = {'version':version,'imported':imported,'error_type':error}
print(json.dumps({'python':list(sys.version_info[:3]),'platform':sys.platform,'packages':packages}))
"""
NODE_PROBE = """
const result = {};
for (const name of ['three', 'vite']) {
  try { await import(name); result[name] = {loaded: true}; }
  catch (_) { result[name] = {loaded: false}; }
}
console.log(JSON.stringify(result));
"""


def safe_path(value: str | Path) -> Path:
    text = str(value)
    if not text or len(text) > 2048 or any(ord(character) < 32 for character in text):
        raise ValueError("路径必须是长度不超过 2048 的有效单行文本")
    return Path(text).expanduser().resolve()


def run_probe(arguments: list[str], *, timeout: float, cwd: Path) -> dict:
    """Never execute a shell, echo command stderr, or expose environment values."""
    try:
        result = subprocess.run(arguments, cwd=cwd, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout,
                                shell=False, check=False,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "timeout"}
    except OSError:
        return {"ok": False, "reason": "execution_failed"}
    if result.returncode != 0:
        return {"ok": False, "reason": "nonzero_exit"}
    if len(result.stdout) > 65536:
        return {"ok": False, "reason": "unexpected_output_size"}
    return {"ok": True, "stdout": result.stdout.strip()}


def version_tuple(value: str) -> tuple[int, ...]:
    if not isinstance(value, str) or not re.fullmatch(r"\d+(?:\.\d+){0,3}", value):
        raise ValueError("不支持的版本格式")
    numbers = tuple(int(item) for item in value.split("."))
    return (*numbers, *(0 for _ in range(3 - len(numbers))))


def marker_applies(marker: str, *, python: str, target_platform: str) -> bool:
    """Evaluate the lock's small marker grammar without eval or third-party code."""
    if not isinstance(marker, str) or len(marker) > 1000:
        raise ValueError("锁文件 marker 无效")
    environment = {"python_full_version": python, "python_version": ".".join(python.split(".")[:2]),
                   "sys_platform": target_platform}

    def visit(node):
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            values = [visit(item) for item in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if (not isinstance(node, ast.Compare) or len(node.ops) != 1
                or not isinstance(node.left, ast.Name) or node.left.id not in environment
                or len(node.comparators) != 1 or not isinstance(node.comparators[0], ast.Constant)
                or not isinstance(node.comparators[0].value, str)):
            raise ValueError("锁文件使用了不支持的 marker；不能猜测依赖版本")
        left, right = environment[node.left.id], node.comparators[0].value
        if node.left.id.startswith("python"):
            left, right = version_tuple(left), version_tuple(right)
        operations = {ast.Eq: lambda: left == right, ast.NotEq: lambda: left != right,
                      ast.Lt: lambda: left < right, ast.LtE: lambda: left <= right,
                      ast.Gt: lambda: left > right, ast.GtE: lambda: left >= right}
        operation = operations.get(type(node.ops[0]))
        if operation is None:
            raise ValueError("锁文件 marker 比较符不受支持")
        return operation()

    try:
        return bool(visit(ast.parse(marker, mode="eval").body))
    except SyntaxError as error:
        raise ValueError("锁文件 marker 无法解析") from error


def locked_python_versions(path: Path, *, python: str, target_platform: str) -> dict:
    if tomllib is None:
        raise ValueError("请使用 Python 3.11 或更新版本运行体检，以读取 uv.lock")
    with path.open("rb") as stream:
        lock = tomllib.load(stream)
    selected = {name: set() for name in DEPENDENCIES}
    for package in lock.get("package", []):
        name = str(package.get("name", "")).lower().replace("_", "-")
        if name not in selected:
            continue
        markers = package.get("resolution-markers", [])
        if markers and not any(marker_applies(marker, python=python, target_platform=target_platform)
                               for marker in markers):
            continue
        selected[name].add(package.get("version"))
    if any(len(versions) != 1 or not all(isinstance(v, str) and v for v in versions)
           for versions in selected.values()):
        raise ValueError("uv.lock 无法为六个运行依赖提供唯一版本；需要检查锁文件")
    return {name: next(iter(versions)) for name, versions in selected.items()}


def node_engine_matches(version: str, requirement: str) -> bool:
    """Support the actual simple >= / caret / OR Node engine contracts; fail closed."""
    current = version_tuple(version)
    if not isinstance(requirement, str) or len(requirement) > 300:
        raise ValueError("Node engines 声明无效")
    results = []
    for alternative in requirement.split("||"):
        tokens = alternative.split()
        if not tokens:
            raise ValueError("Node engines 声明为空")
        matches = []
        for token in tokens:
            match = re.fullmatch(r"(>=|<=|>|<|=|\^)?(\d+(?:\.\d+){0,2})", token)
            if match is None:
                raise ValueError("Node engines 声明超出可验证格式")
            operation, literal = match.groups()
            wanted = version_tuple(literal)
            checks = {">=": current >= wanted, "<=": current <= wanted, ">": current > wanted,
                      "<": current < wanted, "=": current == wanted, None: current == wanted,
                      "^": wanted <= current < (wanted[0] + 1, 0, 0)}
            matches.append(checks[operation])
        results.append(all(matches))
    return any(results)


def _add(checks: list, identifier: str, passed: bool, message: str, action: str = "", *,
         required: bool = True, observed=None, expected=None) -> None:
    checks.append({"id": identifier, "required": required,
                   "status": "pass" if passed else ("fail" if required else "warning"),
                   "message_zh": message, "action_zh": "" if passed else action,
                   "observed": observed, "expected": expected})


def _load_json(path: Path) -> dict:
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("配置文件超过读取上限")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError("配置文件必须是 JSON 对象")
    return value


def _core_checks(root: Path, executable: Path | None, timeout: float, checks: list) -> None:
    for name in ("pyproject.toml", "uv.lock"):
        _add(checks, "file." + name, (root / name).is_file(), f"Python 项目文件：{name}",
             "请使用完整 toolkit 目录，或通过 --repo-root 指定正确目录。")
    selected = executable or root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    exists = selected.is_file()
    _add(checks, "python.executable", exists, "选定的 Python 解释器存在",
         "先运行项目安装脚本建立 .venv，或用 --python 指定 Python 3.11 可执行文件。",
         observed="explicit" if executable else "repository_venv", expected="Python 3.11")
    if not exists:
        return
    result = run_probe([str(selected), "-I", "-B", "-c", PYTHON_PROBE], timeout=timeout, cwd=root)
    try:
        if not result["ok"]:
            raise ValueError("probe failed")
        runtime = json.loads(result["stdout"])
        numbers = runtime["python"]
        if (not isinstance(numbers, list) or len(numbers) != 3
                or any(type(number) is not int or not 0 <= number <= 999 for number in numbers)
                or runtime["platform"] not in {"win32", "linux", "darwin", "emscripten"}
                or not isinstance(runtime["packages"], dict)):
            raise ValueError("invalid runtime response")
    except (ValueError, KeyError, TypeError):
        _add(checks, "python.probe", False, "Python 依赖探测未正常完成",
             "检查解释器和本机依赖安装；若超时，可增加 --timeout（最长 60 秒）。",
             observed=result.get("reason", "invalid_probe_response"))
        return
    version = ".".join(map(str, numbers))
    _add(checks, "python.version", numbers[:2] == [3, 11], "新手流程统一使用 Python 3.11",
         "用 Python 3.11 重新创建 .venv；不要混用其他版本的现有环境。",
         observed=version, expected="3.11.x")
    try:
        pinned = locked_python_versions(root / "uv.lock", python=version,
                                        target_platform=runtime["platform"])
    except (OSError, ValueError, TypeError, KeyError):
        _add(checks, "python.lock", False, "uv.lock 依赖版本无法可靠解析",
             "确认 uv.lock 完整，并使用 Python 3.11 运行体检；不要手工猜测版本。")
        return
    for name in DEPENDENCIES:
        value = runtime["packages"].get(name, {})
        installed = value.get("version") if isinstance(value, dict) else None
        imported = value.get("imported") is True if isinstance(value, dict) else False
        if installed is not None and (not isinstance(installed, str)
                                      or not re.fullmatch(r"[A-Za-z0-9.+!_-]{1,64}", installed)):
            installed = "invalid_version_response"
        _add(checks, "python.package." + name, imported and installed == pinned[name],
             f"{name} 已安装、可导入且版本匹配 uv.lock",
             f"运行 {SETUP_COMMAND}，然后重新体检。",
             observed={"version": installed, "imported": imported}, expected=pinned[name])


def _node_executable(directory: Path | None) -> Path | None:
    if directory is not None:
        candidate = directory / ("node.exe" if os.name == "nt" else "node")
        return candidate if candidate.is_file() else None
    found = shutil.which("node")
    return Path(found).resolve() if found else None


def _npm_cli(node: Path) -> Path | None:
    direct = node.parent / "node_modules/npm/bin/npm-cli.js"
    if direct.is_file():
        return direct
    found = shutil.which("npm")
    if found:
        resolved = Path(found).resolve()
        if resolved.name == "npm-cli.js" and resolved.is_file():
            return resolved
        nearby = resolved.parent / "node_modules/npm/bin/npm-cli.js"
        if nearby.is_file():
            return nearby
    return None


def _web_checks(root: Path, directory: Path | None, timeout: float, checks: list) -> None:
    web = root / "web"
    for name in ("package.json", "package-lock.json"):
        _add(checks, "file.web." + name, (web / name).is_file(), f"网页项目文件：web/{name}",
             "请使用 toolkit 自带的 web 目录，不要指向客户项目网页目录。")
    try:
        package, lock = _load_json(web / "package.json"), _load_json(web / "package-lock.json")
        lock_root = lock["packages"][""]
        expected_modules = {**package.get("dependencies", {}), **package.get("devDependencies", {})}
        consistent = all(package.get(key, {}) == lock_root.get(key, {})
                         for key in ("dependencies", "devDependencies"))
        if not {"three", "vite"}.issubset(expected_modules) or lock.get("lockfileVersion") not in {2, 3}:
            raise ValueError("unexpected web manifest")
    except (OSError, ValueError, KeyError, TypeError):
        _add(checks, "web.lock", False, "网页依赖锁文件无法可靠解析",
             "恢复匹配的 package.json 和 package-lock.json 后运行网页安装阶段。")
        return
    _add(checks, "web.lock", consistent, "网页声明与 package-lock.json 一致",
         "请先修复依赖声明和锁文件不一致，不要只复制 node_modules。")
    node = _node_executable(directory)
    _add(checks, "node.executable", node is not None, "找到 Node.js",
         "安装 Node.js 22 的当前维护版本，或通过 --node-directory 指定其目录。",
         observed="explicit_directory" if directory else "PATH", expected=">=22")
    node_version = None
    if node is not None:
        result = run_probe([str(node), "--version"], timeout=timeout, cwd=web)
        text = result.get("stdout", "")
        if result["ok"] and re.fullmatch(r"v\d+\.\d+\.\d+", text):
            node_version = text[1:]
        requirements = [">=22", package.get("engines", {}).get("node", ">=22")]
        for name in expected_modules:
            entry = lock.get("packages", {}).get("node_modules/" + name, {})
            if entry.get("engines", {}).get("node"):
                requirements.append(entry["engines"]["node"])
        try:
            compatible = node_version is not None and all(node_engine_matches(node_version, item)
                                                         for item in requirements)
        except ValueError:
            compatible = False
        _add(checks, "node.version", compatible, "Node.js 满足项目及锁定工具链版本要求",
             "安装满足锁定 Vite engines 的 Node.js 22 维护版本；仅主版本 22 不一定足够。",
             observed=node_version or result.get("reason", "invalid_version_response"),
             expected=sorted(set(requirements)))
        npm = _npm_cli(node)
        npm_result = (run_probe([str(node), str(npm), "--version"], timeout=timeout, cwd=web)
                      if npm else {"ok": False, "reason": "npm_cli_missing"})
        npm_version = npm_result.get("stdout", "")
        npm_ok = npm_result["ok"] and re.fullmatch(r"\d+\.\d+\.\d+", npm_version) is not None
        _add(checks, "npm.available", npm_ok, "npm 可用（未运行安装命令）",
             "使用包含 npm 的完整 Node.js 发行包，不要只复制 node.exe。",
             observed=npm_version if npm_ok else npm_result.get("reason", "invalid_version_response"))
    for name in expected_modules:
        if not re.fullmatch(r"(?:@[a-z0-9_.-]+/)?[a-z0-9_.-]+", name):
            _add(checks, "web.package_name", False, "网页依赖名称无效", "检查 package.json。")
            continue
        pinned = lock.get("packages", {}).get("node_modules/" + name, {}).get("version")
        try:
            installed = _load_json(web / "node_modules" / name / "package.json").get("version")
        except (OSError, ValueError, TypeError):
            installed = None
        _add(checks, "web.package." + name, isinstance(pinned, str) and installed == pinned,
             f"网页依赖 {name} 与 package-lock.json 一致",
             f"运行 {SETUP_COMMAND} 安装锁定网页依赖（npm ci），然后重新体检。",
             observed=installed, expected=pinned)
    if node is not None and node_version is not None:
        result = run_probe([str(node), "--input-type=module", "-e", NODE_PROBE], timeout=timeout, cwd=web)
        try:
            imports = json.loads(result.get("stdout", "")) if result["ok"] else {}
            loaded = all(imports.get(name, {}).get("loaded") is True for name in ("three", "vite"))
        except (ValueError, TypeError, AttributeError):
            loaded = False
        _add(checks, "web.imports", loaded, "Three.js 与 Vite 可由 Node.js 实际加载",
             "确认 Node 版本并重新安装网页依赖；安装存在不等于运行可用。")


def check_environment(repo_root: Path, *, scope: str = "full", python_executable: Path | None = None,
                      node_directory: Path | None = None, timeout: float = 15.0) -> dict:
    if scope not in {"core", "demo", "web", "full"}:
        raise ValueError("scope 必须是 core、demo、web 或 full")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 1 <= timeout <= 60:
        raise ValueError("timeout 必须在 1 到 60 秒之间")
    root = safe_path(repo_root)
    executable = safe_path(python_executable) if python_executable is not None else None
    node_directory = safe_path(node_directory) if node_directory is not None else None
    checks = []
    _add(checks, "repository.directory", root.is_dir(), "项目根目录存在",
         "通过 --repo-root 指定包含 pyproject.toml 的 toolkit 根目录。")
    if root.is_dir():
        if scope in {"core", "demo", "full"}:
            try:
                _core_checks(root, executable, timeout, checks)
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                _add(checks, "core.configuration", False, "Python 配置或探测结果格式异常",
                     f"检查项目配置，运行 {SETUP_COMMAND} 后重试。")
        if scope in {"web", "full"}:
            try:
                _web_checks(root, node_directory, timeout, checks)
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                _add(checks, "web.configuration", False, "网页配置或探测结果格式异常",
                     f"检查网页配置，运行 {SETUP_COMMAND} 后重试。")
    blender = shutil.which("blender") is not None
    _add(checks, "blender.optional", blender, "Blender 为可选工具；OBJ/GLB 示例不依赖它",
         "可跳过；只有需要 Blender 专用导出或渲染时才安装并加入 PATH。",
         required=False, observed="found_on_PATH" if blender else "not_on_PATH")
    failed = sum(item["status"] == "fail" for item in checks)
    warnings = sum(item["status"] == "warning" for item in checks)
    return {"schema_version": "railway.environment-doctor.v1", "created_at_utc": datetime.now(UTC).isoformat(),
            "scope": scope, "ready": failed == 0, "status": "ready" if failed == 0 else "not_ready",
            "required_failure_count": failed, "warning_count": warnings, "checks": checks,
            "scope_definition": {"core": "Python 3.11 与六个锁定运行依赖",
                                 "demo": "同 core；不要求 Node 或 Blender",
                                 "web": "Node/npm 与网页锁定直接依赖及模块加载",
                                 "full": "core 与 web 两者"}[scope],
            "boundary_zh": "只代表选定范围的基础环境检查；不替代模型生成、网页构建或视觉验收。",
            "network_accessed": False, "dependencies_installed": False, "customer_data_read": False,
            "absolute_tool_paths_in_report": False}


def chinese_summary(report: dict) -> str:
    first = (f"环境检查通过（{report['scope']}）；可进行该范围的下一步。" if report["ready"]
             else f"环境尚未就绪（{report['scope']}）：{report['required_failure_count']} 项必需检查未通过。")
    lines = [first]
    for item in report["checks"]:
        if item["status"] == "fail":
            lines.append(f"- {item['message_zh']}：{item['action_zh']}")
    if report["warning_count"]:
        lines.append("提示：未找到 PATH 中的 Blender 不会阻止 OBJ/GLB 示例。")
    lines.append(report["boundary_zh"])
    return "\n".join(lines)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=safe_path, default=ROOT)
    parser.add_argument("--scope", choices=("core", "demo", "web", "full"), default="full")
    parser.add_argument("--python", type=safe_path, dest="python_executable")
    parser.add_argument("--node-directory", type=safe_path)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--output", type=safe_path, help="可选新 JSON 文件；拒绝覆盖既存文件")
    parser.add_argument("--json", action="store_true", help="stdout 输出 JSON，中文摘要写入 stderr")
    args = parser.parse_args()
    try:
        if args.output is not None and args.output.exists():
            raise FileExistsError("报告文件已存在；请选择新的 --output 路径")
        report = check_environment(args.repo_root, scope=args.scope, python_executable=args.python_executable,
                                   node_directory=args.node_directory, timeout=args.timeout)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
                stream.write("\n")
    except (OSError, ValueError, TypeError):
        parser.exit(2, "无法完成体检：请检查参数、目录权限和报告输出是否已存在；原始错误内容未输出。\n")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    print(chinese_summary(report), file=sys.stderr if args.json else sys.stdout)
    if not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
