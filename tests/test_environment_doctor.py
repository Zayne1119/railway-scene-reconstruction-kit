from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("environment_doctor", ROOT / "scripts/check_environment.py")
DOCTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOCTOR)


class EnvironmentDoctorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="railway-doctor-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "pyproject.toml").write_text('[project]\nname="fixture"\n', encoding="utf-8")
        self.versions = {name: "1.2.3" for name in DOCTOR.DEPENDENCIES}
        text = '\n'.join(f'[[package]]\nname="{name}"\nversion="{version}"\n'
                         for name, version in self.versions.items())
        (self.root / "uv.lock").write_text(text, encoding="utf-8")
        self.python = self.root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        self.python.parent.mkdir(parents=True)
        self.python.write_bytes(b"fixture-not-executed")

    def runtime(self, **updates):
        payload = {"python": [3, 11, 15], "platform": "win32", "packages": {
            name: {"version": version, "imported": True, "error_type": None}
            for name, version in self.versions.items()}}
        payload.update(updates)
        return {"ok": True, "stdout": json.dumps(payload)}

    def web(self):
        root = self.root / "web"
        root.mkdir()
        package = {"dependencies": {"three": "^0.180.0"}, "devDependencies": {"vite": "^7.1.0"},
                   "engines": {"node": ">=22"}}
        lock = {"lockfileVersion": 3, "packages": {"": package,
                "node_modules/three": {"version": "0.180.0"},
                "node_modules/vite": {"version": "7.3.1", "engines": {"node": "^20.19.0 || >=22.12.0"}}}}
        (root / "package.json").write_text(json.dumps(package), encoding="utf-8")
        (root / "package-lock.json").write_text(json.dumps(lock), encoding="utf-8")
        for name, version in (("three", "0.180.0"), ("vite", "7.3.1")):
            path = root / "node_modules" / name / "package.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"version": version}), encoding="utf-8")

    def test_core_ready_without_node_or_blender(self):
        with (patch.object(DOCTOR, "run_probe", return_value=self.runtime()) as run,
              patch.object(DOCTOR.shutil, "which", return_value=None) as which):
            report = DOCTOR.check_environment(self.root, scope="core")
        self.assertTrue(report["ready"])
        self.assertEqual(report["warning_count"], 1)
        self.assertEqual(run.call_count, 1)
        which.assert_called_once_with("blender")
        self.assertFalse(any(item["id"].startswith("node") for item in report["checks"]))
        self.assertFalse(report["network_accessed"])

    def test_demo_is_core_and_optional_blender_never_blocks(self):
        with (patch.object(DOCTOR, "run_probe", return_value=self.runtime()),
              patch.object(DOCTOR.shutil, "which", return_value=None)):
            report = DOCTOR.check_environment(self.root, scope="demo")
        self.assertTrue(report["ready"])
        check = next(item for item in report["checks"] if item["id"] == "blender.optional")
        self.assertFalse(check["required"])
        self.assertEqual(check["status"], "warning")

    def test_python_dependency_missing_or_wrong_version_fails(self):
        for changed in ({"version": None, "imported": False}, {"version": "9.0", "imported": True}):
            payload = json.loads(self.runtime()["stdout"])
            payload["packages"]["numpy"] = changed
            with (self.subTest(changed=changed),
                  patch.object(DOCTOR, "run_probe", return_value={"ok": True, "stdout": json.dumps(payload)}),
                  patch.object(DOCTOR.shutil, "which", return_value=None)):
                report = DOCTOR.check_environment(self.root, scope="core")
            self.assertFalse(report["ready"])
            self.assertEqual(report["required_failure_count"], 1)
            self.assertIn("Railway.ps1 setup", DOCTOR.chinese_summary(report))

    def test_python_312_is_not_onboarding_ready(self):
        with (patch.object(DOCTOR, "run_probe", return_value=self.runtime(python=[3, 12, 1])),
              patch.object(DOCTOR.shutil, "which", return_value=None)):
            report = DOCTOR.check_environment(self.root, scope="core")
        self.assertFalse(report["ready"])
        self.assertEqual(next(item for item in report["checks"] if item["id"] == "python.version")["status"], "fail")

    def test_missing_explicit_interpreter_does_not_fall_back_to_another_python(self):
        with (patch.object(DOCTOR, "run_probe") as run,
              patch.object(DOCTOR.shutil, "which", return_value=None)):
            report = DOCTOR.check_environment(self.root, scope="core", python_executable=self.root / "missing.exe")
        self.assertFalse(report["ready"])
        run.assert_not_called()

    def test_probe_failure_is_actionable_and_does_not_echo_raw_output(self):
        with (patch.object(DOCTOR, "run_probe", return_value={"ok": False, "reason": "timeout", "stderr": "SECRET=xyz"}),
              patch.object(DOCTOR.shutil, "which", return_value=None)):
            report = DOCTOR.check_environment(self.root, scope="core")
        self.assertFalse(report["ready"])
        self.assertNotIn("SECRET", json.dumps(report))
        self.assertIn("timeout", json.dumps(report))

    def test_lock_markers_choose_311_not_highest_version(self):
        path = self.root / "uv.lock"
        text = path.read_text(encoding="utf-8").replace('name="numpy"\nversion="1.2.3"',
            'name="numpy"\nversion="1.2.3"\nresolution-markers=["python_full_version < \'3.12\' and sys_platform == \'win32\'"]')
        text += '\n[[package]]\nname="numpy"\nversion="9.0.0"\nresolution-markers=["python_full_version >= \'3.12\'"]\n'
        path.write_text(text, encoding="utf-8")
        pinned = DOCTOR.locked_python_versions(path, python="3.11.15", target_platform="win32")
        self.assertEqual(pinned["numpy"], "1.2.3")

    def test_ambiguous_lock_and_unsafe_marker_fail_closed(self):
        with self.assertRaises(ValueError):
            DOCTOR.marker_applies("__import__('os').system('not-run')", python="3.11.15", target_platform="win32")
        path = self.root / "uv.lock"
        path.write_text(path.read_text() + '\n[[package]]\nname="numpy"\nversion="9.0"\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            DOCTOR.locked_python_versions(path, python="3.11.15", target_platform="win32")

    def test_web_ready_checks_node_npm_versions_and_real_imports_without_python(self):
        self.web()
        responses = [{"ok": True, "stdout": "v22.14.0"}, {"ok": True, "stdout": "10.9.2"},
                     {"ok": True, "stdout": '{"three":{"loaded":true},"vite":{"loaded":true}}'}]
        with (patch.object(DOCTOR, "_node_executable", return_value=self.root / "node.exe"),
              patch.object(DOCTOR, "_npm_cli", return_value=self.root / "npm-cli.js"),
              patch.object(DOCTOR, "run_probe", side_effect=responses) as run,
              patch.object(DOCTOR.shutil, "which", return_value=None)):
            report = DOCTOR.check_environment(self.root, scope="web")
        self.assertTrue(report["ready"])
        self.assertEqual(run.call_count, 3)
        self.assertFalse(any(item["id"].startswith("python") for item in report["checks"]))
        self.assertEqual(run.call_args_list[1].args[0][1], str(self.root / "npm-cli.js"))

    def test_node_22_below_locked_vite_minimum_is_not_ready(self):
        self.web()
        responses = [{"ok": True, "stdout": "v22.0.0"}, {"ok": True, "stdout": "10.9.2"},
                     {"ok": True, "stdout": '{"three":{"loaded":true},"vite":{"loaded":true}}'}]
        with (patch.object(DOCTOR, "_node_executable", return_value=self.root / "node.exe"),
              patch.object(DOCTOR, "_npm_cli", return_value=self.root / "npm-cli.js"),
              patch.object(DOCTOR, "run_probe", side_effect=responses),
              patch.object(DOCTOR.shutil, "which", return_value=None)):
            report = DOCTOR.check_environment(self.root, scope="web")
        self.assertFalse(report["ready"])
        self.assertEqual(next(item for item in report["checks"] if item["id"] == "node.version")["status"], "fail")

    def test_missing_node_is_required_only_for_web(self):
        self.web()
        with (patch.object(DOCTOR, "_node_executable", return_value=None),
              patch.object(DOCTOR.shutil, "which", return_value=None)):
            report = DOCTOR.check_environment(self.root, scope="web")
        self.assertFalse(report["ready"])

    def test_missing_npm_or_failed_import_never_reports_ready(self):
        self.web()
        with (patch.object(DOCTOR, "_node_executable", return_value=self.root / "node.exe"),
              patch.object(DOCTOR, "_npm_cli", return_value=None),
              patch.object(DOCTOR, "run_probe", side_effect=[
                  {"ok": True, "stdout": "v22.14.0"},
                  {"ok": True, "stdout": '{"three":{"loaded":true},"vite":{"loaded":false}}'}]),
              patch.object(DOCTOR.shutil, "which", return_value=None)):
            report = DOCTOR.check_environment(self.root, scope="web")
        self.assertFalse(report["ready"])
        self.assertEqual(report["required_failure_count"], 2)

    def test_malformed_nested_web_config_fails_as_structured_report(self):
        self.web()
        path = self.root / "web/package.json"
        package = json.loads(path.read_text())
        package["engines"] = "SECRET=invalid"
        path.write_text(json.dumps(package), encoding="utf-8")
        with (patch.object(DOCTOR, "_node_executable", return_value=self.root / "node.exe"),
              patch.object(DOCTOR, "run_probe", return_value={"ok": True, "stdout": "v22.14.0"}),
              patch.object(DOCTOR.shutil, "which", return_value=None)):
            report = DOCTOR.check_environment(self.root, scope="web")
        self.assertFalse(report["ready"])
        self.assertNotIn("SECRET", json.dumps(report))

    def test_cli_refuses_existing_report_without_running_checks(self):
        path = self.root / "report.json"
        path.write_text("unchanged", encoding="utf-8")
        with (patch.object(DOCTOR.sys, "argv", ["check_environment.py", "--output", str(path)]),
              patch.object(DOCTOR.sys, "stderr", io.StringIO()),
              patch.object(DOCTOR.sys, "stdout", io.StringIO()),
              patch.object(DOCTOR, "check_environment") as check,
              self.assertRaises(SystemExit) as error):
            DOCTOR.main()
        self.assertEqual(error.exception.code, 2)
        check.assert_not_called()
        self.assertEqual(path.read_text(encoding="utf-8"), "unchanged")

    def test_subprocess_timeout_is_bounded_and_shell_never_used(self):
        with patch.object(DOCTOR.subprocess, "run", side_effect=subprocess.TimeoutExpired("x", 3)) as run:
            result = DOCTOR.run_probe(["python.exe", "-c", "pass"], timeout=3, cwd=self.root)
        self.assertEqual(result, {"ok": False, "reason": "timeout"})
        self.assertFalse(run.call_args.kwargs["shell"])
        self.assertEqual(run.call_args.kwargs["timeout"], 3)
        self.assertIsInstance(run.call_args.args[0], list)

    def test_invalid_arguments_rejected_and_report_has_no_absolute_tool_paths(self):
        for value in ("x\nsecret", "a" * 2049, ""):
            with self.assertRaises(ValueError):
                DOCTOR.safe_path(value)
        for timeout in (0, 61, True, float("nan")):
            with self.assertRaises(ValueError):
                DOCTOR.check_environment(self.root, timeout=timeout)
        with (patch.object(DOCTOR, "run_probe", return_value=self.runtime()),
              patch.object(DOCTOR.shutil, "which", return_value=None)):
            report = DOCTOR.check_environment(self.root, scope="core")
        self.assertNotIn(str(self.root).replace("\\", "\\\\"), json.dumps(report))
