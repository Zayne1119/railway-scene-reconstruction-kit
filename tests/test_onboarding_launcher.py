"""Windows PowerShell 5.1 integration checks; no real downloads or dependency installs."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("powershell.exe")
SCRIPTS = ("Railway.ps1", "scripts/bootstrap.ps1", "scripts/onboarding_common.ps1",
           "scripts/new_project.ps1")


def literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


@unittest.skipUnless(os.name == "nt" and POWERSHELL, "Requires Windows PowerShell 5.1")
class OnboardingLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="railway onboarding PS51 ")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.root = self.directory / "Toolkit With Spaces"
        self.root.mkdir()
        for relative in SCRIPTS:
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        self.common = self.root / "scripts/onboarding_common.ps1"
        self.launcher = self.root / "Railway.ps1"

    def ps(self, body: str, timeout=15):
        header = "$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[Text.Encoding]::UTF8; "
        encoded = base64.b64encode((header + body).encode("utf-16-le")).decode("ascii")
        return subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                               "-EncodedCommand", encoded], cwd=self.root, capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=timeout,
                              shell=False, check=False,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def fixture_python(self, *, doctor_exit=0):
        configuration = Path(sys.prefix) / "pyvenv.cfg"
        if not configuration.is_file():
            self.skipTest("Fixture requires test runner's normal Python virtual environment")
        executable = self.root / ".venv/Scripts/python.exe"
        executable.parent.mkdir(parents=True)
        shutil.copyfile(sys.executable, executable)
        shutil.copyfile(configuration, self.root / ".venv/pyvenv.cfg")
        (self.root / "scripts/check_environment.py").write_text(
            f"raise SystemExit({doctor_exit})\n", encoding="utf-8")
        package = self.root / ".venv/Lib/site-packages/railway_recon"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "__main__.py").write_text(
            "import json,sys\nfrom pathlib import Path\n"
            "assert sys.argv[1] == 'init'\n"
            "target=Path(sys.argv[2]); target.mkdir(parents=True, exist_ok=False)\n"
            "(target/'arguments.json').write_text(json.dumps(sys.argv[1:]), encoding='utf-8')\n",
            encoding="utf-8")
        return executable

    def fixture_uv(self, exit_code=0):
        path = self.directory / "Mock UV With Spaces.exe"
        source = (
            'public class MockUv { public static int Main(string[] args) { '
            'if (args.Length > 0 && args[0] == "--version") { '
            'System.Console.WriteLine("uv mocked"); return 0; } '
            f'return {exit_code}; }} }}'
        )
        result = self.ps(f"Add-Type -TypeDefinition {literal(source)} -OutputAssembly {literal(path)} -OutputType ConsoleApplication")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return path

    def invoke_archive(self, *, entries=None, checksum=None, archive_name="runtime.zip",
                       checksum_filename=None, runtime_name="safe-runtime", existing=False,
                       executable_relative_path="runtime.exe", strip_directory=None,
                       extraction_python=None):
        archive = self.directory / "local fixture.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            for name, value in (entries or {"runtime.exe": b"never-executed"}).items():
                stream.writestr(name, value)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        checksum_path = self.directory / "local checksums.txt"
        checksum_path.write_text(checksum if checksum is not None else f"{digest}  {checksum_filename or archive_name}\n",
                                 encoding="ascii")
        runtime = self.root / ".runtime"
        if existing:
            (runtime / runtime_name).mkdir(parents=True)
            (runtime / runtime_name / "preserve.txt").write_text("unchanged", encoding="ascii")
        body = f"""
. {literal(self.common)}
function Invoke-WebRequest {{
  param([switch]$UseBasicParsing,[string]$Uri,[string]$OutFile,[int]$TimeoutSec)
  if ($Uri -eq 'https://fixture.invalid/checksums') {{
    Copy-Item -LiteralPath {literal(checksum_path)} -Destination $OutFile
  }} elseif ($Uri -eq 'https://fixture.invalid/archive') {{
    Copy-Item -LiteralPath {literal(archive)} -Destination $OutFile
  }} else {{ throw 'Unexpected URL; no network permitted in test.' }}
}}
try {{
  Install-RailwayPortableArchive -RuntimeRoot {literal(runtime)} -Name {literal(runtime_name)} `
    -ArchiveName {literal(archive_name)} -Url 'https://fixture.invalid/archive' `
    -ChecksumUrl 'https://fixture.invalid/checksums' -ExecutableRelativePath {literal(executable_relative_path)} `
    {('-StripDirectory ' + literal(strip_directory)) if strip_directory is not None else ''} `
    {('-ExtractionPython ' + literal(extraction_python)) if extraction_python is not None else ''} | Out-Null
}} catch {{ Write-Output $_.Exception.Message; exit 1 }}
"""
        return self.ps(body), runtime, digest

    def test_help_works_in_repository_path_with_spaces_without_python(self):
        result = self.ps(f"& {literal(self.launcher)} help; exit $LASTEXITCODE")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Railway handoff launcher", result.stdout)
        self.assertFalse((self.root / ".runtime").exists())
        self.assertFalse((self.root / ".venv").exists())

    def test_missing_python_returns_nonzero_and_never_ready(self):
        result = self.ps(f"& {literal(self.launcher)} doctor -CoreOnly; exit $LASTEXITCODE")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Project Python is missing", result.stdout)

    def test_explicit_missing_node_returns_nonzero(self):
        self.fixture_python()
        missing = self.directory / "Missing Node With Spaces"
        result = self.ps(f"& {literal(self.launcher)} doctor -NodeDirectory {literal(missing)}; exit $LASTEXITCODE")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Node not found", result.stdout)

    def test_doctor_native_exit_1_propagates_through_root_launcher(self):
        self.fixture_python(doctor_exit=1)
        result = self.ps(f"& {literal(self.launcher)} doctor -CoreOnly; exit $LASTEXITCODE")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Check environment failed (exit 1)", result.stdout)

    def test_native_argument_array_preserves_spaces_and_embedded_quotes(self):
        executable = self.fixture_python()
        capture = self.root / "capture args.py"
        result_path = self.root / "captured args.json"
        capture.write_text("import json,sys\nfrom pathlib import Path\n"
                           "Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:]),encoding='utf-8')\n",
                           encoding="utf-8")
        expected = ['Station "North Hall"', "path with spaces", "O'Brien", ""]
        arguments = ",".join(literal(item) for item in (capture, result_path, *expected))
        result = self.ps(f". {literal(self.common)}; Invoke-RailwayCommand {literal(executable)} @({arguments}) 'capture'")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result_path.read_bytes()), expected)

    def test_malicious_project_identifiers_are_rejected_before_creation(self):
        self.fixture_python()
        for identifier in ("../escape", "..\\escape", "C:\\escape", "bad;Write-Output injected", "a" * 81):
            with self.subTest(identifier=identifier):
                result = self.ps(f"& {literal(self.launcher)} new -CoreOnly -ProjectId {literal(identifier)} -Name 'Test'; exit $LASTEXITCODE")
                self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "projects").exists())

    def test_new_project_preserves_name_with_spaces_and_quotes(self):
        self.fixture_python()
        name = 'Research "North Hall"'
        result = self.ps(f"& {literal(self.launcher)} new -CoreOnly -ProjectId sample-one -Name {literal(name)}; exit $LASTEXITCODE")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        arguments = json.loads((self.root / "projects/sample-one/arguments.json").read_bytes())
        self.assertEqual(arguments[arguments.index("--name") + 1], name)

    def test_new_project_refuses_existing_directory_without_modifying_it(self):
        self.fixture_python()
        target = self.root / "projects/existing"
        target.mkdir(parents=True)
        marker = target / "preserve.txt"
        marker.write_text("unchanged", encoding="ascii")
        result = self.ps(f"& {literal(self.launcher)} new -CoreOnly -ProjectId existing -Name 'Existing'; exit $LASTEXITCODE")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(marker.read_text(), "unchanged")
        self.assertEqual([p.name for p in target.iterdir()], ["preserve.txt"])

    def test_offline_bootstrap_works_with_mock_uv_and_real_311_python_in_spaced_paths(self):
        executable = self.fixture_python()
        uv = self.fixture_uv()
        result = self.ps(f"& {literal(self.launcher)} setup -CoreOnly -Offline -UvPath {literal(uv)} -PythonPath {literal(executable)}; exit $LASTEXITCODE")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Setup completed", result.stdout)

    def test_bootstrap_restores_existing_uv_environment_even_on_failure(self):
        executable = self.fixture_python()
        uv = self.fixture_uv(exit_code=17)
        receipt = self.directory / "environment restoration.json"
        variables = ("UV_PYTHON_INSTALL_DIR", "UV_PYTHON_BIN_DIR", "UV_CACHE_DIR",
                     "UV_PROJECT_ENVIRONMENT", "UV_NO_PROGRESS")
        declaration = "; ".join(f"$env:{name}='original-{index}'" for index, name in enumerate(variables))
        fields = "; ".join(f"'{name}'=$env:{name}" for name in variables)
        result = self.ps(f"{declaration}; & {literal(self.root / 'scripts/bootstrap.ps1')} -CoreOnly -Offline "
                         f"-UvPath {literal(uv)} -PythonPath {literal(executable)}; $savedCode=$LASTEXITCODE; "
                         f"@{{ {fields} }} | ConvertTo-Json | Set-Content -LiteralPath {literal(receipt)} -Encoding UTF8; exit $savedCode")
        self.assertNotEqual(result.returncode, 0)
        restored = json.loads(receipt.read_text(encoding="utf-8-sig"))
        self.assertEqual(restored, {name: f"original-{index}" for index, name in enumerate(variables)})

    def test_sha_verified_local_mock_archive_installs_only_to_new_runtime_target(self):
        result, runtime, _ = self.invoke_archive()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((runtime / "safe-runtime/runtime.exe").read_bytes(), b"never-executed")

    def test_archive_hash_mismatch_is_rejected_before_extraction(self):
        result, runtime, _ = self.invoke_archive(checksum="0" * 64 + "  runtime.zip\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SHA-256 mismatch", result.stdout)
        self.assertFalse((runtime / "safe-runtime").exists())

    def test_zip_parent_traversal_is_rejected(self):
        result, runtime, _ = self.invoke_archive(entries={"../escaped.txt": b"unsafe", "runtime.exe": b"never-executed"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unsafe ZIP entry path", result.stdout)
        self.assertFalse((runtime / "safe-runtime").exists())
        self.assertFalse(any(runtime.rglob("escaped.txt")))

    def test_python_extraction_handles_real_zip_member_beyond_legacy_260_path(self):
        executable = self.fixture_python()
        member = "/".join(f"nested_{index}_" + "x" * 35 for index in range(6)) + "/payload.txt"
        fixture_runtime = (self.root / ".runtime").resolve()

        def clean_long_fixture_runtime():
            # Explicitly bound, test-created subtree only; the general temporary
            # directory cleaner may itself use legacy unprefixed Windows paths.
            self.assertTrue(fixture_runtime.is_relative_to(self.root.resolve()))
            self.assertEqual(fixture_runtime.name, ".runtime")
            if fixture_runtime.exists():
                shutil.rmtree("\\\\?\\" + str(fixture_runtime))

        self.addCleanup(clean_long_fixture_runtime)
        result, runtime, _ = self.invoke_archive(
            entries={"runtime.exe": b"never-executed", member: b"long-path-content"},
            extraction_python=executable,
        )
        destination = runtime / "safe-runtime" / member
        self.assertGreater(len(str(destination)), 260)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Extract verified runtime archive", result.stdout)
        self.assertEqual(Path("\\\\?\\" + str(destination)).read_bytes(), b"long-path-content")

    def test_zip_traversal_rejected_before_optional_python_extractor_runs(self):
        executable = self.fixture_python()
        result, runtime, _ = self.invoke_archive(
            entries={"../escaped.txt": b"unsafe", "runtime.exe": b"never-executed"},
            extraction_python=executable,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unsafe ZIP entry path", result.stdout)
        self.assertNotIn("Extract verified runtime archive", result.stdout)
        self.assertFalse((runtime / "safe-runtime").exists())
        self.assertFalse(any(runtime.rglob("escaped.txt")))

    def test_existing_runtime_directory_is_not_overwritten(self):
        result, runtime, _ = self.invoke_archive(existing=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((runtime / "safe-runtime/preserve.txt").read_text(), "unchanged")
        self.assertFalse((runtime / "downloads").exists())

    def test_checksum_filename_requires_exact_match_not_substring(self):
        result, runtime, _ = self.invoke_archive(checksum_filename="runtime.zip.unrelated")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((runtime / "safe-runtime").exists())

    def test_archive_name_parent_absolute_and_stream_paths_rejected_before_staging(self):
        for name in ("../escaped.zip", "C:\\escaped.zip", "runtime.zip:payload"):
            with self.subTest(name=name):
                result, runtime, _ = self.invoke_archive(archive_name=name)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("single filenames", result.stdout)
                self.assertFalse(runtime.exists())

    def test_runtime_name_must_be_a_single_basename(self):
        for name in ("../outside", "folder/name", "runtime:stream"):
            with self.subTest(name=name):
                result, runtime, _ = self.invoke_archive(runtime_name=name)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("single filenames", result.stdout)
                self.assertFalse(runtime.exists())

    def test_executable_subpath_rejects_parent_absolute_and_stream_paths(self):
        for name in ("..\\outside.exe", "C:\\outside.exe", "runtime.exe:stream"):
            with self.subTest(name=name):
                result, runtime, _ = self.invoke_archive(executable_relative_path=name)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("inside the extraction directory", result.stdout)
                self.assertFalse(runtime.exists())

    def test_strip_directory_rejects_parent_absolute_and_stream_paths(self):
        for name in ("..\\..", "C:\\outside", "data:stream"):
            with self.subTest(name=name):
                result, runtime, _ = self.invoke_archive(strip_directory=name)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("inside the extraction directory", result.stdout)
                self.assertFalse(runtime.exists())
