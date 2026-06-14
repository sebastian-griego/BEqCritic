from types import SimpleNamespace

from scripts.verify_reproducibility import run_verification, verification_commands


def test_verification_commands_cover_reproducibility_gate():
    commands = verification_commands("python")

    assert commands[0] == ["python", "-m", "compileall", "-q", "beqcritic", "scripts", "tests"]
    assert any("--check" in command for command in commands)
    assert any("--verify-source-hashes" in command for command in commands)
    assert commands[-1] == ["python", "-m", "pytest", "-q"]


def test_verification_commands_can_skip_pytest():
    commands = verification_commands("python", include_tests=False)

    assert ["python", "-m", "pytest", "-q"] not in commands


def test_run_verification_records_successful_commands(tmp_path):
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    returncode, report = run_verification(
        [["python", "--version"], ["python", "-m", "pytest", "-q"]],
        cwd=tmp_path,
        runner=fake_runner,
    )

    assert returncode == 0
    assert report["schema_version"] == 1
    assert report["status"] == "passed"
    assert report["planned_command_count"] == 2
    assert report["executed_command_count"] == 2
    assert [row["returncode"] for row in report["commands"]] == [0, 0]
    assert calls[0][1]["cwd"] == tmp_path
    assert calls[0][1]["check"] is False


def test_run_verification_stops_and_reports_first_failure(tmp_path):
    def fake_runner(command, **kwargs):
        return SimpleNamespace(returncode=2 if command[-1] == "bad" else 0)

    returncode, report = run_verification(
        [["python", "ok"], ["python", "bad"], ["python", "skipped"]],
        cwd=tmp_path,
        runner=fake_runner,
    )

    assert returncode == 2
    assert report["schema_version"] == 1
    assert report["status"] == "failed"
    assert report["planned_command_count"] == 3
    assert report["executed_command_count"] == 2
    assert report["failed_command"] == ["python", "bad"]
    assert [row["command"] for row in report["commands"]] == [["python", "ok"], ["python", "bad"]]
