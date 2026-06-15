from types import SimpleNamespace

from scripts.verify_reproducibility import (
    collect_git_metadata,
    run_verification,
    verification_commands,
    write_report,
)


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


def test_collect_git_metadata_records_clean_worktree(tmp_path):
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((command, kwargs))
        if command[-2:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="abc123\n", stderr="")
        if command[-2:] == ["branch", "--show-current"]:
            return SimpleNamespace(returncode=0, stdout="main\n", stderr="")
        if command[-2:] == ["status", "--porcelain"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=99, stdout="", stderr="unexpected")

    metadata = collect_git_metadata(tmp_path, runner=fake_runner)

    assert metadata == {
        "available": True,
        "branch": "main",
        "head_commit": "abc123",
        "status_available": True,
        "uncommitted_change_count": 0,
        "working_tree_clean": True,
    }
    assert calls[0][1]["cwd"] == tmp_path
    assert calls[0][1]["capture_output"] is True
    assert calls[0][1]["text"] is True
    assert calls[0][1]["check"] is False


def test_collect_git_metadata_counts_dirty_worktree(tmp_path):
    def fake_runner(command, **kwargs):
        if command[-2:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="abc123\n", stderr="")
        if command[-2:] == ["branch", "--show-current"]:
            return SimpleNamespace(returncode=0, stdout="feature\n", stderr="")
        if command[-2:] == ["status", "--porcelain"]:
            return SimpleNamespace(
                returncode=0,
                stdout=" M a.py\n?? b.py\n",
                stderr="",
            )
        return SimpleNamespace(returncode=99, stdout="", stderr="unexpected")

    metadata = collect_git_metadata(tmp_path, runner=fake_runner)

    assert metadata["status_available"] is True
    assert metadata["working_tree_clean"] is False
    assert metadata["uncommitted_change_count"] == 2


def test_collect_git_metadata_reports_unavailable_status(tmp_path):
    def fake_runner(command, **kwargs):
        if command[-2:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="abc123\n", stderr="")
        if command[-2:] == ["branch", "--show-current"]:
            return SimpleNamespace(returncode=0, stdout="main\n", stderr="")
        if command[-2:] == ["status", "--porcelain"]:
            return SimpleNamespace(returncode=128, stdout="", stderr="status failed\n")
        return SimpleNamespace(returncode=99, stdout="", stderr="unexpected")

    metadata = collect_git_metadata(tmp_path, runner=fake_runner)

    assert metadata == {
        "available": True,
        "branch": "main",
        "head_commit": "abc123",
        "status_available": False,
        "status_error": "status failed",
    }


def test_collect_git_metadata_handles_unavailable_git(tmp_path):
    def fake_runner(command, **kwargs):
        return SimpleNamespace(returncode=128, stdout="", stderr="not a git repo\n")

    metadata = collect_git_metadata(tmp_path, runner=fake_runner)

    assert metadata == {"available": False, "error": "not a git repo"}


def test_write_report_uses_lf_newlines(tmp_path):
    path = tmp_path / "nested" / "report.json"

    write_report(path, {"schema_version": 1, "status": "passed"})

    report_bytes = path.read_bytes()
    assert report_bytes.endswith(b"\n")
    assert b"\r\n" not in report_bytes
