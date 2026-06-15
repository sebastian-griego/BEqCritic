#!/usr/bin/env python3
"""Run the local reproducibility gate used by CI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]


def verification_commands(python: str, *, include_tests: bool = True) -> list[list[str]]:
    commands = [
        [python, "-m", "compileall", "-q", "beqcritic", "scripts", "tests"],
        [
            python,
            "scripts/summarize_nlverifier_paper_metrics.py",
            "--results-dir",
            "results",
            "--output-json",
            "results/nlverifier_paper_metrics.json",
            "--output-md",
            "results/nlverifier_paper_metrics.md",
            "--output-tex",
            "paper/generated/nlverifier_main_table.tex",
            "--check",
        ],
        [
            python,
            "scripts/summarize_nlverifier_paper_metrics.py",
            "--output-json",
            "results/nlverifier_paper_metrics.json",
            "--verify-source-hashes",
        ],
    ]
    if include_tests:
        commands.append([python, "-m", "pytest", "-q"])
    return commands


def run_verification(
    commands: list[list[str]],
    *,
    cwd: Path = ROOT,
    runner: Callable[..., Any] = subprocess.run,
) -> tuple[int, dict[str, Any]]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "planned_command_count": len(commands),
        "commands": [],
    }
    started = time.perf_counter()
    for command in commands:
        print("+ " + " ".join(command), flush=True)
        command_started = time.perf_counter()
        completed = runner(command, cwd=cwd, check=False)
        elapsed = time.perf_counter() - command_started
        returncode = int(getattr(completed, "returncode", 0))
        report["commands"].append(
            {
                "command": command,
                "returncode": returncode,
                "elapsed_seconds": round(elapsed, 3),
            }
        )
        if returncode != 0:
            report["status"] = "failed"
            report["failed_command"] = command
            report["executed_command_count"] = len(report["commands"])
            report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            return returncode, report

    report["executed_command_count"] = len(report["commands"])
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return 0, report


def collect_git_metadata(
    cwd: Path = ROOT,
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    def git(args: list[str]) -> Any:
        return runner(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )

    head = git(["rev-parse", "HEAD"])
    if int(getattr(head, "returncode", 1)) != 0:
        error = str(getattr(head, "stderr", "") or getattr(head, "stdout", "")).strip()
        return {"available": False, "error": error}

    branch = git(["branch", "--show-current"])
    branch_name = ""
    if int(getattr(branch, "returncode", 1)) == 0:
        branch_name = str(getattr(branch, "stdout", "")).strip()
    status = git(["status", "--porcelain"])
    if int(getattr(status, "returncode", 1)) != 0:
        error = str(
            getattr(status, "stderr", "") or getattr(status, "stdout", "")
        ).strip()
        return {
            "available": True,
            "branch": branch_name,
            "head_commit": str(getattr(head, "stdout", "")).strip(),
            "status_available": False,
            "status_error": error,
        }
    status_lines = [
        line
        for line in str(getattr(status, "stdout", "")).splitlines()
        if line.strip()
    ]
    return {
        "available": True,
        "branch": branch_name,
        "head_commit": str(getattr(head, "stdout", "")).strip(),
        "status_available": True,
        "uncommitted_change_count": len(status_lines),
        "working_tree_clean": len(status_lines) == 0,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="run compile and artifact checks without invoking pytest",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for subprocess checks",
    )
    parser.add_argument(
        "--report-json",
        default="",
        help="optional path to write a machine-readable verification report",
    )
    args = parser.parse_args()

    commands = verification_commands(args.python, include_tests=not args.skip_tests)
    returncode, report = run_verification(commands)
    report["python"] = args.python
    report["include_tests"] = not args.skip_tests
    report["git"] = collect_git_metadata()

    if args.report_json:
        report_path = Path(args.report_json)
        write_report(report_path, report)
        print(f"Wrote {report_path}")

    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
