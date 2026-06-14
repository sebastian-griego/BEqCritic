#!/usr/bin/env python3
"""Run the local reproducibility gate used by CI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


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
    args = parser.parse_args()

    for command in verification_commands(args.python, include_tests=not args.skip_tests):
        print("+ " + " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
