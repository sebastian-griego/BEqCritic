"""
CLI: filter grouped candidates by Lean typechecking.

This script requires a working Lean toolchain on PATH. By default it runs `lean`
on a temporary `.lean` file containing a small header + one candidate declaration.

Input JSONL (one problem per line):
  {"problem_id": "...", "candidates": ["theorem ... := by sorry", ...]}

Output JSONL keeps only candidates that typecheck:
  {"problem_id": "...", "candidates": ["...", ...]}
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..jsonl import iter_jsonl_objects


@dataclass(frozen=True)
class _CheckResult:
    ok: bool
    stderr: str


def _lean_header(imports: list[str], extra_header: str) -> str:
    lines: list[str] = []
    for imp in imports:
        imp_s = str(imp).strip()
        if not imp_s:
            continue
        lines.append(f"import {imp_s}")
    if extra_header.strip():
        lines.append(extra_header.rstrip())
    return "\n".join(lines) + ("\n\n" if lines else "")


def _check_one(
    lean_cmd: list[str],
    lean_workdir: str | None,
    header: str,
    decl: str,
    tmp_dir: str,
    timeout_s: float,
    keep_files: bool,
) -> _CheckResult:
    fd, path = tempfile.mkstemp(prefix="beqcritic_typecheck_", suffix=".lean", dir=tmp_dir, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(header)
            f.write(decl)
            f.write("\n")
        proc = subprocess.run(
            lean_cmd + [path],
            cwd=lean_workdir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s if timeout_s > 0 else None,
        )
        ok = proc.returncode == 0
        return _CheckResult(ok=ok, stderr=str(proc.stderr or ""))
    except subprocess.TimeoutExpired:
        return _CheckResult(ok=False, stderr=f"TIMEOUT after {timeout_s}s")
    finally:
        if not keep_files:
            try:
                os.remove(path)
            except OSError:
                pass


def _iter_jsonl(path: str) -> Iterable[dict]:
    for _line_no, row in iter_jsonl_objects(path):
        yield row


def _load_input_rows(
    path: str,
    *,
    problem_id_key: str,
    candidates_key: str,
    header_key: str,
) -> list[dict]:
    rows: list[dict] = []
    for line_no, obj in iter_jsonl_objects(path):
        if problem_id_key not in obj:
            raise ValueError(f"Missing {problem_id_key!r} at {Path(path)}:{line_no}")
        pid = str(obj[problem_id_key])
        cands = obj.get(candidates_key) or []
        if not isinstance(cands, list):
            raise ValueError(
                f"Expected {candidates_key!r} to be a list at {Path(path)}:{line_no} "
                f"for problem_id={pid!r}"
            )
        if header_key and header_key not in obj:
            raise ValueError(
                f"Missing {header_key!r} at {Path(path)}:{line_no} "
                f"for problem_id={pid!r}"
            )
        rows.append(obj)
    return rows


def _open_text_output(path: str):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path.open("w", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--problem-id-key", type=str, default="problem_id")
    p.add_argument("--candidates-key", type=str, default="candidates")
    p.add_argument(
        "--lean-cmd",
        type=str,
        default="lean",
        help="Command used to invoke Lean (e.g. 'lean' or 'lake env lean').",
    )
    p.add_argument(
        "--imports",
        type=str,
        default="",
        help="Comma-separated imports to prepend (e.g. 'Mathlib').",
    )
    p.add_argument(
        "--header",
        type=str,
        default="",
        help="Extra Lean header text prepended verbatim after imports (optional).",
    )
    p.add_argument(
        "--header-key",
        type=str,
        default="",
        help="If set, also prepend per-problem header read from this JSON field (e.g. 'lean4_src_header').",
    )
    p.add_argument(
        "--lean-workdir",
        type=str,
        default="",
        help="Working directory for the Lean command (useful for `lake env lean` in a Mathlib project).",
    )
    p.add_argument("--jobs", type=int, default=4, help="Number of concurrent typechecks")
    p.add_argument("--timeout-s", type=float, default=30.0, help="Per-candidate timeout (0 = no timeout)")
    p.add_argument("--keep-temp", action="store_true", help="Keep temporary .lean files (for debugging)")
    p.add_argument("--drop-empty-problems", action="store_true", help="Drop problems with 0 passing candidates")
    p.add_argument(
        "--errors-jsonl",
        type=str,
        default="",
        help="Optional JSONL to write failing candidates + stderr for debugging.",
    )
    args = p.parse_args(argv)

    # Support multi-word lean commands like: --lean-cmd "lake env lean"
    lean_cmd = [s for s in str(args.lean_cmd).split(" ") if s]
    if not lean_cmd:
        raise ValueError("--lean-cmd must be non-empty")
    if shutil.which(lean_cmd[0]) is None:
        raise FileNotFoundError(
            f"Lean command not found: {lean_cmd[0]!r}. "
            "Install Lean or pass a working command via --lean-cmd."
        )

    imports = [s.strip() for s in str(args.imports).split(",") if s.strip()]
    global_header = _lean_header(imports=imports, extra_header=str(args.header))
    lean_workdir = str(args.lean_workdir).strip() or None
    rows = _load_input_rows(
        str(args.input),
        problem_id_key=str(args.problem_id_key),
        candidates_key=str(args.candidates_key),
        header_key=str(args.header_key),
    )

    errors_out = _open_text_output(args.errors_jsonl) if args.errors_jsonl else None
    tmp_dir_obj = None
    tmp_dir = None
    try:
        if args.keep_temp:
            tmp_dir = tempfile.mkdtemp(prefix="beqcritic_typecheck_")
            print(f"Keeping temporary .lean files under: {tmp_dir}")
        else:
            tmp_dir_obj = tempfile.TemporaryDirectory(prefix="beqcritic_typecheck_")
            tmp_dir = tmp_dir_obj.name

        with _open_text_output(args.output) as fout:
            for obj in rows:
                pid = str(obj[args.problem_id_key])
                cands = obj.get(args.candidates_key) or []
                if not cands:
                    if not args.drop_empty_problems:
                        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    continue

                header = global_header
                if args.header_key:
                    per_header = "" if obj.get(args.header_key) is None else str(obj.get(args.header_key))
                    header = per_header.rstrip() + "\n\n" + global_header

                keep_mask = [False] * len(cands)

                with ThreadPoolExecutor(max_workers=max(1, int(args.jobs))) as ex:
                    futs = {}
                    for i, decl in enumerate(cands):
                        fut = ex.submit(
                            _check_one,
                            lean_cmd,
                            lean_workdir,
                            header,
                            str(decl),
                            str(tmp_dir),
                            float(args.timeout_s),
                            bool(args.keep_temp),
                        )
                        futs[fut] = i
                    for fut in as_completed(futs):
                        i = futs[fut]
                        res = fut.result()
                        keep_mask[i] = bool(res.ok)
                        if errors_out is not None and not res.ok:
                            errors_out.write(
                                json.dumps(
                                    {
                                        "problem_id": pid,
                                        "candidate_index": int(i),
                                        "candidate": str(cands[i]),
                                        "stderr": str(res.stderr),
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )

                kept = [str(c) for c, ok in zip(cands, keep_mask) if ok]
                if not kept and args.drop_empty_problems:
                    continue
                obj[args.candidates_key] = kept
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    finally:
        if errors_out is not None:
            errors_out.close()
        if tmp_dir_obj is not None:
            tmp_dir_obj.cleanup()


if __name__ == "__main__":
    main()
