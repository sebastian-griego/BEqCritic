"""
CLI: annotate grouped candidates with BEq+-compatible typechecking info.

This uses the same Lean wrapping logic as BEq+ evaluation (lean-interact,
dataset headers, theorem wrapper, `sorry`), then records which candidates
typecheck under the per-problem header.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from .beq_plus_eval import _load_dataset_rows, _require_lean_interact


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _stable_hash(header: str, candidate: str) -> str:
    h = hashlib.sha256()
    h.update((header + "\n\n" + candidate).encode("utf-8"))
    return h.hexdigest()


def _load_cache(path: str) -> dict[str, bool]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(obj, dict):
        return {str(k): bool(v) for k, v in obj.items()}
    return {}


def _save_cache(path: str, cache: dict[str, bool]) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _typecheck_candidate(
    stmt: str,
    header: str,
    *,
    server: "object",
    timeout_s: int,
) -> bool:
    from lean_interact import Command
    from lean_interact.interface import CommandResponse, LeanError
    from lean_interact.utils import clean_last_theorem_string

    try:
        code = header.rstrip() + "\n\n" + clean_last_theorem_string(stmt, "candidate", add_sorry=True) + "\n"
    except Exception:
        return False
    try:
        out = server.run(Command(cmd=code), timeout=timeout_s)
        if isinstance(out, LeanError):
            return False
        if not isinstance(out, CommandResponse):
            return False
        return bool(out.lean_code_is_valid(allow_sorry=True))
    except TimeoutError:
        return False
    except Exception:
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="PAug/ProofNetVerif")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--dataset-id-key", type=str, default="id")
    p.add_argument("--dataset-header-key", type=str, default="lean4_src_header")

    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--output-filtered", type=str, default="")
    p.add_argument("--summary-md", type=str, default="")
    p.add_argument("--cache-path", type=str, default="")

    p.add_argument("--problem-id-key", type=str, default="problem_id")
    p.add_argument("--candidates-key", type=str, default="candidates")
    p.add_argument("--labels-key", type=str, default="labels")

    p.add_argument("--lean-version", type=str, default="v4.8.0")
    p.add_argument("--timeout-s", type=int, default=60)
    p.add_argument("--max-problems", type=int, default=0)
    p.add_argument("--shuffle-seed", type=int, default=0)
    args = p.parse_args()

    _require_lean_interact()
    from lean_interact import AutoLeanServer, LeanREPLConfig
    from lean_interact.project import TempRequireProject

    dataset_rows = _load_dataset_rows(
        dataset=str(args.dataset),
        split=str(args.split),
        id_key=str(args.dataset_id_key),
        ref_key="lean4_formalization",
        header_key=str(args.dataset_header_key),
    )

    cache = _load_cache(str(args.cache_path))

    pids = []
    rows = []
    for obj in _iter_jsonl(str(args.input)):
        if args.problem_id_key not in obj:
            raise ValueError(f"Missing {args.problem_id_key!r} in input row: {obj}")
        pid = str(obj[args.problem_id_key])
        if pid not in dataset_rows:
            continue
        pids.append(pid)
        rows.append(obj)

    if int(args.shuffle_seed):
        rnd = random.Random(int(args.shuffle_seed))
        idxs = list(range(len(rows)))
        rnd.shuffle(idxs)
        rows = [rows[i] for i in idxs]
        pids = [pids[i] for i in idxs]
    if int(args.max_problems) > 0:
        rows = rows[: int(args.max_problems)]
        pids = pids[: int(args.max_problems)]

    proj = TempRequireProject(lean_version=str(args.lean_version), require="mathlib", verbose=True)
    cfg = LeanREPLConfig(project=proj, verbose=False)
    server = AutoLeanServer(config=cfg)

    out_path = Path(str(args.output))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = out_path.open("w", encoding="utf-8")

    filtered_f = None
    if str(args.output_filtered).strip():
        filtered_path = Path(str(args.output_filtered))
        filtered_path.parent.mkdir(parents=True, exist_ok=True)
        filtered_f = filtered_path.open("w", encoding="utf-8")

    total_candidates = 0
    total_typechecks = 0
    per_problem_counts: list[int] = []
    no_survivors = 0

    try:
        for pid, obj in zip(pids, rows):
            header = dataset_rows[pid].header
            cands = obj.get(args.candidates_key) or []
            if not isinstance(cands, list):
                raise ValueError(f"Expected {args.candidates_key!r} to be a list for problem_id={pid!r}")

            typechecks: list[bool] = []
            for cand in cands:
                cand_s = "" if cand is None else str(cand)
                key = _stable_hash(header, cand_s)
                if key in cache:
                    ok = bool(cache[key])
                else:
                    ok = _typecheck_candidate(
                        cand_s,
                        header,
                        server=server,
                        timeout_s=int(args.timeout_s),
                    )
                    cache[key] = bool(ok)
                typechecks.append(bool(ok))

            n_ok = sum(1 for x in typechecks if x)
            total_candidates += len(typechecks)
            total_typechecks += n_ok
            per_problem_counts.append(n_ok)
            if n_ok == 0:
                no_survivors += 1

            out_obj = dict(obj)
            out_obj["typechecks"] = typechecks
            out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

            if filtered_f is not None:
                if n_ok > 0:
                    keep = [i for i, ok in enumerate(typechecks) if ok]
                else:
                    keep = list(range(len(typechecks)))
                filtered = dict(obj)
                filtered["typechecks"] = [typechecks[i] for i in keep]
                if args.labels_key in filtered:
                    labels = filtered.get(args.labels_key) or []
                    if isinstance(labels, list) and len(labels) == len(typechecks):
                        filtered[args.labels_key] = [labels[i] for i in keep]
                filtered[args.candidates_key] = [cands[i] for i in keep]
                if n_ok == 0:
                    filtered["no_typecheck_survivors"] = True
                filtered_f.write(json.dumps(filtered, ensure_ascii=False) + "\n")
    finally:
        out_f.close()
        if filtered_f is not None:
            filtered_f.close()
        _save_cache(str(args.cache_path), cache)

    if args.summary_md:
        counts = Counter(per_problem_counts)
        lines = [
            "# BEq+ typecheck summary",
            "",
            f"Problems: {len(per_problem_counts)}",
            f"Candidates: {total_candidates}",
            f"Typechecks: {total_typechecks} ({(100.0 * total_typechecks / max(1, total_candidates)):.1f}%)",
            f"No typecheck survivors: {no_survivors}",
            "",
            "## Typechecks per problem",
            "",
            "| typechecking candidates | problems |",
            "|---:|---:|",
        ]
        for k in sorted(counts):
            lines.append(f"| {k} | {counts[k]} |")
        lines.append("")
        Path(args.summary_md).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
