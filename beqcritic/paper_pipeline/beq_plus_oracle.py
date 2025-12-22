"""
CLI: compute BEq+ oracle accuracy for a grouped candidate pool.

Oracle@pool = fraction of problems where at least one candidate in the pool
is BEq+-equivalent to the reference statement.

Optionally evaluates one or two selection files to compute selection gap.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass

from .beq_plus_eval import _load_dataset_rows, _require_lean_interact, beq_plus, _load_jsonl_map


@dataclass(frozen=True)
class _SelectionInfo:
    name: str
    choices: dict[str, str]


def _load_grouped_candidates(
    path: str,
    problem_id_key: str,
    candidates_key: str,
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if problem_id_key not in obj:
                raise ValueError(f"Missing {problem_id_key!r} in {path}: {obj}")
            pid = str(obj[problem_id_key])
            cands = obj.get(candidates_key) or []
            if not isinstance(cands, list):
                raise ValueError(f"Expected {candidates_key!r} to be a list for problem_id={pid!r}")
            out[pid] = ["" if c is None else str(c) for c in cands]
    return out


def _find_candidate_index(candidates: list[str], stmt: str) -> int | None:
    try:
        return candidates.index(stmt)
    except ValueError:
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="PAug/ProofNetVerif")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--dataset-id-key", type=str, default="id")
    p.add_argument("--dataset-ref-key", type=str, default="lean4_formalization")
    p.add_argument("--dataset-header-key", type=str, default="lean4_src_header")

    p.add_argument("--input", type=str, required=True, help="Grouped candidates JSONL")
    p.add_argument("--problem-id-key", type=str, default="problem_id")
    p.add_argument("--candidates-key", type=str, default="candidates")

    p.add_argument("--selections-a", type=str, default="")
    p.add_argument("--selections-b", type=str, default="")
    p.add_argument("--a-name", type=str, default="A")
    p.add_argument("--b-name", type=str, default="B")
    p.add_argument("--selection-problem-id-key", type=str, default="problem_id")
    p.add_argument("--selection-chosen-key", type=str, default="chosen")

    p.add_argument("--lean-version", type=str, default="v4.8.0")
    p.add_argument("--timeout-s", type=int, default=60)
    p.add_argument("--max-problems", type=int, default=0)
    p.add_argument("--shuffle-seed", type=int, default=0)
    p.add_argument("--output-jsonl", type=str, default="")
    args = p.parse_args()

    _require_lean_interact()
    from lean_interact import AutoLeanServer, LeanREPLConfig
    from lean_interact.project import TempRequireProject

    dataset_rows = _load_dataset_rows(
        dataset=str(args.dataset),
        split=str(args.split),
        id_key=str(args.dataset_id_key),
        ref_key=str(args.dataset_ref_key),
        header_key=str(args.dataset_header_key),
    )
    grouped = _load_grouped_candidates(
        path=str(args.input),
        problem_id_key=str(args.problem_id_key),
        candidates_key=str(args.candidates_key),
    )

    sel_a = None
    if str(args.selections_a).strip():
        sel_a = _SelectionInfo(
            name=str(args.a_name),
            choices=_load_jsonl_map(
                str(args.selections_a),
                key=str(args.selection_problem_id_key),
                value=str(args.selection_chosen_key),
            ),
        )
    sel_b = None
    if str(args.selections_b).strip():
        sel_b = _SelectionInfo(
            name=str(args.b_name),
            choices=_load_jsonl_map(
                str(args.selections_b),
                key=str(args.selection_problem_id_key),
                value=str(args.selection_chosen_key),
            ),
        )

    pids = set(grouped.keys()) & set(dataset_rows.keys())
    if sel_a is not None:
        pids &= set(sel_a.choices.keys())
    if sel_b is not None:
        pids &= set(sel_b.choices.keys())
    pids = sorted(pids)
    if not pids:
        raise SystemExit("No overlapping problem_ids between candidate pool, dataset split, and selections.")

    if int(args.shuffle_seed):
        rnd = random.Random(int(args.shuffle_seed))
        rnd.shuffle(pids)
    if int(args.max_problems) > 0:
        pids = pids[: int(args.max_problems)]

    proj = TempRequireProject(lean_version=str(args.lean_version), require="mathlib", verbose=True)
    cfg = LeanREPLConfig(project=proj, verbose=False)
    server = AutoLeanServer(config=cfg)

    out_path = str(args.output_jsonl).strip()
    out_f = open(out_path, "w", encoding="utf-8") if out_path else None

    oracle_hits = 0
    a_hits = 0
    b_hits = 0
    a_missing = 0
    b_missing = 0

    try:
        for idx, pid in enumerate(pids, start=1):
            row = dataset_rows[pid]
            cands = grouped[pid]

            checked: set[int] = set()
            oracle_ok = False

            a_ok = None
            b_ok = None
            a_in_pool = None
            b_in_pool = None

            def _eval_stmt(stmt: str) -> bool:
                return bool(beq_plus(row.ref, stmt, row.header, server=server, timeout_s=int(args.timeout_s)))

            # Evaluate selections first (if present), to reuse their result in the oracle.
            if sel_a is not None:
                stmt = sel_a.choices.get(pid, "")
                idx_a = _find_candidate_index(cands, stmt)
                a_in_pool = idx_a is not None
                if idx_a is not None:
                    a_ok = _eval_stmt(stmt)
                    checked.add(idx_a)
                    oracle_ok = oracle_ok or a_ok
                else:
                    a_ok = _eval_stmt(stmt)
                    a_missing += 1

            if sel_b is not None:
                stmt = sel_b.choices.get(pid, "")
                idx_b = _find_candidate_index(cands, stmt)
                b_in_pool = idx_b is not None
                if idx_b is not None and idx_b not in checked:
                    b_ok = _eval_stmt(stmt)
                    checked.add(idx_b)
                    oracle_ok = oracle_ok or b_ok
                else:
                    b_ok = _eval_stmt(stmt)
                    if idx_b is None:
                        b_missing += 1

            # Oracle: scan remaining candidates until a BEq+ hit is found.
            if not oracle_ok:
                for i, stmt in enumerate(cands):
                    if i in checked:
                        continue
                    ok = _eval_stmt(stmt)
                    if ok:
                        oracle_ok = True
                        break

            oracle_hits += int(oracle_ok)
            if a_ok is True:
                a_hits += 1
            if b_ok is True:
                b_hits += 1

            if out_f is not None:
                rec = {
                    "problem_id": pid,
                    "oracle_ok": bool(oracle_ok),
                }
                if sel_a is not None:
                    rec.update({"a_ok": bool(a_ok), "a_name": sel_a.name, "a_in_pool": bool(a_in_pool)})
                if sel_b is not None:
                    rec.update({"b_ok": bool(b_ok), "b_name": sel_b.name, "b_in_pool": bool(b_in_pool)})
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if idx % 10 == 0 or idx == len(pids):
                msg = f"[{idx}/{len(pids)}] oracle={oracle_hits / max(1, idx):.3f}"
                if sel_a is not None:
                    msg += f", {sel_a.name}={a_hits / max(1, idx):.3f}"
                if sel_b is not None:
                    msg += f", {sel_b.name}={b_hits / max(1, idx):.3f}"
                print(msg)

        total = len(pids)
        print(f"Oracle: {oracle_hits}/{total} = {oracle_hits / max(1, total):.3f}")
        if sel_a is not None:
            print(f"{sel_a.name}: {a_hits}/{total} = {a_hits / max(1, total):.3f}")
            print(f"{sel_a.name} missing from pool: {a_missing}")
            print(f"Oracle gap ({sel_a.name}): {oracle_hits - a_hits}")
        if sel_b is not None:
            print(f"{sel_b.name}: {b_hits}/{total} = {b_hits / max(1, total):.3f}")
            print(f"{sel_b.name} missing from pool: {b_missing}")
            print(f"Oracle gap ({sel_b.name}): {oracle_hits - b_hits}")
    finally:
        if out_f is not None:
            out_f.close()


if __name__ == "__main__":
    main()
