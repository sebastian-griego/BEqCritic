"""
CLI: evaluate selection outputs with the BEq+ equivalence metric (LeanInteract).

This is the evaluation stage used in the paper: prove mutual implication between
the selected statement and a reference statement under the per-problem Lean header.

Notes:
  - Requires `lean-interact` (and a working `lake`/`elan` toolchain). First run
    will download/build Mathlib for the requested Lean version.
  - This tool is intentionally "paper-style": it pulls `lean4_src_header` and
    `lean4_formalization` from a HuggingFace dataset split and matches by id.

Example (compare two selector outputs on ProofNetVerif test):
  python -m beqcritic.paper_pipeline.beq_plus_eval \\
    --dataset PAug/ProofNetVerif --split test \\
    --selections-a selfbleu_selection.jsonl --a-name selfbleu \\
    --selections-b beqcritic_selection.jsonl --b-name beqcritic \\
    --lean-version v4.8.0 --timeout-s 60 --max-problems 50
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset


def _load_jsonl_map(path: str, key: str, value: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if key not in obj:
                raise ValueError(f"Missing {key!r} in {path}: {obj}")
            if value not in obj:
                raise ValueError(f"Missing {value!r} in {path}: {obj}")
            pid = str(obj[key])
            out[pid] = str(obj[value])
    return out


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        raise ValueError("Empty list")
    if q <= 0:
        return float(min(xs))
    if q >= 1:
        return float(max(xs))
    ys = sorted(float(x) for x in xs)
    i = (len(ys) - 1) * float(q)
    lo = int(i)
    hi = min(len(ys) - 1, lo + 1)
    t = i - lo
    return float((1.0 - t) * ys[lo] + t * ys[hi])


@dataclass(frozen=True)
class _DatasetRow:
    ref: str
    header: str


def _load_dataset_rows(
    dataset: str,
    split: str,
    id_key: str,
    ref_key: str,
    header_key: str,
) -> dict[str, _DatasetRow]:
    ds = load_dataset(dataset, split=split)
    out: dict[str, _DatasetRow] = {}
    for r in ds:
        if id_key not in r:
            raise ValueError(f"Missing {id_key!r} in dataset row keys: {list(r.keys())}")
        pid = str(r[id_key])
        if ref_key not in r:
            raise ValueError(f"Missing {ref_key!r} in dataset row for id={pid!r}")
        if header_key not in r:
            raise ValueError(f"Missing {header_key!r} in dataset row for id={pid!r}")
        out[pid] = _DatasetRow(ref=str(r[ref_key]), header=str(r[header_key]))
    return out


def _require_lean_interact():
    try:
        from lean_interact import AutoLeanServer, Command, LeanREPLConfig  # noqa: F401
        from lean_interact.interface import CommandResponse, LeanError, Pos, message_intersects_code  # noqa: F401
        from lean_interact.project import TempRequireProject  # noqa: F401
        from lean_interact.utils import clean_last_theorem_string, indent_code, split_conclusion  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing optional dependency `lean-interact`.\n"
            "Install with: `pip install lean-interact`"
        ) from e


def beq_plus(
    formalization_1: str,
    formalization_2: str,
    src_header: str,
    *,
    server: "object",
    timeout_s: int,
    verbose: bool = False,
) -> bool:
    """
    BEq+ equivalence check: prove both directions using a small tactic palette.

    This implementation mirrors the LeanInteract reference example (but keeps I/O minimal).
    """
    from lean_interact import Command
    from lean_interact.interface import CommandResponse, LeanError, Pos, message_intersects_code
    from lean_interact.utils import clean_last_theorem_string, indent_code, split_conclusion

    base_thm_name = "base_theorem"
    reformulated_thm_name = "reformulated_theorem"

    def extract_exact_proof(lean_output: CommandResponse, proof_start_line: int | None = None) -> str | None:
        start = Pos(line=proof_start_line, column=0) if proof_start_line else None
        for message in lean_output.messages:
            if start is not None and not message_intersects_code(message, start, None):
                continue
            if message.severity == "error":
                return None
            if message.severity == "info" and isinstance(message.data, str) and message.data.startswith("Try this:"):
                return message.data.split("Try this:", 1)[1].strip()
        return None

    def check_proof_sub(formal_code: str, formal_2_start_line: int, proof: str, *, indent_level: int = 2) -> str | None:
        prepended = "\nintros\n"
        try:
            out = server.run(Command(cmd=formal_code + indent_code(prepended + proof, indent_level)), timeout=timeout_s)
            if isinstance(out, LeanError):
                return None
            if not isinstance(out, CommandResponse):
                return None
            if proof == "sorry":
                if out.lean_code_is_valid(start_pos=Pos(line=formal_2_start_line, column=0)):
                    return proof
                return None
            if out.lean_code_is_valid(start_pos=Pos(line=formal_2_start_line, column=0), allow_sorry=False):
                if proof == "exact?":
                    return extract_exact_proof(out, proof_start_line=formal_2_start_line)
                return proof
        except TimeoutError:
            return None
        except Exception:
            return None
        return None

    def prove_all(tactics: list[str]) -> str:
        prove_independent = " ; ".join([f"(all_goals try {t})" for t in tactics])
        prove_combined = "all_goals (" + " ; ".join([f"(try {t})" for t in tactics]) + ")"
        return "all_goals intros\nfirst | (" + prove_independent + ") | (" + prove_combined + ")"

    solver_tactics_apply = ["tauto", "simp_all_arith!", "noncomm_ring", "exact?"]
    solver_tactics_have = ["tauto", "simp_all_arith!", "exact? using this"]
    proof_all_apply = prove_all(solver_tactics_apply)
    proof_all_have = prove_all(solver_tactics_have)

    res = [False, False]
    for i, (base_thm, reform_thm) in enumerate([(formalization_1, formalization_2), (formalization_2, formalization_1)]):
        if verbose:
            print(f"== Checking {'1 -> 2' if i == 0 else '2 -> 1'}")
        try:
            formal_1_code = (
                str(src_header).rstrip()
                + "\n\n"
                + clean_last_theorem_string(base_thm, base_thm_name, add_sorry=True)
                + "\n\n"
            )
            formal_2_start_line = formal_1_code.count("\n") + 1
            formal_2_code = f"{clean_last_theorem_string(reform_thm, reformulated_thm_name, add_sorry=False)} := by"
        except Exception:
            break

        formal_code = formal_1_code + formal_2_code

        # Preliminary check: the target is well-typed.
        if check_proof_sub(formal_code, formal_2_start_line, "sorry") is None:
            break

        # 1) BEqL attempt via exact?
        proof_exact = check_proof_sub(formal_code, formal_2_start_line, "exact?")
        if proof_exact and base_thm_name in proof_exact:
            res[i] = True
            continue

        # Skip if provable by assumption (can introduce false positives).
        if check_proof_sub(formal_code, formal_2_start_line, "assumption"):
            continue

        # 2) apply the base theorem directly
        proof_apply = check_proof_sub(formal_code, formal_2_start_line, f"apply {base_thm_name}\n" + proof_all_apply)
        if proof_apply:
            res[i] = True
            continue

        # 3) have: add the conclusion of the base theorem as hypothesis
        provable_without_have = False
        try:
            out_nohave = server.run(Command(cmd=formal_2_code + proof_all_have), timeout=timeout_s)
            if isinstance(out_nohave, CommandResponse):
                provable_without_have = out_nohave.lean_code_is_valid(allow_sorry=False)
        except Exception:
            provable_without_have = False

        if not provable_without_have:
            idx_conclusion = split_conclusion(formal_1_code)
            if idx_conclusion:
                idx_end_conclusion = formal_1_code.rfind(":=")
                conclusion = formal_1_code[idx_conclusion:idx_end_conclusion].strip()
                have_stmt_proof = (
                    f"have {conclusion} := by\n"
                    + indent_code(f"apply_rules [{base_thm_name}]\n" + proof_all_apply, 2)
                    + "\n"
                )
                proof_have = check_proof_sub(formal_code, formal_2_start_line, have_stmt_proof + proof_all_have)
                if proof_have:
                    res[i] = True
                    continue

        # 4) convert with tolerance
        for max_step in range(0, 5):
            proof_convert = check_proof_sub(
                formal_code,
                formal_2_start_line,
                f"convert (config := .unfoldSameFun) {base_thm_name} using {max_step}\n" + proof_all_apply,
            )
            if proof_convert:
                res[i] = True
                break

        if not res[i]:
            break

    return bool(res[0] and res[1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="PAug/ProofNetVerif")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--dataset-id-key", type=str, default="id")
    p.add_argument("--dataset-ref-key", type=str, default="lean4_formalization")
    p.add_argument("--dataset-header-key", type=str, default="lean4_src_header")

    p.add_argument("--selections-a", type=str, required=True)
    p.add_argument("--selections-b", type=str, default="")
    p.add_argument("--a-name", type=str, default="A")
    p.add_argument("--b-name", type=str, default="B")
    p.add_argument("--selection-problem-id-key", type=str, default="problem_id")
    p.add_argument("--selection-chosen-key", type=str, default="chosen")

    p.add_argument("--lean-version", type=str, default="v4.8.0", help="Lean toolchain tag (used for Mathlib pin).")
    p.add_argument("--timeout-s", type=int, default=60)
    p.add_argument("--max-problems", type=int, default=0)
    p.add_argument("--shuffle-seed", type=int, default=0, help="If non-zero, shuffle problem order for sampling.")
    p.add_argument("--bootstrap", type=int, default=0, help="If >0 and B is provided, bootstrap CI over paired diff.")
    p.add_argument("--output-jsonl", type=str, default="", help="Optional per-problem results JSONL.")
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
    sel_a = _load_jsonl_map(str(args.selections_a), key=str(args.selection_problem_id_key), value=str(args.selection_chosen_key))
    sel_b = (
        _load_jsonl_map(str(args.selections_b), key=str(args.selection_problem_id_key), value=str(args.selection_chosen_key))
        if str(args.selections_b).strip()
        else None
    )

    pids = sorted(set(sel_a.keys()) & set(dataset_rows.keys()))
    if sel_b is not None:
        pids = sorted(set(pids) & set(sel_b.keys()))
    if not pids:
        raise SystemExit("No overlapping problem_ids between selections and dataset split.")

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
    try:
        a_hits: list[int] = []
        b_hits: list[int] = []
        for idx, pid in enumerate(pids, start=1):
            row = dataset_rows[pid]
            a_stmt = sel_a[pid]
            a_ok = beq_plus(row.ref, a_stmt, row.header, server=server, timeout_s=int(args.timeout_s))
            a_hits.append(int(a_ok))

            b_ok = None
            if sel_b is not None:
                b_stmt = sel_b[pid]
                b_ok = beq_plus(row.ref, b_stmt, row.header, server=server, timeout_s=int(args.timeout_s))
                b_hits.append(int(bool(b_ok)))

            if out_f is not None:
                rec = {"problem_id": pid, "a_ok": bool(a_ok), "a_name": str(args.a_name)}
                if sel_b is not None:
                    rec.update({"b_ok": bool(b_ok), "b_name": str(args.b_name)})
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if idx % 10 == 0 or idx == len(pids):
                a_acc = sum(a_hits) / max(1, len(a_hits))
                msg = f"[{idx}/{len(pids)}] {args.a_name} acc={a_acc:.3f}"
                if sel_b is not None:
                    b_acc = sum(b_hits) / max(1, len(b_hits))
                    msg += f", {args.b_name} acc={b_acc:.3f}"
                print(msg)

        a_acc = sum(a_hits) / max(1, len(a_hits))
        print(f"{args.a_name}: {sum(a_hits)}/{len(a_hits)} = {a_acc:.3f}")
        if sel_b is not None:
            b_acc = sum(b_hits) / max(1, len(b_hits))
            print(f"{args.b_name}: {sum(b_hits)}/{len(b_hits)} = {b_acc:.3f}")

            both = sum(1 for a, b in zip(a_hits, b_hits) if a and b)
            a_only = sum(1 for a, b in zip(a_hits, b_hits) if a and not b)
            b_only = sum(1 for a, b in zip(a_hits, b_hits) if b and not a)
            neither = len(a_hits) - both - a_only - b_only
            print(f"Both correct: {both}")
            print(f"{args.a_name} only: {a_only}")
            print(f"{args.b_name} only: {b_only}")
            print(f"Neither: {neither}")

            if int(args.bootstrap) > 0:
                n = len(a_hits)
                rnd = random.Random(0)
                diffs: list[float] = []
                for _ in range(int(args.bootstrap)):
                    idxs = [rnd.randrange(n) for _ in range(n)]
                    da = sum(a_hits[i] for i in idxs) / n
                    db = sum(b_hits[i] for i in idxs) / n
                    diffs.append(db - da)
                lo = _quantile(diffs, 0.025)
                hi = _quantile(diffs, 0.975)
                print(f"Paired diff ({args.b_name}-{args.a_name}) 95% CI: [{lo:.4f}, {hi:.4f}]")
    finally:
        if out_f is not None:
            out_f.close()


if __name__ == "__main__":
    main()

