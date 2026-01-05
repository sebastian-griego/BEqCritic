"""
Analyze NLVerifier misses against BEq+ oracle.

For problems where oracle is true but NLVerifier is wrong, compute:
  - rank of the first BEq+-correct candidate under NLVerifier scores
  - score gap between top1 and best-correct
  - near-duplicate stats among top-K by BEqCritic similarity
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

from .beq_plus_eval import _require_lean_interact, beq_plus
from ..hf_datasets import load_dataset_split
from ..modeling import BeqCritic
from ..select import score_candidate_matrix
from ..verifier import NLVerifier


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _load_oracle_map(path: str, key: str = "oracle_ok") -> dict[str, bool]:
    out: dict[str, bool] = {}
    for obj in _iter_jsonl(path):
        pid = str(obj.get("problem_id"))
        out[pid] = bool(obj.get(key))
    return out


def _load_ok_map(path: str, key: str = "b_ok") -> dict[str, bool]:
    out: dict[str, bool] = {}
    for obj in _iter_jsonl(path):
        pid = str(obj.get("problem_id"))
        out[pid] = bool(obj.get(key))
    return out


def _load_candidates(path: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for obj in _iter_jsonl(path):
        pid = str(obj.get("problem_id"))
        cands = obj.get("candidates") or []
        if not isinstance(cands, list):
            raise ValueError(f"Expected candidates list for problem_id={pid!r}")
        out[pid] = ["" if c is None else str(c) for c in cands]
    return out


def _load_dataset_maps(
    dataset: str,
    split: str,
    id_key: str,
    nl_key: str,
    ref_key: str,
    header_key: str,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    ds = load_dataset_split(dataset, split)
    nl_map: dict[str, str] = {}
    ref_map: dict[str, str] = {}
    header_map: dict[str, str] = {}
    for r in ds:
        pid = str(r.get(id_key))
        if pid not in nl_map:
            nl_map[pid] = "" if r.get(nl_key) is None else str(r.get(nl_key))
            ref_map[pid] = "" if r.get(ref_key) is None else str(r.get(ref_key))
            header_map[pid] = "" if r.get(header_key) is None else str(r.get(header_key))
    return nl_map, ref_map, header_map


def _rank_buckets(ranks: list[int]) -> dict[str, int]:
    def _count_le(x: int) -> int:
        return sum(1 for r in ranks if r <= x)

    return {
        "le_1": _count_le(1),
        "le_2": _count_le(2),
        "le_5": _count_le(5),
        "le_10": _count_le(10),
        "le_20": _count_le(20),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--oracle", type=str, required=True, help="Oracle results JSONL with oracle_ok")
    p.add_argument("--verifier-results", type=str, required=True, help="BEq+ results JSONL with NLVerifier ok key")
    p.add_argument("--verifier-ok-key", type=str, default="b_ok")
    p.add_argument("--candidates", type=str, required=True, help="Grouped candidates JSONL")
    p.add_argument("--verifier-model", type=str, required=True)
    p.add_argument("--beqcritic-model", type=str, default="", help="Optional BEqCritic model for similarity stats")

    p.add_argument("--dataset", type=str, default="PAug/ProofNetVerif")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--dataset-id-key", type=str, default="id")
    p.add_argument("--dataset-nl-key", type=str, default="nl_statement")
    p.add_argument("--dataset-ref-key", type=str, default="lean4_formalization")
    p.add_argument("--dataset-header-key", type=str, default="lean4_src_header")

    p.add_argument("--top-k-sim", type=int, default=5)
    p.add_argument("--sim-threshold", type=float, default=0.5)
    p.add_argument("--timeout-s", type=int, default=60)
    p.add_argument("--lean-version", type=str, default="v4.8.0")
    p.add_argument("--max-problems", type=int, default=0)
    p.add_argument("--output-jsonl", type=str, required=True)
    p.add_argument("--summary-md", type=str, default="")
    args = p.parse_args()

    oracle_map = _load_oracle_map(str(args.oracle))
    verifier_ok_map = _load_ok_map(str(args.verifier_results), key=str(args.verifier_ok_key))
    candidates = _load_candidates(str(args.candidates))
    nl_map, ref_map, header_map = _load_dataset_maps(
        dataset=str(args.dataset),
        split=str(args.split),
        id_key=str(args.dataset_id_key),
        nl_key=str(args.dataset_nl_key),
        ref_key=str(args.dataset_ref_key),
        header_key=str(args.dataset_header_key),
    )

    target_pids = [
        pid for pid, ok in oracle_map.items()
        if ok and not verifier_ok_map.get(pid, False)
    ]
    target_pids = [pid for pid in target_pids if pid in candidates and pid in nl_map]
    if int(args.max_problems) > 0:
        target_pids = target_pids[: int(args.max_problems)]

    _require_lean_interact()
    from lean_interact import AutoLeanServer, LeanREPLConfig
    from lean_interact.project import TempRequireProject

    proj = TempRequireProject(lean_version=str(args.lean_version), require="mathlib", verbose=True)
    cfg = LeanREPLConfig(project=proj, verbose=False)
    server = AutoLeanServer(config=cfg)

    verifier = NLVerifier(
        model_name_or_path=str(args.verifier_model),
        device=None,
        use_features=True,
    )

    critic = None
    if str(args.beqcritic_model).strip():
        critic = BeqCritic(model_name_or_path=str(args.beqcritic_model), device=None)

    out_path = Path(str(args.output_jsonl))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ranks: list[int] = []
    gaps: list[float] = []
    top1_sim_counts: list[int] = []
    top1_max_sims: list[float] = []

    with out_path.open("w", encoding="utf-8") as fout:
        for idx, pid in enumerate(target_pids, start=1):
            cands = candidates[pid]
            nl = nl_map[pid]
            ref = ref_map[pid]
            header = header_map[pid]

            scores = verifier.score_pairs([nl] * len(cands), cands, batch_size=32)
            order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
            top1 = order[0]

            best_correct_rank = None
            best_correct_idx = None
            best_correct_score = None
            checks = 0
            for r, i in enumerate(order, start=1):
                ok = beq_plus(ref, cands[i], header, server=server, timeout_s=int(args.timeout_s))
                checks += 1
                if ok:
                    best_correct_rank = int(r)
                    best_correct_idx = int(i)
                    best_correct_score = float(scores[i])
                    break

            if best_correct_rank is None:
                continue

            gap = float(scores[top1]) - float(best_correct_score)
            ranks.append(int(best_correct_rank))
            gaps.append(float(gap))

            sim_stats = {}
            if critic is not None and int(args.top_k_sim) > 1:
                k = min(int(args.top_k_sim), len(order))
                top_idx = order[:k]
                top_cands = [cands[i] for i in top_idx]
                _, sim = score_candidate_matrix(top_cands, critic=critic, batch_size=8)
                top1_pos = 0
                sims_top1 = [sim[top1_pos][j] for j in range(len(top_cands)) if j != top1_pos]
                max_sim = max(sims_top1) if sims_top1 else 0.0
                count_ge = sum(1 for s in sims_top1 if s >= float(args.sim_threshold))
                top1_sim_counts.append(int(count_ge))
                top1_max_sims.append(float(max_sim))
                sim_stats = {
                    "topk": int(k),
                    "top1_sim_ge_threshold": int(count_ge),
                    "top1_max_sim": float(max_sim),
                }

            rec = {
                "problem_id": pid,
                "best_correct_rank": int(best_correct_rank),
                "best_correct_index": int(best_correct_idx),
                "best_correct_score": float(best_correct_score),
                "top1_index": int(top1),
                "top1_score": float(scores[top1]),
                "score_gap": float(gap),
                "checks_until_correct": int(checks),
            }
            rec.update(sim_stats)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if not ranks:
        print("No oracle-miss problems found.")
        return

    buckets = _rank_buckets(ranks)
    gap_mean = sum(gaps) / max(1, len(gaps))
    gap_median = sorted(gaps)[len(gaps) // 2] if gaps else 0.0

    print(f"Problems analyzed: {len(ranks)}")
    print(f"Rank <= 2: {buckets['le_2']}/{len(ranks)}")
    print(f"Rank <= 5: {buckets['le_5']}/{len(ranks)}")
    print(f"Rank <= 10: {buckets['le_10']}/{len(ranks)}")
    print(f"Mean score gap: {gap_mean:.4f}")
    print(f"Median score gap: {gap_median:.4f}")

    if args.summary_md:
        lines = [
            "# Verifier oracle-miss analysis",
            "",
            f"Problems analyzed: {len(ranks)}",
            "",
            "| rank bucket | count |",
            "|---|---:|",
            f"| <=1 | {buckets['le_1']} |",
            f"| <=2 | {buckets['le_2']} |",
            f"| <=5 | {buckets['le_5']} |",
            f"| <=10 | {buckets['le_10']} |",
            f"| <=20 | {buckets['le_20']} |",
            "",
            f"Mean score gap: {gap_mean:.4f}",
            f"Median score gap: {gap_median:.4f}",
            "",
        ]
        if top1_sim_counts:
            sim_mean = sum(top1_sim_counts) / max(1, len(top1_sim_counts))
            max_sim_mean = sum(top1_max_sims) / max(1, len(top1_max_sims))
            lines.extend(
                [
                    f"Mean top1 neighbors >= {float(args.sim_threshold):.2f}: {sim_mean:.2f}",
                    f"Mean top1 max sim: {max_sim_mean:.3f}",
                    "",
                ]
            )
        Path(str(args.summary_md)).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
