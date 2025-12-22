"""
CLI: verifier-driven selection with BEqCritic clustering on top-M candidates.

Algorithm:
  1) score all candidates with the verifier
  2) keep top-M by verifier score
  3) cluster top-M using BEqCritic similarity
  4) score clusters by an aggregator over verifier scores
  5) choose best cluster, then pick a representative within it
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Iterable

from .features import extract_features
from .hf_datasets import load_dataset_split
from .modeling import BeqCritic
from .select import ranked_components_from_scores, score_candidate_matrix
from .textnorm import normalize_lean_statement
from .verifier import NLVerifier


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _load_nl_map(dataset: str, split: str, id_key: str, nl_key: str) -> dict[str, str]:
    ds = load_dataset_split(dataset, split)
    out: dict[str, str] = {}
    for r in ds:
        pid = str(r.get(id_key))
        if pid in out:
            continue
        if nl_key not in r:
            raise ValueError(f"Missing {nl_key!r} in dataset row for id={pid!r}")
        out[pid] = "" if r[nl_key] is None else str(r[nl_key])
    return out


def _cluster_aggregate(
    scores: list[float],
    mode: str,
    top_k: int,
    temp: float,
) -> float:
    if not scores:
        return float("-inf")
    if mode == "max":
        return float(max(scores))
    s = sorted(scores, reverse=True)
    if mode == "mean_top2":
        k = min(2, len(s))
        return float(sum(s[:k]) / max(1, k))
    if mode == "mean_topk":
        k = min(max(1, int(top_k)), len(s))
        return float(sum(s[:k]) / max(1, k))
    if mode == "logsumexp":
        t = float(temp)
        if t <= 0:
            raise ValueError(f"cluster-temp must be > 0, got {t!r}")
        mx = max(s)
        acc = sum(math.exp((x - mx) / t) for x in s)
        return float(mx + t * math.log(max(1e-12, acc)))
    raise ValueError(f"Unknown cluster-score={mode!r}")


def _simplicity_penalty(
    stmt: str,
    *,
    weight_chars: float,
    weight_binders: float,
    weight_prop_assumptions: float,
    chars_scale: float,
) -> float:
    if chars_scale <= 0:
        raise ValueError(f"simple-chars-scale must be > 0, got {chars_scale!r}")
    norm = normalize_lean_statement(stmt)
    f = extract_features(norm)
    return (
        float(weight_chars) * (float(f.n_chars) / float(chars_scale))
        + float(weight_binders) * float(f.n_binders)
        + float(weight_prop_assumptions) * float(f.n_prop_assumptions)
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--verifier-model", type=str, required=True)
    p.add_argument("--beqcritic-model", type=str, required=True)
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)

    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--dataset-id-key", type=str, default="id")
    p.add_argument("--dataset-nl-key", type=str, default="nl_statement")

    p.add_argument("--problem-id-key", type=str, default="problem_id")
    p.add_argument("--candidates-key", type=str, default="candidates")
    p.add_argument("--typechecks-key", type=str, default="typechecks")

    p.add_argument("--device", type=str, default="")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--use-features", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--top-m", type=int, default=20, help="Top-M candidates by verifier score to cluster (0 = all).")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--mutual-k", type=int, default=0)
    p.add_argument("--triangle-prune-margin", type=float, default=0.0)
    p.add_argument("--cluster-mode", type=str, default="components", choices=["components", "support"])
    p.add_argument("--support-frac", type=float, default=0.7)

    p.add_argument(
        "--cluster-score",
        type=str,
        default="mean_top2",
        choices=["max", "mean_top2", "mean_topk", "logsumexp"],
    )
    p.add_argument("--cluster-score-k", type=int, default=3, help="k for mean_topk")
    p.add_argument("--cluster-temp", type=float, default=1.0, help="temperature for logsumexp")

    p.add_argument("--rep-mode", type=str, default="best", choices=["best", "simple"])
    p.add_argument("--simple-delta", type=float, default=0.1, help="Score delta for simple tie-break")
    p.add_argument("--simple-weight-chars", type=float, default=1.0)
    p.add_argument("--simple-weight-binders", type=float, default=0.5)
    p.add_argument("--simple-weight-prop-assumptions", type=float, default=0.25)
    p.add_argument("--simple-chars-scale", type=float, default=100.0)
    args = p.parse_args()

    nl_map = _load_nl_map(str(args.dataset), str(args.split), str(args.dataset_id_key), str(args.dataset_nl_key))

    verifier = NLVerifier(
        model_name_or_path=str(args.verifier_model),
        max_length=int(args.max_length),
        device=str(args.device).strip() or None,
        use_features=bool(args.use_features),
    )
    critic = BeqCritic(
        model_name_or_path=str(args.beqcritic_model),
        max_length=int(args.max_length),
        device=str(args.device).strip() or None,
    )

    with open(args.output, "w", encoding="utf-8") as fout:
        for obj in _iter_jsonl(str(args.input)):
            if args.problem_id_key not in obj:
                raise ValueError(f"Missing {args.problem_id_key!r} in input row: {obj}")
            pid = str(obj[args.problem_id_key])
            cands = obj.get(args.candidates_key) or []
            if not isinstance(cands, list):
                raise ValueError(f"Expected {args.candidates_key!r} to be a list for problem_id={pid!r}")
            if not cands:
                continue
            if pid not in nl_map:
                raise ValueError(f"Missing nl_statement for problem_id={pid!r}")
            nl = nl_map[pid]

            scores = verifier.score_pairs([nl] * len(cands), [str(c) for c in cands], batch_size=int(args.batch_size))

            typechecks = obj.get(args.typechecks_key)
            valid_idx = list(range(len(cands)))
            no_typecheck_survivors = False
            if typechecks is not None:
                if not isinstance(typechecks, list):
                    raise ValueError(f"Expected {args.typechecks_key!r} to be a list for problem_id={pid!r}")
                if len(typechecks) != len(cands):
                    raise ValueError(
                        f"Length mismatch for {pid}: candidates={len(cands)} {args.typechecks_key}={len(typechecks)}"
                    )
                mask = [bool(x) for x in typechecks]
                if any(mask):
                    valid_idx = [i for i, ok in enumerate(mask) if ok]
                else:
                    no_typecheck_survivors = True

            order = sorted(valid_idx, key=lambda i: (-scores[i], i))
            if not order:
                continue

            top_m = int(args.top_m)
            if top_m <= 0 or top_m >= len(order):
                top_idx = order
            else:
                top_idx = order[:top_m]

            top_cands = [cands[i] for i in top_idx]
            _, sim = score_candidate_matrix(top_cands, critic=critic, batch_size=8)

            comp_stats, _ = ranked_components_from_scores(
                scores=sim,
                threshold=float(args.threshold),
                component_rank="size",
                mutual_top_k=int(args.mutual_k),
                triangle_prune_margin=float(args.triangle_prune_margin),
                cluster_mode=str(args.cluster_mode),
                support_frac=float(args.support_frac),
            )

            best_comp = None
            best_score = None
            best_coh = None
            for comp, coh in comp_stats:
                comp_scores = [scores[top_idx[i]] for i in comp]
                cs = _cluster_aggregate(
                    comp_scores,
                    mode=str(args.cluster_score),
                    top_k=int(args.cluster_score_k),
                    temp=float(args.cluster_temp),
                )
                if best_score is None or cs > best_score:
                    best_score = cs
                    best_comp = comp
                    best_coh = coh
                elif cs == best_score and best_comp is not None:
                    if len(comp) > len(best_comp):
                        best_comp = comp
                        best_coh = coh

            if best_comp is None:
                continue

            comp_orig = [top_idx[i] for i in best_comp]
            comp_scores = [scores[i] for i in comp_orig]
            best_score_in_comp = max(comp_scores)

            chosen_idx = max(comp_orig, key=lambda i: scores[i])
            if str(args.rep_mode) == "simple":
                delta = float(args.simple_delta)
                cand_pool = [i for i in comp_orig if scores[i] >= best_score_in_comp - delta]
                if cand_pool:
                    chosen_idx = min(
                        cand_pool,
                        key=lambda i: (
                            _simplicity_penalty(
                                cands[i],
                                weight_chars=float(args.simple_weight_chars),
                                weight_binders=float(args.simple_weight_binders),
                                weight_prop_assumptions=float(args.simple_weight_prop_assumptions),
                                chars_scale=float(args.simple_chars_scale),
                            ),
                            -scores[i],
                            len(normalize_lean_statement(cands[i])),
                            i,
                        ),
                    )

            out = {
                "problem_id": pid,
                "chosen_index": int(chosen_idx),
                "chosen": cands[chosen_idx],
                "chosen_score": float(scores[chosen_idx]),
                "cluster_indices": [int(i) for i in comp_orig],
                "cluster_size": int(len(comp_orig)),
                "cluster_score": float(best_score) if best_score is not None else None,
                "cluster_cohesion": float(best_coh) if best_coh is not None else None,
                "top_m": int(len(top_idx)),
            }
            if no_typecheck_survivors:
                out["no_typecheck_survivors"] = True
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
