"""
CLI: temperature scaling for a BeqCritic checkpoint.

This fits a single scalar temperature T>0 on a labeled set of statement pairs, using
standard temperature scaling (minimize NLL on a held-out set).

This implementation derives (candidate_i, candidate_j) labels from a grouped candidates
JSONL file that includes per-candidate correctness labels:
  - positive: both candidates are correct for the same problem
  - negative: one correct, one incorrect for the same problem

By default, writes `temperature.json` into the checkpoint directory.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .features import extract_features
from .modeling import BeqCritic
from .textnorm import normalize_lean_statement


@dataclass(frozen=True)
class _Pair:
    a: str
    b: str
    label: int


def _load_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def _prefixed_candidates(candidates: list[str]) -> list[str]:
    norm = [normalize_lean_statement(c) for c in candidates]
    prefixes = [extract_features(s).to_prefix() for s in norm]
    return [f"{p} {s}" for p, s in zip(prefixes, norm)]


def _sample_pairs_for_problem(
    candidates: list[str],
    labels: list[int],
    rnd: random.Random,
    max_pos_pairs: int,
    max_neg_pairs: int,
    symmetrize: bool,
) -> list[_Pair]:
    if len(candidates) != len(labels):
        raise ValueError(f"Candidates/labels length mismatch: {len(candidates)} vs {len(labels)}")

    correct = [i for i, y in enumerate(labels) if int(y) == 1]
    incorrect = [i for i, y in enumerate(labels) if int(y) == 0]
    if not correct:
        return []

    pref = _prefixed_candidates(candidates)
    out: list[_Pair] = []

    if max_pos_pairs >= 0 and len(correct) >= 2:
        pos_pairs: list[tuple[int, int]] = []
        for a in range(len(correct)):
            for b in range(a + 1, len(correct)):
                pos_pairs.append((correct[a], correct[b]))
        rnd.shuffle(pos_pairs)
        if max_pos_pairs > 0:
            pos_pairs = pos_pairs[: int(max_pos_pairs)]
        for i, j in pos_pairs:
            out.append(_Pair(a=pref[i], b=pref[j], label=1))
            if symmetrize:
                out.append(_Pair(a=pref[j], b=pref[i], label=1))

    if max_neg_pairs >= 0 and incorrect:
        neg_pairs: list[tuple[int, int]] = [(i, j) for i in correct for j in incorrect]
        rnd.shuffle(neg_pairs)
        if max_neg_pairs > 0:
            neg_pairs = neg_pairs[: int(max_neg_pairs)]
        for i, j in neg_pairs:
            out.append(_Pair(a=pref[i], b=pref[j], label=0))
            if symmetrize:
                out.append(_Pair(a=pref[j], b=pref[i], label=0))

    return out


def _default_output_path(model: str) -> str | None:
    p = Path(model)
    if p.exists() and p.is_dir():
        return str(p / "temperature.json")
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="Path to a local BeqCritic checkpoint directory")
    p.add_argument("--input", type=str, required=True, help="Grouped candidates JSONL (must include `labels`)")
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="Output JSON path (default: <model>/temperature.json when --model is a local dir)",
    )
    p.add_argument("--device", type=str, default="", help="e.g. cuda:0, cuda:1, or cpu (default: auto)")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-problems", type=int, default=0, help="Limit problems read (0 = no limit)")
    p.add_argument("--max-pos-pairs-per-problem", type=int, default=64, help="0 = unlimited, -1 = none")
    p.add_argument("--max-neg-pairs-per-problem", type=int, default=64, help="0 = unlimited, -1 = none")
    p.add_argument("--max-pairs", type=int, default=200000, help="Limit total pairs used for calibration (0 = no limit)")
    p.add_argument("--symmetrize", action="store_true", help="Include both (A,B) and (B,A) for every sampled pair")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lbfgs-iters", type=int, default=50)
    args = p.parse_args()

    out_path = args.output.strip() or _default_output_path(args.model)
    if not out_path:
        raise ValueError("--output is required when --model is not a local directory")

    rnd = random.Random(int(args.seed))
    max_pos = int(args.max_pos_pairs_per_problem)
    max_neg = int(args.max_neg_pairs_per_problem)

    pairs: list[_Pair] = []
    n_seen = 0
    for obj in _load_lines(args.input):
        if args.max_problems and n_seen >= int(args.max_problems):
            break
        candidates = obj.get("candidates") or []
        labels = obj.get("labels") or []
        if not candidates:
            continue
        if not isinstance(labels, list) or len(labels) != len(candidates):
            raise ValueError(f"Missing or invalid labels for {obj.get('problem_id')}: n_cand={len(candidates)}")
        pairs.extend(
            _sample_pairs_for_problem(
                candidates=candidates,
                labels=[1 if int(x) else 0 for x in labels],
                rnd=rnd,
                max_pos_pairs=max_pos,
                max_neg_pairs=max_neg,
                symmetrize=bool(args.symmetrize),
            )
        )
        n_seen += 1

    if not pairs:
        raise SystemExit("No calibration pairs generated (need at least one correct candidate in some problems).")

    rnd.shuffle(pairs)
    if args.max_pairs and int(args.max_pairs) > 0:
        pairs = pairs[: int(args.max_pairs)]

    labels = [int(p.label) for p in pairs]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise SystemExit(f"Need both positive and negative pairs; got n_pos={n_pos} n_neg={n_neg}")

    device = args.device.strip() or None
    critic = BeqCritic(model_name_or_path=args.model, max_length=int(args.max_length), device=device)

    pair_texts = [(p.a, p.b) for p in pairs]
    # Always score at T=1.0 here; we are fitting T.
    probs = critic.score_pairs(pair_texts, batch_size=int(args.batch_size), temperature=1.0)
    if len(probs) != len(labels):
        raise RuntimeError("Internal error: score/label length mismatch")

    eps = 1e-6
    p0 = torch.tensor(probs, dtype=torch.float32).clamp(eps, 1.0 - eps)
    y = torch.tensor(labels, dtype=torch.float32)
    margins = torch.log(p0) - torch.log1p(-p0)  # logit(p)

    def _nll(m: torch.Tensor, y_: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        p_t = torch.sigmoid(m / t)
        return F.binary_cross_entropy(p_t, y_)

    with torch.no_grad():
        loss_before = float(_nll(margins, y, torch.tensor(1.0)).item())

    log_t = torch.zeros((), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=int(args.lbfgs_iters))

    def closure():
        opt.zero_grad()
        t = torch.exp(log_t)
        loss = _nll(margins, y, t)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        t_hat = float(torch.exp(log_t).clamp(1e-4, 1e4).item())
        loss_after = float(_nll(margins, y, torch.tensor(t_hat)).item())

    out = {
        "temperature": float(t_hat),
        "n_pairs": int(len(pairs)),
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "loss_before": float(loss_before),
        "loss_after": float(loss_after),
        "seed": int(args.seed),
        "symmetrize": bool(args.symmetrize),
        "max_pos_pairs_per_problem": int(max_pos),
        "max_neg_pairs_per_problem": int(max_neg),
        "max_pairs": int(args.max_pairs),
        "input": str(args.input),
        "model": str(args.model),
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"temperature={t_hat:.4f}  n_pairs={len(pairs)}  loss {loss_before:.4f} -> {loss_after:.4f}")


if __name__ == "__main__":
    main()
