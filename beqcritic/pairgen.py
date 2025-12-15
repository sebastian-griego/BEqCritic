"""
Pair generation for training an equivalence scorer.

Two training signals are supported.

1) pred_vs_ref:
   (prediction, reference) -> label, where label is semantic correctness of the prediction.
   If the reference statement is correct, then "prediction correct" implies equivalence to reference.

2) cand_vs_cand:
   (candidate_i, candidate_j) -> label, derived from per-problem correctness labels:
     - positive: both correct for the same problem
     - negative: one correct, one incorrect for the same problem

   This creates direct supervision for candidate clustering and self-consistency selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Any
import random

@dataclass
class PairExample:
    a: str
    b: str
    label: int
    task: str
    problem_id: str | None = None

def _as_str(x: Any) -> str:
    return "" if x is None else str(x)

def make_pred_vs_ref_pairs(
    rows: Iterable[dict],
    pred_key: str,
    ref_key: str,
    label_key: str,
    problem_id_key: str | None = None,
) -> Iterator[PairExample]:
    for r in rows:
        pid = _as_str(r.get(problem_id_key)) if problem_id_key else None
        yield PairExample(
            a=_as_str(r.get(pred_key)),
            b=_as_str(r.get(ref_key)),
            label=int(r.get(label_key)),
            task="pred_vs_ref",
            problem_id=pid,
        )

def make_cand_vs_cand_pairs(
    rows: Iterable[dict],
    pred_key: str,
    label_key: str,
    problem_id_key: str,
    max_pos_per_problem: int = 16,
    max_neg_per_problem: int = 32,
    seed: int = 0,
) -> Iterator[PairExample]:
    rnd = random.Random(seed)
    by_pid: dict[str, list[dict]] = {}
    for r in rows:
        pid = _as_str(r.get(problem_id_key))
        by_pid.setdefault(pid, []).append(r)

    for pid, group in by_pid.items():
        correct = [g for g in group if int(g.get(label_key)) == 1]
        incorrect = [g for g in group if int(g.get(label_key)) == 0]

        if len(correct) >= 2:
            pairs = []
            for i in range(len(correct)):
                for j in range(i + 1, len(correct)):
                    pairs.append((correct[i], correct[j]))
            rnd.shuffle(pairs)
            for a, b in pairs[:max_pos_per_problem]:
                yield PairExample(
                    a=_as_str(a.get(pred_key)),
                    b=_as_str(b.get(pred_key)),
                    label=1,
                    task="cand_vs_cand",
                    problem_id=pid,
                )

        if len(correct) >= 1 and len(incorrect) >= 1:
            pairs = []
            for a in correct:
                for b in incorrect:
                    pairs.append((a, b))
            rnd.shuffle(pairs)
            for a, b in pairs[:max_neg_per_problem]:
                yield PairExample(
                    a=_as_str(a.get(pred_key)),
                    b=_as_str(b.get(pred_key)),
                    label=0,
                    task="cand_vs_cand",
                    problem_id=pid,
                )
