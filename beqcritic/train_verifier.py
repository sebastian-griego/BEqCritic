"""
Train NLVerifier, a reference-free NL->Lean reranker: score (nl_statement, lean_statement) pairs.

Supports:
  - Pairwise ranking loss per problem id (score(pos) > score(neg))
  - Listwise softmax loss per problem id (optimize top-1 selection)
  - Optional hard-negative mining rounds
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from .bleu import sym_bleu
from .features import extract_features
from .hf_datasets import load_dataset_split
from .textnorm import normalize_lean_statement, normalize_whitespace

REPO_ROOT = Path(__file__).resolve().parents[1]


def _split_rows_by_problem_id(
    rows: list[dict],
    problem_id_key: str | None,
    eval_size: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    if not rows:
        return [], []
    if not problem_id_key:
        return _split_rows_random(rows, eval_size, seed)

    missing = [i for i, r in enumerate(rows) if problem_id_key not in r]
    if missing:
        raise ValueError(
            f"Cannot split by problem id: {problem_id_key!r} missing from {len(missing)} rows. "
            f"Pass the correct --problem-id-key."
        )

    pids = sorted({str(r.get(problem_id_key)) for r in rows})
    if len(pids) < 2:
        return _split_rows_random(rows, eval_size, seed)

    rnd = random.Random(seed)
    rnd.shuffle(pids)
    n_eval = int(len(pids) * float(eval_size))
    n_eval = max(1, n_eval)
    n_eval = min(n_eval, len(pids) - 1)
    eval_pids = set(pids[:n_eval])

    eval_rows = [r for r in rows if str(r.get(problem_id_key)) in eval_pids]
    train_rows = [r for r in rows if str(r.get(problem_id_key)) not in eval_pids]
    return train_rows, eval_rows


def _split_rows_random(rows: list[dict], eval_size: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not rows:
        return [], []
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    n_eval = int(len(rows) * float(eval_size))
    n_eval = max(1, n_eval)
    n_eval = min(n_eval, len(rows) - 1) if len(rows) > 1 else 1
    eval_rows = [rows[i] for i in idx[:n_eval]]
    train_rows = [rows[i] for i in idx[n_eval:]]
    return train_rows, eval_rows


def _resolve_base_model(name_or_path: str) -> str:
    p = Path(name_or_path)
    if p.exists():
        return name_or_path

    if not p.is_absolute():
        from_repo = REPO_ROOT / p
        if from_repo.exists():
            return str(from_repo)

    if "/" in name_or_path:
        local = REPO_ROOT / "hf_models" / name_or_path.replace("/", "--")
        if local.exists():
            return str(local)

    return name_or_path


def _as_label(x: Any) -> int:
    if isinstance(x, bool):
        x = int(x)
    elif isinstance(x, str):
        x = x.strip()
    try:
        x_int = int(x)
    except Exception as exc:
        raise ValueError(f"label must be 0/1, got {x!r}") from exc
    if x_int not in (0, 1):
        raise ValueError(f"label must be 0/1, got {x_int!r}")
    return int(x_int)


def _rows_from_hf(
    ds: "object",
    *,
    max_rows: int = 0,
    max_problems: int = 0,
    problem_id_key: str | None = None,
) -> list[dict]:
    rows: list[dict] = []
    allowed_pids: set[str] = set()
    for r in ds:
        if max_rows and len(rows) >= int(max_rows):
            break

        if max_problems and problem_id_key:
            pid = str(r.get(problem_id_key))
            if pid not in allowed_pids:
                if len(allowed_pids) >= int(max_problems):
                    continue
                allowed_pids.add(pid)

        rows.append(dict(r))
    return rows


def _prepare_text_pairs(
    nl_list: list[str],
    lean_list: list[str],
    use_features: bool,
) -> tuple[list[str], list[str]]:
    nl_clean = [normalize_whitespace(x) for x in nl_list]
    lean_clean = [normalize_lean_statement(x) for x in lean_list]
    if use_features:
        lean_clean = [f"{extract_features(s).to_prefix()} {s}" for s in lean_clean]
    return nl_clean, lean_clean


def _maybe_sample_indices(indices: list[int], max_count: int, rnd: random.Random) -> list[int]:
    if max_count <= 0 or len(indices) <= max_count:
        return list(indices)
    return rnd.sample(indices, int(max_count))


@dataclass(frozen=True)
class PairwiseExample:
    nl: str
    pos: str
    neg: str
    problem_id: str | None = None


@dataclass(frozen=True)
class ListwiseExample:
    nl: str
    candidates: list[str]
    labels: list[int]
    problem_id: str | None = None
    sampled_indices: list[int] | None = None


@dataclass(frozen=True)
class ProblemGroup:
    problem_id: str
    nl: str
    candidates: list[str]
    labels: list[int]


@dataclass
class ListwiseBuildStats:
    groups_total: int = 0
    groups_used: int = 0
    skipped_no_pos: int = 0
    skipped_no_neg: int = 0
    skipped_invalid: int = 0
    raw_candidate_counts: list[int] = field(default_factory=list)
    pos_counts: list[int] = field(default_factory=list)
    neg_counts: list[int] = field(default_factory=list)
    label_hist: Counter = field(default_factory=Counter)
    sampled_pos_fracs: list[float] = field(default_factory=list)


@dataclass
class MiningStats:
    groups_total: int = 0
    top1_neg: int = 0
    pos_score_sum: float = 0.0
    pos_score_count: int = 0
    hard_score_sum: float = 0.0
    hard_score_count: int = 0


def _build_pairwise_examples(rows: list[dict], args) -> list[PairwiseExample]:
    for r in rows:
        r[args.label_key] = _as_label(r.get(args.label_key))

    if not args.problem_id_key:
        raise ValueError("--problem-id-key is required for pairwise ranking")

    rnd = random.Random(args.seed)
    by_pid: dict[str, list[dict]] = {}
    for r in rows:
        pid = str(r.get(args.problem_id_key))
        by_pid.setdefault(pid, []).append(r)

    examples: list[PairwiseExample] = []
    for pid, group in by_pid.items():
        nl = ""
        for r in group:
            nl_val = r.get(args.nl_key)
            if nl_val:
                nl = str(nl_val)
                break
        if not nl:
            continue

        pos = [g for g in group if int(g.get(args.label_key)) == 1]
        neg = [g for g in group if int(g.get(args.label_key)) == 0]
        if not pos or not neg:
            continue

        pairs: list[tuple[dict, dict]] = [(p, n) for p in pos for n in neg]
        if args.neg_sampling == "random":
            rnd.shuffle(pairs)
        elif args.neg_sampling == "hard":
            scored = []
            for p, n in pairs:
                sp = "" if p.get(args.pred_key) is None else str(p.get(args.pred_key))
                sn = "" if n.get(args.pred_key) is None else str(n.get(args.pred_key))
                scored.append((sym_bleu(sp, sn), rnd.random(), p, n))
            # Hard negatives: high BLEU similarity to the positive.
            scored.sort(key=lambda t: (-t[0], t[1]))
            pairs = [(p, n) for _, _, p, n in scored]
        else:
            raise ValueError(f"Unknown neg_sampling={args.neg_sampling!r}")

        if args.max_pairs_per_problem > 0:
            pairs = pairs[: int(args.max_pairs_per_problem)]

        for p, n in pairs:
            p_stmt = "" if p.get(args.pred_key) is None else str(p.get(args.pred_key))
            n_stmt = "" if n.get(args.pred_key) is None else str(n.get(args.pred_key))
            examples.append(PairwiseExample(nl=nl, pos=p_stmt, neg=n_stmt, problem_id=pid))

    rnd.shuffle(examples)
    return examples


def _build_groups(rows: list[dict], args) -> list[ProblemGroup]:
    for r in rows:
        r[args.label_key] = _as_label(r.get(args.label_key))

    if not args.problem_id_key:
        raise ValueError("--problem-id-key is required for listwise ranking")

    grouped: dict[str, dict[str, Any]] = {}
    for r in rows:
        pid = str(r.get(args.problem_id_key))
        if pid not in grouped:
            grouped[pid] = {"nl": "", "candidates": [], "labels": []}
        entry = grouped[pid]
        if not entry["nl"]:
            nl_val = r.get(args.nl_key)
            if nl_val:
                entry["nl"] = str(nl_val)
        cand = "" if r.get(args.pred_key) is None else str(r.get(args.pred_key))
        entry["candidates"].append(cand)
        entry["labels"].append(int(r.get(args.label_key)))

    groups: list[ProblemGroup] = []
    for pid, entry in grouped.items():
        nl = entry["nl"]
        if not nl:
            continue
        groups.append(
            ProblemGroup(
                problem_id=pid,
                nl=nl,
                candidates=entry["candidates"],
                labels=entry["labels"],
            )
        )
    return groups


def _safe_mean(values: list[int | float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _safe_median(values: list[int | float]) -> float:
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _update_listwise_stats(
    stats: ListwiseBuildStats,
    raw_count: int,
    pos_count: int,
    neg_count: int,
    labels: list[int],
) -> None:
    stats.groups_used += 1
    stats.raw_candidate_counts.append(int(raw_count))
    stats.pos_counts.append(int(pos_count))
    stats.neg_counts.append(int(neg_count))
    stats.label_hist.update(labels)
    total = pos_count + neg_count
    stats.sampled_pos_fracs.append(float(pos_count) / float(total) if total else 0.0)


def _validate_listwise_stats(
    stats: ListwiseBuildStats,
    label: str,
    *,
    max_no_pos_rate: float,
    max_no_neg_rate: float,
) -> None:
    total = max(1, stats.groups_total)
    no_pos_rate = stats.skipped_no_pos / total
    no_neg_rate = stats.skipped_no_neg / total
    if no_pos_rate > float(max_no_pos_rate):
        raise ValueError(
            f"{label}: skipped_no_pos rate {no_pos_rate:.1%} exceeds {float(max_no_pos_rate):.0%}"
        )
    if no_neg_rate > float(max_no_neg_rate):
        raise ValueError(
            f"{label}: skipped_no_neg rate {no_neg_rate:.1%} exceeds {float(max_no_neg_rate):.0%}"
        )
    if stats.groups_used == 0:
        raise ValueError(f"{label}: no usable groups after filtering")


def _print_listwise_stats(stats: ListwiseBuildStats, label: str) -> None:
    total_labels = sum(stats.label_hist.values())
    total_pos = int(stats.label_hist.get(1, 0))
    total_neg = int(stats.label_hist.get(0, 0))
    pos_frac = total_pos / max(1, total_labels)
    print(
        f"{label} groups: seen={stats.groups_total}, used={stats.groups_used}, "
        f"skipped_no_pos={stats.skipped_no_pos}, skipped_no_neg={stats.skipped_no_neg}, "
        f"skipped_invalid={stats.skipped_invalid}"
    )
    print(
        f"{label} mean/median pos per group: "
        f"{_safe_mean(stats.pos_counts):.2f}/{_safe_median(stats.pos_counts):.2f}, "
        f"neg per group: {_safe_mean(stats.neg_counts):.2f}/{_safe_median(stats.neg_counts):.2f}"
    )
    print(
        f"{label} mean candidates before sampling: {_safe_mean(stats.raw_candidate_counts):.2f}"
    )
    print(
        f"{label} label_hist: {{0: {total_neg}, 1: {total_pos}}}, "
        f"sampled_pos_frac mean/median: "
        f"{_safe_mean(stats.sampled_pos_fracs):.3f}/{_safe_median(stats.sampled_pos_fracs):.3f}, "
        f"pos_frac_overall: {pos_frac:.3f}"
    )


def _update_mining_stats(
    stats: MiningStats,
    labels: list[int],
    scores: list[float],
    pos_idx: list[int],
    hard_idx: list[int],
) -> None:
    stats.groups_total += 1
    if scores:
        top_idx = max(range(len(scores)), key=lambda i: scores[i])
        if int(labels[top_idx]) == 0:
            stats.top1_neg += 1
    for i in pos_idx:
        stats.pos_score_sum += float(scores[i])
        stats.pos_score_count += 1
    for i in hard_idx:
        stats.hard_score_sum += float(scores[i])
        stats.hard_score_count += 1


def _print_mining_stats(stats: MiningStats, label: str) -> None:
    pos_mean = stats.pos_score_sum / max(1, stats.pos_score_count)
    hard_mean = stats.hard_score_sum / max(1, stats.hard_score_count)
    top1_neg_frac = stats.top1_neg / max(1, stats.groups_total)
    print(
        f"{label} mining: groups={stats.groups_total}, "
        f"top1_neg_frac={top1_neg_frac:.3f}, "
        f"pos_score_mean={pos_mean:.3f}, hard_neg_score_mean={hard_mean:.3f}"
    )


def _compute_listwise_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    offset = 0
    sizes = [int(x) for x in group_sizes.detach().cpu().tolist()]
    for size in sizes:
        if size <= 0:
            continue
        group_logits = logits[offset : offset + size]
        group_labels = labels[offset : offset + size]
        offset += size
        pos_mask = group_labels > 0.5
        if not bool(pos_mask.any()):
            continue
        target = pos_mask.float() / pos_mask.float().sum()
        logp = (group_logits / float(tau)).log_softmax(0)
        losses.append(-(target * logp).sum())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def _build_listwise_examples(
    groups: list[ProblemGroup],
    *,
    rnd: random.Random,
    negatives_per_group: int,
    max_positives: int,
) -> tuple[list[ListwiseExample], ListwiseBuildStats]:
    examples: list[ListwiseExample] = []
    stats = ListwiseBuildStats()
    for g in groups:
        stats.groups_total += 1
        if len(g.candidates) != len(g.labels):
            stats.skipped_invalid += 1
            continue
        pos_idx = [i for i, lab in enumerate(g.labels) if int(lab) == 1]
        neg_idx = [i for i, lab in enumerate(g.labels) if int(lab) == 0]
        if not pos_idx or not neg_idx:
            if not pos_idx:
                stats.skipped_no_pos += 1
            if not neg_idx:
                stats.skipped_no_neg += 1
            continue
        if set(pos_idx) & set(neg_idx):
            stats.skipped_invalid += 1
            continue

        pos_idx = _maybe_sample_indices(pos_idx, max_positives, rnd)
        neg_idx = _maybe_sample_indices(neg_idx, negatives_per_group, rnd)
        if not pos_idx or not neg_idx:
            stats.skipped_invalid += 1
            continue
        if any(i < 0 or i >= len(g.candidates) for i in pos_idx + neg_idx):
            stats.skipped_invalid += 1
            continue

        pairs: list[tuple[str, int, int]] = []
        pairs.extend([(g.candidates[i], 1, i) for i in pos_idx])
        pairs.extend([(g.candidates[i], 0, i) for i in neg_idx])
        rnd.shuffle(pairs)

        if not pairs:
            continue
        candidates, labels, indices = zip(*pairs)
        labels_list = [int(x) for x in labels]
        _update_listwise_stats(stats, len(g.candidates), len(pos_idx), len(neg_idx), labels_list)
        examples.append(
            ListwiseExample(
                nl=g.nl,
                candidates=list(candidates),
                labels=labels_list,
                problem_id=g.problem_id,
                sampled_indices=list(indices),
            )
        )
    return examples, stats


@torch.inference_mode()
def _score_pairs(
    model,
    tokenizer,
    nl_list: list[str],
    lean_list: list[str],
    *,
    max_length: int,
    use_features: bool,
    batch_size: int,
) -> list[float]:
    if len(nl_list) != len(lean_list):
        raise ValueError(f"Length mismatch: nl={len(nl_list)} lean={len(lean_list)}")
    device = next(model.parameters()).device
    scores: list[float] = []
    for i in range(0, len(nl_list), batch_size):
        chunk_nl = nl_list[i : i + batch_size]
        chunk_lean = lean_list[i : i + batch_size]
        nl_clean, lean_clean = _prepare_text_pairs(chunk_nl, chunk_lean, use_features)
        enc = tokenizer(
            nl_clean,
            lean_clean,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        ).to(device)
        logits = model(**enc).logits
        if logits.dim() == 2 and logits.size(-1) == 1:
            vals = logits.squeeze(-1)
        elif logits.dim() == 2 and logits.size(-1) >= 2:
            vals = logits[:, 1]
        else:
            vals = logits.view(-1)
        scores.extend(vals.detach().cpu().tolist())
    return scores


def _score_grouped_candidates(
    groups: list[ProblemGroup],
    model,
    tokenizer,
    *,
    max_length: int,
    use_features: bool,
    batch_size: int,
) -> list[list[float]]:
    if not groups:
        return []
    flat_nl: list[str] = []
    flat_cands: list[str] = []
    sizes: list[int] = []
    for g in groups:
        sizes.append(len(g.candidates))
        flat_nl.extend([g.nl] * len(g.candidates))
        flat_cands.extend(g.candidates)
    scores = _score_pairs(
        model,
        tokenizer,
        flat_nl,
        flat_cands,
        max_length=max_length,
        use_features=use_features,
        batch_size=batch_size,
    )
    grouped_scores: list[list[float]] = []
    offset = 0
    for n in sizes:
        grouped_scores.append(scores[offset : offset + n])
        offset += n
    return grouped_scores


def _build_mined_listwise_examples(
    groups: list[ProblemGroup],
    grouped_scores: list[list[float]],
    args,
    *,
    rnd: random.Random,
) -> tuple[list[ListwiseExample], ListwiseBuildStats, MiningStats]:
    examples: list[ListwiseExample] = []
    stats = ListwiseBuildStats()
    mining_stats = MiningStats()
    for g, scores in zip(groups, grouped_scores):
        stats.groups_total += 1
        if len(g.candidates) != len(g.labels):
            stats.skipped_invalid += 1
            continue
        pos_idx = [i for i, lab in enumerate(g.labels) if int(lab) == 1]
        neg_idx = [i for i, lab in enumerate(g.labels) if int(lab) == 0]
        if not pos_idx or not neg_idx:
            if not pos_idx:
                stats.skipped_no_pos += 1
            if not neg_idx:
                stats.skipped_no_neg += 1
            continue
        if set(pos_idx) & set(neg_idx):
            stats.skipped_invalid += 1
            continue

        pos_idx = _maybe_sample_indices(pos_idx, int(args.listwise_max_positives), rnd)
        if not pos_idx:
            stats.skipped_invalid += 1
            continue
        if any(i < 0 or i >= len(g.candidates) for i in pos_idx):
            stats.skipped_invalid += 1
            continue
        if any(int(g.labels[i]) != 1 for i in pos_idx):
            raise ValueError(f"Mining positives include non-positive labels for problem_id={g.problem_id}")

        hard_sorted = sorted(neg_idx, key=lambda i: (-scores[i], i))
        hard = hard_sorted[: int(args.mining_hard_k)]
        remaining = [i for i in neg_idx if i not in hard]
        rnd.shuffle(remaining)
        easy = remaining[: int(args.mining_easy_k)]
        if any(int(g.labels[i]) != 0 for i in hard + easy):
            raise ValueError(f"Mining negatives include positive labels for problem_id={g.problem_id}")

        neg_pool = list(dict.fromkeys(hard + easy))
        max_negs = int(args.listwise_negatives)
        if max_negs > 0:
            if len(neg_pool) < max_negs:
                remaining = [i for i in neg_idx if i not in neg_pool]
                rnd.shuffle(remaining)
                neg_pool.extend(remaining[: max_negs - len(neg_pool)])
            else:
                neg_pool = neg_pool[:max_negs]

        if not neg_pool:
            continue

        pairs: list[tuple[str, int, int]] = []
        pairs.extend([(g.candidates[i], 1, i) for i in pos_idx])
        pairs.extend([(g.candidates[i], 0, i) for i in neg_pool])
        rnd.shuffle(pairs)
        if not pairs:
            continue
        candidates, labels, indices = zip(*pairs)
        labels_list = [int(x) for x in labels]
        _update_listwise_stats(stats, len(g.candidates), len(pos_idx), len(neg_pool), labels_list)
        _update_mining_stats(mining_stats, g.labels, scores, pos_idx, hard)
        examples.append(
            ListwiseExample(
                nl=g.nl,
                candidates=list(candidates),
                labels=labels_list,
                problem_id=g.problem_id,
                sampled_indices=list(indices),
            )
        )
    return examples, stats, mining_stats


def _build_mined_pairwise_examples(
    groups: list[ProblemGroup],
    grouped_scores: list[list[float]],
    args,
    *,
    rnd: random.Random,
) -> tuple[list[PairwiseExample], MiningStats]:
    examples: list[PairwiseExample] = []
    mining_stats = MiningStats()
    for g, scores in zip(groups, grouped_scores):
        if len(g.candidates) != len(g.labels):
            continue
        pos_idx = [i for i, lab in enumerate(g.labels) if int(lab) == 1]
        neg_idx = [i for i, lab in enumerate(g.labels) if int(lab) == 0]
        if not pos_idx or not neg_idx:
            continue

        hard_sorted = sorted(neg_idx, key=lambda i: (-scores[i], i))
        hard = hard_sorted[: int(args.mining_hard_k)]
        remaining = [i for i in neg_idx if i not in hard]
        rnd.shuffle(remaining)
        easy = remaining[: int(args.mining_easy_k)]
        if any(int(g.labels[i]) != 0 for i in hard + easy):
            raise ValueError(f"Mining negatives include positive labels for problem_id={g.problem_id}")
        neg_pool = list(dict.fromkeys(hard + easy))
        if not neg_pool:
            continue

        pairs: list[tuple[int, int]] = [(p, n) for p in pos_idx for n in neg_pool]
        rnd.shuffle(pairs)
        if int(args.max_pairs_per_problem) > 0:
            pairs = pairs[: int(args.max_pairs_per_problem)]

        for p_idx, n_idx in pairs:
            if int(g.labels[p_idx]) != 1:
                raise ValueError(f"Mining positives include non-positive labels for problem_id={g.problem_id}")
            examples.append(
                PairwiseExample(
                    nl=g.nl,
                    pos=g.candidates[p_idx],
                    neg=g.candidates[n_idx],
                    problem_id=g.problem_id,
                )
            )
        _update_mining_stats(mining_stats, g.labels, scores, pos_idx, hard)
    rnd.shuffle(examples)
    return examples, mining_stats


def _to_hf_dataset(examples: list[PairwiseExample]) -> Dataset:
    return Dataset.from_dict(
        {
            "nl": [e.nl for e in examples],
            "pos": [e.pos for e in examples],
            "neg": [e.neg for e in examples],
            "problem_id": [e.problem_id for e in examples],
        }
    )


def _to_listwise_dataset(examples: list[ListwiseExample]) -> Dataset:
    data = {
        "nl": [e.nl for e in examples],
        "candidates": [list(e.candidates) for e in examples],
        "labels": [list(e.labels) for e in examples],
        "problem_id": [e.problem_id for e in examples],
    }
    if any(e.sampled_indices is not None for e in examples):
        data["sampled_indices"] = [e.sampled_indices for e in examples]
    return Dataset.from_dict(data)


class PairwiseRankingTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pos_inputs = {k[:-4]: v for k, v in inputs.items() if k.endswith("_pos")}
        neg_inputs = {k[:-4]: v for k, v in inputs.items() if k.endswith("_neg")}
        pos_logits = model(**pos_inputs).logits.squeeze(-1)
        neg_logits = model(**neg_inputs).logits.squeeze(-1)
        diff = pos_logits - neg_logits
        loss = F.binary_cross_entropy_with_logits(diff, torch.ones_like(diff))
        if return_outputs:
            return loss, {"diff": diff}
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        if prediction_loss_only:
            loss = self.compute_loss(model, inputs)
            return (loss.detach(), None, None)
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        diff = outputs["diff"].detach()
        labels = torch.ones_like(diff)
        return loss.detach(), diff, labels


class ListwiseRankingTrainer(Trainer):
    def __init__(self, *args, tau: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = float(tau)

    @staticmethod
    def _iter_groups(group_sizes: torch.Tensor) -> list[int]:
        return [int(x) for x in group_sizes.detach().cpu().tolist()]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        group_sizes = inputs["group_sizes"]
        model_inputs = {k: v for k, v in inputs.items() if k not in {"labels", "group_sizes"}}
        logits = model(**model_inputs).logits.squeeze(-1)
        loss = _compute_listwise_loss(logits, labels, group_sizes, self.tau)
        if return_outputs:
            return loss, {"logits": logits, "labels": labels, "group_sizes": group_sizes}
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        if prediction_loss_only:
            loss = self.compute_loss(model, inputs)
            return (loss.detach(), None, None)
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        logits = outputs["logits"].detach()
        labels = outputs["labels"].detach()
        group_sizes = outputs["group_sizes"].detach()

        top1_correct: list[float] = []
        offset = 0
        for size in self._iter_groups(group_sizes):
            if size <= 0:
                continue
            group_logits = logits[offset : offset + size]
            group_labels = labels[offset : offset + size]
            offset += size
            top_idx = int(torch.argmax(group_logits))
            top1_correct.append(1.0 if float(group_labels[top_idx]) > 0.5 else 0.0)

        preds = torch.tensor(top1_correct)
        return loss.detach(), preds.cpu(), preds.cpu()


class ListwiseStatsCallback(TrainerCallback):
    def __init__(self, stats: ListwiseBuildStats, label: str):
        self.stats = stats
        self.label = label
        self._seen_epochs: set[int] = set()

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch is None:
            return
        epoch = int(round(state.epoch))
        if epoch in self._seen_epochs:
            return
        self._seen_epochs.add(epoch)
        _print_listwise_stats(self.stats, f"{self.label} epoch {epoch}")


def compute_pairwise_metrics(eval_pred):
    diffs, labels = eval_pred
    diffs = np.asarray(diffs)
    if diffs.ndim > 1:
        diffs = diffs.reshape(-1)
    acc = float((diffs > 0).mean()) if diffs.size else 0.0
    mean_margin = float(diffs.mean()) if diffs.size else 0.0
    return {"rank_acc": acc, "mean_margin": mean_margin}


def compute_listwise_metrics(eval_pred):
    preds, _labels = eval_pred
    preds = np.asarray(preds)
    if preds.ndim > 1:
        preds = preds.reshape(-1)
    acc = float(preds.mean()) if preds.size else 0.0
    return {"rank_acc": acc}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train NLVerifier (NL->Lean reranker) on (nl_statement, lean_statement) pairs."
    )
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--eval-split", type=str, default="", help="Optional separate eval split")
    p.add_argument("--eval-size", type=float, default=0.1)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--max-problems", type=int, default=0)
    p.add_argument("--problem-id-key", type=str, default="problem_id")
    p.add_argument("--nl-key", type=str, default="nl_statement")
    p.add_argument("--pred-key", type=str, default="prediction")
    p.add_argument("--label-key", type=str, default="label")

    p.add_argument("--base-model", type=str, default="microsoft/deberta-v3-base")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--use-features", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--loss", type=str, default="pairwise", choices=["pairwise", "listwise"])
    p.add_argument("--listwise-tau", type=float, default=1.0)
    p.add_argument("--listwise-negatives", type=int, default=0, help="0 = use all negatives")
    p.add_argument("--listwise-max-positives", type=int, default=0, help="0 = use all positives")
    p.add_argument("--listwise-eval-negatives", type=int, default=0, help="0 = use listwise-negatives")
    p.add_argument("--debug-batch", type=int, default=0, help="Debug listwise batches without training.")
    p.add_argument(
        "--max-skip-no-pos-rate",
        type=float,
        default=0.4,
        help="Fail if skipped_no_pos / total_groups exceeds this rate.",
    )
    p.add_argument(
        "--max-skip-no-neg-rate",
        type=float,
        default=0.05,
        help="Fail if skipped_no_neg / total_groups exceeds this rate.",
    )

    p.add_argument("--max-pairs-per-problem", type=int, default=64)
    p.add_argument("--neg-sampling", type=str, default="random", choices=["random", "hard"])
    p.add_argument("--mining-rounds", type=int, default=0)
    p.add_argument("--mining-hard-k", type=int, default=4)
    p.add_argument("--mining-easy-k", type=int, default=1)
    p.add_argument("--mining-epochs", type=float, default=1.0)
    p.add_argument("--warmup-epochs", type=float, default=0.0)
    p.add_argument("--mining-batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_split:
        train_raw = load_dataset_split(args.dataset, args.split)
        eval_raw = load_dataset_split(args.dataset, args.eval_split)
        train_rows = _rows_from_hf(
            train_raw,
            max_rows=int(args.max_rows),
            max_problems=int(args.max_problems),
            problem_id_key=str(args.problem_id_key) if args.problem_id_key else None,
        )
        eval_rows = _rows_from_hf(
            eval_raw,
            max_rows=int(args.max_rows),
            max_problems=int(args.max_problems),
            problem_id_key=str(args.problem_id_key) if args.problem_id_key else None,
        )
    else:
        raw = load_dataset_split(args.dataset, args.split)
        rows = _rows_from_hf(
            raw,
            max_rows=int(args.max_rows),
            max_problems=int(args.max_problems),
            problem_id_key=str(args.problem_id_key) if args.problem_id_key else None,
        )
        train_rows, eval_rows = _split_rows_by_problem_id(
            rows,
            args.problem_id_key if args.problem_id_key else None,
            args.eval_size,
            args.seed,
        )

    base_model = _resolve_base_model(args.base_model)
    local_only = Path(base_model).exists()
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, local_files_only=local_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=1,
        local_files_only=local_only,
    )

    use_features = bool(args.use_features)
    rnd = random.Random(args.seed)

    def _print_pairwise_stats(name: str, examples: list[PairwiseExample]) -> None:
        by_pid = Counter(e.problem_id for e in examples)
        print(f"{name} pairs: {len(examples)} (problems={len(by_pid)})")

    def preprocess(batch):
        nl_clean = [normalize_whitespace(x) for x in batch["nl"]]
        pos_clean = [normalize_lean_statement(x) for x in batch["pos"]]
        neg_clean = [normalize_lean_statement(x) for x in batch["neg"]]

        if use_features:
            pos_clean = [f"{extract_features(s).to_prefix()} {s}" for s in pos_clean]
            neg_clean = [f"{extract_features(s).to_prefix()} {s}" for s in neg_clean]

        enc_pos = tok(
            nl_clean,
            pos_clean,
            truncation=True,
            max_length=args.max_length,
        )
        enc_neg = tok(
            nl_clean,
            neg_clean,
            truncation=True,
            max_length=args.max_length,
        )

        out = {}
        for k, v in enc_pos.items():
            out[f"{k}_pos"] = v
        for k, v in enc_neg.items():
            out[f"{k}_neg"] = v
        return out

    class _PairwiseCollator:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(self, features):
            pos = []
            neg = []
            for f in features:
                pos.append({k[:-4]: v for k, v in f.items() if k.endswith("_pos")})
                neg.append({k[:-4]: v for k, v in f.items() if k.endswith("_neg")})
            pos_batch = self.tokenizer.pad(pos, return_tensors="pt")
            neg_batch = self.tokenizer.pad(neg, return_tensors="pt")
            out = {}
            for k, v in pos_batch.items():
                out[f"{k}_pos"] = v
            for k, v in neg_batch.items():
                out[f"{k}_neg"] = v
            return out

    class _ListwiseCollator:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(self, features):
            nl_list: list[str] = []
            cand_list: list[str] = []
            labels: list[int] = []
            group_sizes: list[int] = []
            for f in features:
                nl = "" if f.get("nl") is None else str(f.get("nl"))
                cands = f.get("candidates") or []
                labs = f.get("labels") or []
                if len(cands) != len(labs):
                    raise ValueError("Candidates/labels length mismatch in listwise batch.")
                group_sizes.append(len(cands))
                for cand, lab in zip(cands, labs):
                    nl_list.append(nl)
                    cand_list.append("" if cand is None else str(cand))
                    lab_int = int(lab)
                    if lab_int not in (0, 1):
                        raise ValueError(f"label must be 0/1, got {lab_int!r}")
                    labels.append(lab_int)
            nl_clean, lean_clean = _prepare_text_pairs(nl_list, cand_list, use_features)
            enc = self.tokenizer(
                nl_clean,
                lean_clean,
                padding=True,
                truncation=True,
                max_length=int(args.max_length),
                return_tensors="pt",
            )
            enc["labels"] = torch.tensor(labels, dtype=torch.float)
            enc["group_sizes"] = torch.tensor(group_sizes, dtype=torch.long)
            return enc

    def _run_debug_batches(examples: list[ListwiseExample], label: str) -> None:
        if int(args.debug_batch) <= 0:
            return
        if args.loss != "listwise":
            raise ValueError("--debug-batch requires --loss listwise")
        if not examples:
            raise ValueError("No listwise examples available for debug mode.")
        batch_size = int(args.batch_size)
        max_batches = min(int(args.debug_batch), math.ceil(len(examples) / max(1, batch_size)))
        device = next(model.parameters()).device
        model.eval()
        for b in range(max_batches):
            batch_examples = examples[b * batch_size : (b + 1) * batch_size]
            features = [
                {"nl": e.nl, "candidates": e.candidates, "labels": e.labels, "sampled_indices": e.sampled_indices}
                for e in batch_examples
            ]
            batch = collator(features)
            labels = batch.pop("labels").to(device)
            group_sizes = batch.pop("group_sizes").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                logits = model(**batch).logits.squeeze(-1)
                loss = _compute_listwise_loss(logits, labels, group_sizes, float(args.listwise_tau))

            if b == 0:
                print(f"{label} debug batch (size={len(batch_examples)})")
                for ex in batch_examples:
                    cand_lens = [len(c) for c in ex.candidates]
                    print(
                        f"problem_id={ex.problem_id} nl_len={len(ex.nl)} "
                        f"cand_lens={cand_lens} labels={ex.labels} "
                        f"sampled_indices={ex.sampled_indices}"
                    )
            print(f"debug batch {b + 1}/{max_batches} loss={loss.item():.4f}")

        raise SystemExit("Debug batch complete.")

    def _make_pairwise_dataset(examples: list[PairwiseExample]) -> Dataset:
        ds = _to_hf_dataset(examples)
        return ds.map(preprocess, batched=True, remove_columns=ds.column_names)

    def _make_trainer(
        train_dataset: Dataset,
        num_epochs: float,
        listwise_stats: ListwiseBuildStats | None = None,
        label: str = "Train",
    ):
        targs = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=float(num_epochs),
            learning_rate=float(args.lr),
            per_device_train_batch_size=int(args.batch_size),
            per_device_eval_batch_size=int(args.batch_size),
            gradient_accumulation_steps=int(args.grad_accum),
            eval_strategy="steps",
            eval_steps=200,
            save_steps=200,
            logging_steps=50,
            save_total_limit=2,
            save_only_model=True,
            remove_unused_columns=False,
            fp16=bool(args.fp16),
            bf16=bool(args.bf16),
            report_to=[],
            load_best_model_at_end=True,
            metric_for_best_model="rank_acc",
            greater_is_better=True,
            ddp_find_unused_parameters=False,
        )

        if args.loss == "pairwise":
            return PairwiseRankingTrainer(
                model=model,
                args=targs,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tok,
                data_collator=collator,
                compute_metrics=compute_pairwise_metrics,
            )
        trainer = ListwiseRankingTrainer(
            model=model,
            args=targs,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tok,
            data_collator=collator,
            compute_metrics=compute_listwise_metrics,
            tau=float(args.listwise_tau),
        )
        if listwise_stats is not None:
            trainer.add_callback(ListwiseStatsCallback(listwise_stats, label))
        return trainer

    if args.loss == "pairwise":
        eval_examples = _build_pairwise_examples(eval_rows, args)
        _print_pairwise_stats("Eval", eval_examples)
        eval_dataset = _make_pairwise_dataset(eval_examples)
        collator = _PairwiseCollator(tok)
    else:
        eval_groups = _build_groups(eval_rows, args)
        eval_negatives = int(args.listwise_eval_negatives)
        if eval_negatives <= 0:
            eval_negatives = int(args.listwise_negatives)
        eval_examples, eval_stats = _build_listwise_examples(
            eval_groups,
            rnd=random.Random(args.seed + 1),
            negatives_per_group=eval_negatives,
            max_positives=int(args.listwise_max_positives),
        )
        _validate_listwise_stats(
            eval_stats,
            "Eval",
            max_no_pos_rate=float(args.max_skip_no_pos_rate),
            max_no_neg_rate=float(args.max_skip_no_neg_rate),
        )
        _print_listwise_stats(eval_stats, "Eval")
        eval_dataset = _to_listwise_dataset(eval_examples)
        collator = _ListwiseCollator(tok)

    final_trainer = None
    train_groups = _build_groups(train_rows, args) if (args.loss == "listwise" or int(args.mining_rounds) > 0) else None

    if args.loss == "listwise" and int(args.debug_batch) > 0:
        debug_examples, debug_stats = _build_listwise_examples(
            train_groups,
            rnd=rnd,
            negatives_per_group=int(args.listwise_negatives),
            max_positives=int(args.listwise_max_positives),
        )
        _validate_listwise_stats(
            debug_stats,
            "Debug",
            max_no_pos_rate=float(args.max_skip_no_pos_rate),
            max_no_neg_rate=float(args.max_skip_no_neg_rate),
        )
        _print_listwise_stats(debug_stats, "Debug")
        _run_debug_batches(debug_examples, "Debug")

    if int(args.mining_rounds) > 0:
        if float(args.warmup_epochs) > 0:
            if args.loss == "pairwise":
                warmup_examples = _build_pairwise_examples(train_rows, args)
                _print_pairwise_stats("Warmup", warmup_examples)
                warmup_dataset = _make_pairwise_dataset(warmup_examples)
                warmup_stats = None
            else:
                warmup_examples, warmup_stats = _build_listwise_examples(
                    train_groups,
                    rnd=rnd,
                    negatives_per_group=int(args.listwise_negatives),
                    max_positives=int(args.listwise_max_positives),
                )
                _validate_listwise_stats(
                    warmup_stats,
                    "Warmup",
                    max_no_pos_rate=float(args.max_skip_no_pos_rate),
                    max_no_neg_rate=float(args.max_skip_no_neg_rate),
                )
                _print_listwise_stats(warmup_stats, "Warmup")
                warmup_dataset = _to_listwise_dataset(warmup_examples)
            final_trainer = _make_trainer(
                warmup_dataset,
                float(args.warmup_epochs),
                listwise_stats=warmup_stats,
                label="Warmup",
            )
            final_trainer.train(resume_from_checkpoint=False)
            model = final_trainer.model

        for round_idx in range(int(args.mining_rounds)):
            model.eval()
            grouped_scores = _score_grouped_candidates(
                train_groups,
                model,
                tok,
                max_length=int(args.max_length),
                use_features=use_features,
                batch_size=int(args.mining_batch_size),
            )
            model.train()

            if args.loss == "pairwise":
                mined_examples, mining_stats = _build_mined_pairwise_examples(
                    train_groups, grouped_scores, args, rnd=rnd
                )
                _print_pairwise_stats(f"Mining {round_idx + 1}", mined_examples)
                _print_mining_stats(mining_stats, f"Mining {round_idx + 1}")
                train_dataset = _make_pairwise_dataset(mined_examples)
                listwise_stats = None
            else:
                mined_examples, mined_stats, mining_stats = _build_mined_listwise_examples(
                    train_groups, grouped_scores, args, rnd=rnd
                )
                _validate_listwise_stats(
                    mined_stats,
                    f"Mining {round_idx + 1}",
                    max_no_pos_rate=float(args.max_skip_no_pos_rate),
                    max_no_neg_rate=float(args.max_skip_no_neg_rate),
                )
                _print_listwise_stats(mined_stats, f"Mining {round_idx + 1}")
                _print_mining_stats(mining_stats, f"Mining {round_idx + 1}")
                train_dataset = _to_listwise_dataset(mined_examples)
                listwise_stats = mined_stats

            final_trainer = _make_trainer(
                train_dataset,
                float(args.mining_epochs),
                listwise_stats=listwise_stats,
                label=f"Mining {round_idx + 1}",
            )
            final_trainer.train(resume_from_checkpoint=False)
            model = final_trainer.model
    else:
        if args.loss == "pairwise":
            train_examples = _build_pairwise_examples(train_rows, args)
            _print_pairwise_stats("Train", train_examples)
            train_dataset = _make_pairwise_dataset(train_examples)
            listwise_stats = None
        else:
            train_examples, train_stats = _build_listwise_examples(
                train_groups,
                rnd=rnd,
                negatives_per_group=int(args.listwise_negatives),
                max_positives=int(args.listwise_max_positives),
            )
            _validate_listwise_stats(
                train_stats,
                "Train",
                max_no_pos_rate=float(args.max_skip_no_pos_rate),
                max_no_neg_rate=float(args.max_skip_no_neg_rate),
            )
            _print_listwise_stats(train_stats, "Train")
            train_dataset = _to_listwise_dataset(train_examples)
            listwise_stats = train_stats

        final_trainer = _make_trainer(
            train_dataset,
            float(args.epochs),
            listwise_stats=listwise_stats,
            label="Train",
        )
        final_trainer.train(resume_from_checkpoint=False)

    if final_trainer is None:
        raise SystemExit("No training rounds were executed.")

    final_trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    metrics = final_trainer.evaluate()
    print("Final eval:", metrics)


if __name__ == "__main__":
    main()
