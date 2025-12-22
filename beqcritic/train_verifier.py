"""
Train a reference-free verifier: score (nl_statement, lean_statement) pairs.

Uses a pairwise ranking loss per problem id:
  score(nl, pos) > score(nl, neg)
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import dataclass
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


def _guess_bool_label(x: Any) -> int:
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, str):
        xl = x.strip().lower()
        if xl in ["1", "true", "yes", "correct", "ok"]:
            return 1
        if xl in ["0", "false", "no", "incorrect", "wrong"]:
            return 0
    raise ValueError(f"Cannot interpret label value: {x!r}")


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


@dataclass(frozen=True)
class PairwiseExample:
    nl: str
    pos: str
    neg: str
    problem_id: str | None = None


def _build_pairwise_examples(rows: list[dict], args) -> list[PairwiseExample]:
    for r in rows:
        r[args.label_key] = _guess_bool_label(r.get(args.label_key))

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


def _to_hf_dataset(examples: list[PairwiseExample]) -> Dataset:
    return Dataset.from_dict(
        {
            "nl": [e.nl for e in examples],
            "pos": [e.pos for e in examples],
            "neg": [e.neg for e in examples],
            "problem_id": [e.problem_id for e in examples],
        }
    )


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


def compute_metrics(eval_pred):
    diffs, labels = eval_pred
    diffs = np.asarray(diffs)
    if diffs.ndim > 1:
        diffs = diffs.reshape(-1)
    acc = float((diffs > 0).mean()) if diffs.size else 0.0
    mean_margin = float(diffs.mean()) if diffs.size else 0.0
    return {"rank_acc": acc, "mean_margin": mean_margin}


def main() -> None:
    p = argparse.ArgumentParser()
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

    p.add_argument("--max-pairs-per-problem", type=int, default=64)
    p.add_argument("--neg-sampling", type=str, default="random", choices=["random", "hard"])
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

    train_examples = _build_pairwise_examples(train_rows, args)
    eval_examples = _build_pairwise_examples(eval_rows, args)

    def _print_example_stats(name: str, examples: list[PairwiseExample]) -> None:
        by_pid = Counter(e.problem_id for e in examples)
        print(f"{name} pairs: {len(examples)} (problems={len(by_pid)})")

    _print_example_stats("Train", train_examples)
    _print_example_stats("Eval", eval_examples)

    train_ds = _to_hf_dataset(train_examples)
    eval_ds = _to_hf_dataset(eval_examples)

    base_model = _resolve_base_model(args.base_model)
    local_only = Path(base_model).exists()
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, local_files_only=local_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=1,
        local_files_only=local_only,
    )

    use_features = bool(args.use_features)

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

    train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    eval_tok = eval_ds.map(preprocess, batched=True, remove_columns=eval_ds.column_names)

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

    collator = _PairwiseCollator(tok)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=float(args.epochs),
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

    trainer = PairwiseRankingTrainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    metrics = trainer.evaluate()
    print("Final eval:", metrics)


if __name__ == "__main__":
    main()
