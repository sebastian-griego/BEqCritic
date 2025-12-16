"""
Training script for BeqCritic.

Example:
  python -m beqcritic.train_beq_critic \
    --dataset PAug/ProofNetVerif \
    --split train \
    --pred-key prediction \
    --ref-key reference \
    --label-key label \
    --problem-id-key problem_id \
    --base-model microsoft/deberta-v3-base \
    --output-dir checkpoints/beqcritic_deberta \
    --task-mix pred_vs_ref,cand_vs_cand

Design choices:
  - No Lean calls during training
  - Two-class classifier (equivalent vs not)
  - Includes cheap structural feature tokens
"""
from __future__ import annotations

import argparse
import random
from typing import Any
from pathlib import Path

import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)

from .textnorm import normalize_lean_statement
from .features import featurize_pair
from .pairgen import make_pred_vs_ref_pairs, make_cand_vs_cand_pairs, PairExample
from .hf_datasets import load_dataset_split

REPO_ROOT = Path(__file__).resolve().parents[1]

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

def _rows_from_hf(ds) -> list[dict]:
    return [dict(r) for r in ds]

def _build_examples(rows: list[dict], args) -> list[PairExample]:
    for r in rows:
        r[args.label_key] = _guess_bool_label(r.get(args.label_key))

    examples: list[PairExample] = []
    task_mix = [t.strip() for t in args.task_mix.split(",") if t.strip()]

    if "pred_vs_ref" in task_mix:
        if not args.ref_key:
            raise ValueError("--ref-key is required for pred_vs_ref")
        examples.extend(list(make_pred_vs_ref_pairs(
            rows,
            args.pred_key,
            args.ref_key,
            args.label_key,
            args.problem_id_key if args.problem_id_key else None,
        )))

    if "cand_vs_cand" in task_mix:
        if not args.problem_id_key:
            raise ValueError("--problem-id-key is required for cand_vs_cand")
        examples.extend(list(make_cand_vs_cand_pairs(
            rows,
            pred_key=args.pred_key,
            label_key=args.label_key,
            problem_id_key=args.problem_id_key,
            max_pos_per_problem=args.max_pos_per_problem,
            max_neg_per_problem=args.max_neg_per_problem,
            seed=args.seed,
        )))

    random.Random(args.seed).shuffle(examples)
    return examples

def _to_hf_dataset(examples: list[PairExample]) -> Dataset:
    return Dataset.from_dict({
        "a": [e.a for e in examples],
        "b": [e.b for e in examples],
        "label": [e.label for e in examples],
        "task": [e.task for e in examples],
        "problem_id": [e.problem_id for e in examples],
    })

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # logits shape [N, 2]
    probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    p1 = probs[:, 1]
    preds = (p1 >= 0.5).astype(np.int32)
    labels = labels.astype(np.int32)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 2 * precision * recall / max(1e-12, (precision + recall))
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": acc}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--pred-key", type=str, default="prediction")
    p.add_argument("--ref-key", type=str, default="reference")
    p.add_argument("--label-key", type=str, default="label")
    p.add_argument("--problem-id-key", type=str, default="problem_id")
    p.add_argument("--task-mix", type=str, default="pred_vs_ref,cand_vs_cand")

    p.add_argument("--base-model", type=str, default="microsoft/deberta-v3-base")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--max-length", type=int, default=512)

    p.add_argument("--max-pos-per-problem", type=int, default=16)
    p.add_argument("--max-neg-per-problem", type=int, default=32)

    p.add_argument("--eval-size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)

    raw = load_dataset_split(args.dataset, args.split)
    rows = _rows_from_hf(raw)
    examples = _build_examples(rows, args)
    ds = _to_hf_dataset(examples)

    ds = ds.shuffle(seed=args.seed)
    n_eval = max(1, int(len(ds) * args.eval_size))
    eval_ds = ds.select(range(n_eval))
    train_ds = ds.select(range(n_eval, len(ds)))

    base_model = _resolve_base_model(args.base_model)
    local_only = Path(base_model).exists()
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, local_files_only=local_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
        local_files_only=local_only,
    )

    def preprocess(batch):
        a_clean = [normalize_lean_statement(x) for x in batch["a"]]
        b_clean = [normalize_lean_statement(x) for x in batch["b"]]

        a_pref, b_pref = [], []
        for aa, bb in zip(a_clean, b_clean):
            fa, fb = featurize_pair(aa, bb)
            a_pref.append(fa + " " + aa)
            b_pref.append(fb + " " + bb)

        enc = tok(
            a_pref, b_pref,
            truncation=True,
            max_length=args.max_length,
        )
        enc["labels"] = batch["label"]
        return enc

    train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    eval_tok = eval_ds.map(preprocess, batched=True, remove_columns=eval_ds.column_names)

    collator = DataCollatorWithPadding(tok)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=float(args.epochs),
        learning_rate=float(args.lr),
        per_device_train_batch_size=int(args.batch_size),
        per_device_eval_batch_size=int(args.batch_size),
        gradient_accumulation_steps=int(args.grad_accum),
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=50,
        save_total_limit=2,
        fp16=bool(args.fp16),
        bf16=bool(args.bf16),
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    trainer = Trainer(
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
