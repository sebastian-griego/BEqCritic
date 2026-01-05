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
import inspect
import random
from pathlib import Path
from collections import Counter
import os

import numpy as np
import torch
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
from .labels import coerce_binary_label

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


class DebugTrainer(Trainer):
    def __init__(self, *args, debug_checks: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.debug_checks = debug_checks
        self._supports_num_items_in_batch = (
            "num_items_in_batch" in inspect.signature(super().compute_loss).parameters
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        if self.debug_checks:
            base_model = model.module if hasattr(model, "module") else model
            labels = inputs.get("labels")
            if labels is not None:
                labels_t = labels if torch.is_tensor(labels) else torch.as_tensor(labels)
                bad = ~((labels_t == -100) | ((labels_t >= 0) & (labels_t < base_model.config.num_labels)))
                if torch.any(bad).item():
                    bad_vals = torch.unique(labels_t[bad]).detach().cpu().tolist()
                    raise ValueError(f"Bad labels {bad_vals}, num_labels={base_model.config.num_labels}")

            input_ids = inputs.get("input_ids")
            if input_ids is not None:
                ids_t = input_ids if torch.is_tensor(input_ids) else torch.as_tensor(input_ids)
                vocab_size = base_model.get_input_embeddings().num_embeddings
                bad = (ids_t < 0) | (ids_t >= vocab_size)
                if torch.any(bad).item():
                    bad_vals = torch.unique(ids_t[bad]).detach().cpu().tolist()
                    raise ValueError(f"Bad token ids {bad_vals}, vocab_size={vocab_size}")

        if self._supports_num_items_in_batch:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
                **kwargs,
            )
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)


def _rows_from_hf(
    ds: "object",
    *,
    max_rows: int = 0,
    max_problems: int = 0,
    problem_id_key: str | None = None,
) -> list[dict]:
    """
    Materialize a HF dataset split into a list of dicts, with optional size limits.

    max_problems limits the number of unique problem IDs observed, when problem_id_key is provided.
    """
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

def _build_examples(rows: list[dict], args) -> list[PairExample]:
    for r in rows:
        r[args.label_key] = coerce_binary_label(r.get(args.label_key))

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
            pos_sampling=args.cand_pos_sampling,
            neg_sampling=args.cand_neg_sampling,
        )))

    if getattr(args, "symmetrize", False):
        examples = examples + [
            PairExample(a=e.b, b=e.a, label=e.label, task=e.task, problem_id=e.problem_id)
            for e in examples
        ]

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
    p.add_argument("--max-rows", type=int, default=0, help="Limit number of dataset rows loaded (0 = no limit).")
    p.add_argument(
        "--max-problems",
        type=int,
        default=0,
        help="Limit number of unique problems loaded (0 = no limit; requires --problem-id-key).",
    )
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
    p.add_argument(
        "--cand-pos-sampling",
        type=str,
        default="random",
        choices=["random", "hard"],
        help="cand_vs_cand positives: random pairs or hard (low BLEU) pairs.",
    )
    p.add_argument(
        "--cand-neg-sampling",
        type=str,
        default="random",
        choices=["random", "hard"],
        help="cand_vs_cand negatives: random pairs or hard (high BLEU) pairs.",
    )

    p.add_argument("--eval-size", type=float, default=0.1, help="Fraction of problem IDs to use for eval (group split)")
    p.add_argument("--eval-split", type=str, default="", help="Optional separate eval split (disables --eval-size)")
    p.add_argument(
        "--allow-problem-id-overlap",
        action="store_true",
        help="Allow train/eval overlap on --problem-id-key when using --eval-split (debug only; can cause leakage).",
    )
    p.add_argument(
        "--write-split-ids",
        action="store_true",
        help="Write train/eval problem_id lists into --output-dir for reproducibility.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--symmetrize", action="store_true", help="Train on both (A,B) and (B,A) for every pair")

    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--debug-checks", action="store_true", help="Enable label/token range checks before loss.")
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

        if args.problem_id_key:
            train_pids = {str(r.get(args.problem_id_key)) for r in train_rows}
            eval_pids = {str(r.get(args.problem_id_key)) for r in eval_rows}
            overlap = sorted(train_pids & eval_pids)
            if overlap and not args.allow_problem_id_overlap:
                hint = ""
                if len(overlap) == len(eval_pids):
                    hint = (
                        f" Note: every eval problem_id is also present in train. "
                        f"This often happens when a local/offline dataset has a synthetic '{args.split}' split "
                        f"(e.g., '{args.split}' == valid+test). "
                        f"Use --split valid with --eval-size, or re-download a dataset with disjoint splits."
                    )
                example = ", ".join(overlap[:5])
                raise RuntimeError(
                    f"Train/eval splits overlap on problem IDs ({len(overlap)} overlapping).{hint} "
                    f"Example overlapping IDs: {example}"
                )
            if overlap and args.allow_problem_id_overlap:
                rank = os.environ.get("RANK", "0")
                if rank == "0":
                    example = ", ".join(overlap[:5])
                    print(
                        f"WARNING: train/eval overlap on problem IDs ({len(overlap)} overlapping); "
                        f"--allow-problem-id-overlap enabled. Example IDs: {example}"
                    )
            print(
                f"Dataset split: {args.split} -> train ({len(train_rows)} rows, {len(train_pids)} problems), "
                f"{args.eval_split} -> eval ({len(eval_rows)} rows, {len(eval_pids)} problems)"
            )
            if args.write_split_ids:
                (out_dir / "train_problem_ids.txt").write_text("\n".join(sorted(train_pids)) + "\n", encoding="utf-8")
                (out_dir / "eval_problem_ids.txt").write_text("\n".join(sorted(eval_pids)) + "\n", encoding="utf-8")
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

        if args.problem_id_key:
            train_pids = {str(r.get(args.problem_id_key)) for r in train_rows}
            eval_pids = {str(r.get(args.problem_id_key)) for r in eval_rows}
            if train_pids & eval_pids:
                raise RuntimeError("Problem-id split failed: train/eval problem IDs overlap")
            print(
                f"Group split by {args.problem_id_key}: "
                f"{len(train_rows)} train rows ({len(train_pids)} problems), "
                f"{len(eval_rows)} eval rows ({len(eval_pids)} problems)"
            )
            if args.write_split_ids:
                (out_dir / "train_problem_ids.txt").write_text("\n".join(sorted(train_pids)) + "\n", encoding="utf-8")
                (out_dir / "eval_problem_ids.txt").write_text("\n".join(sorted(eval_pids)) + "\n", encoding="utf-8")

    train_examples = _build_examples(train_rows, args)
    eval_examples = _build_examples(eval_rows, args)

    def _print_example_stats(name: str, examples: list[PairExample]) -> None:
        tasks = Counter(e.task for e in examples)
        labels = Counter(int(e.label) for e in examples)
        print(
            f"{name} examples: {len(examples)} "
            f"(pos={labels.get(1, 0)}, neg={labels.get(0, 0)}) "
            f"by_task={dict(tasks)}"
        )

    _print_example_stats("Train", train_examples)
    _print_example_stats("Eval", eval_examples)
    train_ds = _to_hf_dataset(train_examples)
    eval_ds = _to_hf_dataset(eval_examples)

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
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=50,
        save_total_limit=2,
        save_only_model=True,
        fp16=bool(args.fp16),
        bf16=bool(args.bf16),
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        ddp_find_unused_parameters=False,
        seed=int(args.seed),
    )

    trainer = DebugTrainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
        debug_checks=bool(args.debug_checks),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    metrics = trainer.evaluate()
    print("Final eval:", metrics)

if __name__ == "__main__":
    main()
