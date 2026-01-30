"""
CLI: create ID-disjoint train/dev/test splits from one or more dataset splits.

Example:
  python -m beqcritic.make_id_disjoint_split \
    --dataset PAug/ProofNetVerif \
    --input-splits valid,test \
    --id-key id \
    --output-dir hf_datasets/ProofNetVerif_id_disjoint \
    --seed 0 \
    --train-frac 0.7 --dev-frac 0.15
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import concatenate_datasets

from .hf_datasets import load_dataset_split


def _parse_splits(value: str) -> list[str]:
    splits = [s.strip() for s in value.split(",") if s.strip()]
    if not splits:
        raise ValueError("No input splits provided.")
    return splits


def _json_default(obj: Any) -> Any:
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return list(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    try:
        import numpy as np

        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return str(obj)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=_json_default) + "\n")


def _count_ids(rows: list[dict[str, Any]], id_key: str) -> int:
    return len({str(r.get(id_key)) for r in rows})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--input-splits", type=str, default="valid,test")
    p.add_argument("--id-key", type=str, default="id")
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--dev-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument(
        "--dev-name",
        type=str,
        default="dev",
        help="Split name to use for the development set (e.g., dev or valid).",
    )
    p.add_argument(
        "--write-id-lists",
        action="store_true",
        help="Write train/dev/test problem_id lists into the output directory.",
    )
    args = p.parse_args()

    if args.train_frac <= 0 or args.dev_frac <= 0 or args.train_frac + args.dev_frac >= 1:
        raise ValueError("Invalid split fractions: require train_frac>0, dev_frac>0, train_frac+dev_frac<1.")

    input_splits = _parse_splits(args.input_splits)
    datasets = [load_dataset_split(args.dataset, split) for split in input_splits]
    if not datasets:
        raise ValueError("No datasets loaded. Check --dataset and --input-splits.")
    ds_all = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)

    id_key = str(args.id_key)
    ids_set: set[str] = set()
    for row in ds_all:
        if id_key not in row:
            raise KeyError(f"Missing {id_key!r} in dataset row: keys={list(row.keys())}")
        ids_set.add(str(row.get(id_key)))

    ids = sorted(ids_set)
    if len(ids) < 3:
        raise ValueError(f"Need at least 3 unique ids to build splits, found {len(ids)}.")

    rng = random.Random(int(args.seed))
    rng.shuffle(ids)

    n_total = len(ids)
    n_train = int(args.train_frac * n_total)
    n_dev = int(args.dev_frac * n_total)
    n_test = n_total - n_train - n_dev
    if min(n_train, n_dev, n_test) <= 0:
        raise ValueError(
            f"Split sizes too small for {n_total} ids: train={n_train}, dev={n_dev}, test={n_test}."
        )

    train_ids = set(ids[:n_train])
    dev_ids = set(ids[n_train : n_train + n_dev])
    test_ids = set(ids[n_train + n_dev :])

    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], args.dev_name: [], "test": []}
    for row in ds_all:
        pid = str(row.get(id_key))
        if pid in train_ids:
            split_rows["train"].append(row)
        elif pid in dev_ids:
            split_rows[args.dev_name].append(row)
        else:
            split_rows["test"].append(row)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, rows in split_rows.items():
        path = out_dir / f"{split_name}.jsonl"
        _write_jsonl(path, rows)
        n_ids = _count_ids(rows, id_key)
        print(f"Wrote {len(rows)} rows ({n_ids} ids) -> {path}")

    if args.write_id_lists:
        id_lists = {
            "train": train_ids,
            args.dev_name: dev_ids,
            "test": test_ids,
        }
        for split_name, id_set in id_lists.items():
            path = out_dir / f"{split_name}_ids.txt"
            path.write_text("\n".join(sorted(id_set)) + "\n", encoding="utf-8")
        print("Wrote split id lists.")


if __name__ == "__main__":
    main()
