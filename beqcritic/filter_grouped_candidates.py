"""
CLI: filter a grouped candidates JSONL file by problem_id.

This is useful for creating train/dev subsets after training writes split ids via
`--write-split-ids` in `beqcritic.train_beq_critic`.
"""
from __future__ import annotations

import argparse
import json


def _load_ids(path: str) -> set[str]:
    ids: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            ids.add(s)
    return ids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--problem-ids-file", type=str, required=True)
    p.add_argument("--invert", action="store_true", help="Keep problem_ids NOT in --problem-ids-file")
    p.add_argument("--max-problems", type=int, default=0, help="Limit problems written (0 = no limit)")
    args = p.parse_args()

    keep = _load_ids(args.problem_ids_file)

    n_in = 0
    n_out = 0
    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            n_in += 1
            obj = json.loads(line)
            pid = obj.get("problem_id")
            if pid is None:
                raise ValueError(f"Missing problem_id in input: {obj}")
            pid = str(pid)
            ok = (pid in keep)
            if args.invert:
                ok = not ok
            if not ok:
                continue
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_out += 1
            if args.max_problems and n_out >= int(args.max_problems):
                break

    print(f"Read {n_in} problems, wrote {n_out} to {args.output}")


if __name__ == "__main__":
    main()

