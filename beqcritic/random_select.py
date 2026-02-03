"""
CLI: select a random candidate per problem (uniform over candidates).

Determinism: selection is seeded per problem_id using a stable hash of (seed, problem_id),
so results are reproducible and independent of input ordering.

Input JSONL (one problem per line):
  {"problem_id": "...", "candidates": ["...", ...]}

Output JSONL format is compatible with `beqcritic.evaluate_selection`:
  {"problem_id": "...", "chosen": "...", "chosen_index": 3, ...}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random


def _stable_index(pid: str, n: int, seed: int) -> int:
    if n <= 0:
        raise ValueError("n must be > 0")
    key = f"{seed}:{pid}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    # Use a 64-bit slice of the digest for a deterministic RNG seed.
    val = int(digest[:16], 16)
    rng = random.Random(val)
    return int(rng.randrange(n))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    seed = int(args.seed)

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            pid = obj.get("problem_id")
            candidates = obj.get("candidates") or []
            if pid is None:
                raise ValueError(f"Missing problem_id in input row: {obj}")
            if not isinstance(candidates, list):
                raise ValueError(f"Expected candidates list for problem_id={pid!r}")
            if not candidates:
                continue

            idx = _stable_index(str(pid), len(candidates), seed)
            out = {
                "problem_id": str(pid),
                "chosen": str(candidates[int(idx)]),
                "chosen_index": int(idx),
                "selection_method": "random",
                "seed": seed,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
