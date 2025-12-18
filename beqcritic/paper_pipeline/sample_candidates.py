"""
CLI: sample multiple candidates per problem using HuggingFace Transformers.

This is an optional convenience for running an end-to-end "paper style" pipeline
without a separate sampling repo. It is intentionally minimal and does not try to
handle every chat-template edge case.

Input JSONL (one problem per line):
  {"problem_id": "...", "prompt": "..."}

Output JSONL (one candidate per line):
  {"problem_id": "...", "sample_idx": 0, "candidate": "...", "seed": 123}
"""

from __future__ import annotations

import argparse
import json
import random
import zlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--problem-id-key", type=str, default="problem_id")
    p.add_argument("--prompt-key", type=str, default="prompt")
    p.add_argument("--num-samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="", help="e.g. cuda:0, cuda:1, or cpu (default: auto)")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    args = p.parse_args()

    device = args.device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")

    _set_seed(int(args.seed))

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            pid = obj.get(args.problem_id_key)
            prompt = obj.get(args.prompt_key)
            if pid is None or prompt is None:
                raise ValueError(f"Missing {args.problem_id_key!r} or {args.prompt_key!r} in input row: {obj}")
            prompt_s = str(prompt)
            enc = tok(prompt_s, return_tensors="pt").to(device)
            prompt_len = int(enc["input_ids"].shape[1])

            # Sample deterministically per problem for reproducibility.
            base_seed = int(args.seed) ^ int(zlib.crc32(str(pid).encode("utf-8")))

            for i in range(int(args.num_samples)):
                _set_seed(base_seed + i)
                out_ids = model.generate(
                    **enc,
                    do_sample=True,
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    max_new_tokens=int(args.max_new_tokens),
                    repetition_penalty=float(args.repetition_penalty),
                    pad_token_id=tok.eos_token_id,
                )
                gen = tok.decode(out_ids[0][prompt_len:], skip_special_tokens=True)
                fout.write(
                    json.dumps(
                        {
                            str(args.problem_id_key): pid,
                            "sample_idx": int(i),
                            "candidate": gen,
                            "seed": int(base_seed + i),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
