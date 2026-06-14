"""
CLI: NLVerifier-Select (pick the best candidate with NLVerifier).

Scores each candidate against the problem's natural language statement with NLVerifier and
selects the max-score candidate.
"""

from __future__ import annotations

import argparse
import json
from math import exp, isfinite
from pathlib import Path
from typing import Iterable

from .hf_datasets import load_dataset_split
from .verifier import NLVerifier


def _load_nl_map(dataset: str, split: str, id_key: str, nl_key: str) -> dict[str, str]:
    ds = load_dataset_split(dataset, split)
    out: dict[str, str] = {}
    for r in ds:
        if id_key not in r:
            raise ValueError(f"Missing {id_key!r} in dataset row keys: {list(r.keys())}")
        pid = str(r[id_key])
        if pid in out:
            continue
        if nl_key not in r:
            raise ValueError(f"Missing {nl_key!r} in dataset row for id={pid!r}")
        out[pid] = "" if r[nl_key] is None else str(r[nl_key])
    return out


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _load_score_temperature(path: str, fallback: float) -> float:
    if not str(path).strip():
        return _validate_temperature(fallback)
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if "temperature" in obj and isinstance(obj["temperature"], dict):
        value = obj["temperature"].get("fitted", obj["temperature"].get("input", fallback))
    else:
        value = obj.get("temperature", obj.get("fitted_temperature", fallback))
    return _validate_temperature(float(value))


def _selection_confidence(
    scores: list[float],
    chosen_index: int,
    *,
    temperature: float = 1.0,
    minimize: bool = False,
    eligible_indices: list[int] | None = None,
) -> dict:
    if not scores:
        raise ValueError("scores must be non-empty")
    if chosen_index < 0 or chosen_index >= len(scores):
        raise ValueError(f"chosen_index out of range: {chosen_index} (n={len(scores)})")
    temperature = _validate_temperature(temperature)
    eligible = list(range(len(scores))) if eligible_indices is None else [int(i) for i in eligible_indices]
    if chosen_index not in eligible:
        eligible.append(int(chosen_index))
    bad = [i for i in eligible if i < 0 or i >= len(scores)]
    if bad:
        raise ValueError(f"eligible_indices out of range: {bad[:5]} (n={len(scores)})")

    ranked = sorted(eligible, key=lambda i: (-_oriented_score(scores[i], minimize=minimize), i))
    runner_up = next((idx for idx in ranked if idx != chosen_index), None)
    chosen_oriented = _oriented_score(scores[chosen_index], minimize=minimize)
    chosen_probability = _sigmoid(chosen_oriented / temperature)
    out = {
        "score_temperature": float(temperature),
        "chosen_probability": float(chosen_probability),
        "eligible_count": int(len(set(eligible))),
    }
    if runner_up is None:
        out.update(
            {
                "runner_up_index": None,
                "runner_up_score": None,
                "runner_up_probability": None,
                "score_margin": None,
                "probability_margin": None,
            }
        )
        return out

    runner_up_oriented = _oriented_score(scores[runner_up], minimize=minimize)
    runner_up_probability = _sigmoid(runner_up_oriented / temperature)
    out.update(
        {
            "runner_up_index": int(runner_up),
            "runner_up_score": float(scores[runner_up]),
            "runner_up_probability": float(runner_up_probability),
            "score_margin": float(chosen_oriented - runner_up_oriented),
            "probability_margin": float(chosen_probability - runner_up_probability),
        }
    )
    return out


def _oriented_score(score: float, *, minimize: bool) -> float:
    return -float(score) if minimize else float(score)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = exp(-x)
        return 1.0 / (1.0 + z)
    z = exp(x)
    return z / (1.0 + z)


def _validate_temperature(value: float) -> float:
    temperature = float(value)
    if not isfinite(temperature) or temperature <= 0:
        raise ValueError(f"score temperature must be a positive finite value, got {value!r}")
    return temperature


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description="NLVerifier-Select: choose the top-scoring candidate with NLVerifier.")
    p.add_argument("--model", type=str, action="append", required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--dataset-id-key", type=str, default="id")
    p.add_argument("--dataset-nl-key", type=str, default="nl_statement")

    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--problem-id-key", type=str, default="problem_id")
    p.add_argument("--candidates-key", type=str, default="candidates")
    p.add_argument("--typechecks-key", type=str, default="typechecks")

    p.add_argument("--device", type=str, default="", help="e.g. cuda:0, cuda:1, or cpu (default: auto)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--use-features", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--minimize", action="store_true", help="Select the lowest score instead of the highest.")
    p.add_argument("--emit-scores", action="store_true")
    p.add_argument("--emit-confidence", action="store_true", help="Emit calibrated probability and margin fields.")
    p.add_argument(
        "--score-temperature",
        type=float,
        default=1.0,
        help="Temperature for sigmoid(score / T) confidence fields.",
    )
    p.add_argument(
        "--calibration-json",
        type=str,
        default="",
        help="Optional nlverifier_calibration JSON; uses temperature.fitted when present.",
    )
    p.add_argument("--stats-md", type=str, default="", help="Optional NLVerifier-Select markdown summary path")
    p.add_argument("--stats-json", type=str, default="", help="Optional NLVerifier-Select JSON summary path")
    args = p.parse_args()

    nl_map = _load_nl_map(str(args.dataset), str(args.split), str(args.dataset_id_key), str(args.dataset_nl_key))
    score_temperature = (
        _load_score_temperature(str(args.calibration_json), float(args.score_temperature))
        if args.emit_confidence
        else 1.0
    )

    model_names = args.model if isinstance(args.model, list) else [args.model]
    verifiers = [
        NLVerifier(
            model_name_or_path=str(name),
            max_length=int(args.max_length),
            device=str(args.device).strip() or None,
            use_features=bool(args.use_features),
        )
        for name in model_names
    ]

    total = 0
    top1_typechecks = 0
    rescued_by_typecheck = 0
    no_typecheck_survivors = 0
    chosen_probabilities: list[float] = []
    probability_margins: list[float] = []

    with open(args.output, "w", encoding="utf-8") as fout:
        for obj in _iter_jsonl(str(args.input)):
            if args.problem_id_key not in obj:
                raise ValueError(f"Missing {args.problem_id_key!r} in input row: {obj}")
            pid = str(obj[args.problem_id_key])
            cands = obj.get(args.candidates_key) or []
            if not isinstance(cands, list):
                raise ValueError(f"Expected {args.candidates_key!r} to be a list for problem_id={pid!r}")
            if not cands:
                continue
            if pid not in nl_map:
                raise ValueError(f"Missing nl_statement for problem_id={pid!r}")
            nl = nl_map[pid]

            all_scores: list[list[float]] = []
            for verifier in verifiers:
                all_scores.append(
                    verifier.score_pairs([nl] * len(cands), [str(c) for c in cands], batch_size=int(args.batch_size))
                )
            if not all_scores:
                raise ValueError("No models provided.")
            n = len(all_scores[0])
            if any(len(s) != n for s in all_scores):
                raise ValueError("Score length mismatch across models.")
            scores = [sum(vals) / len(all_scores) for vals in zip(*all_scores)]
            if args.minimize:
                raw_idx = min(range(len(scores)), key=lambda i: scores[i])
            else:
                raw_idx = max(range(len(scores)), key=lambda i: scores[i])

            typechecks = obj.get(args.typechecks_key)
            typecheck_mask = None
            if typechecks is not None:
                if not isinstance(typechecks, list):
                    raise ValueError(f"Expected {args.typechecks_key!r} to be a list for problem_id={pid!r}")
                if len(typechecks) != len(cands):
                    raise ValueError(
                        f"Length mismatch for {pid}: candidates={len(cands)} {args.typechecks_key}={len(typechecks)}"
                    )
                typecheck_mask = [bool(x) for x in typechecks]

            best_idx = raw_idx
            raw_top1_typechecks = None
            no_survivors = False
            eligible_indices = None
            confidence_scope = "all_candidates"
            if typecheck_mask is not None:
                raw_top1_typechecks = bool(typecheck_mask[raw_idx])
                if any(typecheck_mask):
                    survivors = [i for i, ok in enumerate(typecheck_mask) if ok]
                    eligible_indices = survivors
                    confidence_scope = "typecheck_survivors"
                    if args.minimize:
                        best_idx = min(survivors, key=lambda i: scores[i])
                    else:
                        best_idx = max(survivors, key=lambda i: scores[i])
                    if raw_top1_typechecks is False and typecheck_mask[best_idx]:
                        rescued_by_typecheck += 1
                else:
                    no_survivors = True
                    no_typecheck_survivors += 1

            total += 1
            if raw_top1_typechecks:
                top1_typechecks += 1
            out = {
                "problem_id": pid,
                "chosen_index": int(best_idx),
                "chosen": cands[best_idx],
                "score": float(scores[best_idx]),
            }
            if raw_top1_typechecks is not None:
                out["raw_top1_index"] = int(raw_idx)
                out["raw_top1_typechecks"] = bool(raw_top1_typechecks)
            if no_survivors:
                out["no_typecheck_survivors"] = True
            if args.emit_scores:
                out["scores"] = [float(s) for s in scores]
            if args.emit_confidence:
                conf = _selection_confidence(
                    scores,
                    int(best_idx),
                    temperature=score_temperature,
                    minimize=bool(args.minimize),
                    eligible_indices=eligible_indices,
                )
                out.update(conf)
                out["confidence_scope"] = confidence_scope
                chosen_probabilities.append(float(conf["chosen_probability"]))
                if conf["probability_margin"] is not None:
                    probability_margins.append(float(conf["probability_margin"]))
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    if args.stats_md or args.stats_json:
        stats = {
            "problems": total,
            "top1_typechecks_rate": (top1_typechecks / total) if total else 0.0,
            "top1_typechecks": top1_typechecks,
            "rescued_by_typecheck_filter": rescued_by_typecheck,
            "no_typecheck_survivors": no_typecheck_survivors,
        }
        if args.emit_confidence:
            stats["score_temperature"] = score_temperature
            stats["mean_chosen_probability"] = _mean(chosen_probabilities)
            stats["mean_probability_margin"] = _mean(probability_margins)
        if args.stats_json:
            with open(args.stats_json, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
                f.write("\n")
        if args.stats_md:
            lines = [
                "# NLVerifier-Select typecheck stats",
                "",
                f"Problems: {total}",
                "",
                "| metric | value |",
                "|---|---:|",
                f"| top1_typechecks_rate | {100.0 * stats['top1_typechecks_rate']:.1f}% |",
                f"| top1_typechecks | {stats['top1_typechecks']} |",
                f"| rescued_by_typecheck_filter | {stats['rescued_by_typecheck_filter']} |",
                f"| no_typecheck_survivors | {stats['no_typecheck_survivors']} |",
            ]
            if args.emit_confidence:
                lines.extend(
                    [
                        f"| score_temperature | {stats['score_temperature']:.4f} |",
                        f"| mean_chosen_probability | {100.0 * stats['mean_chosen_probability']:.1f}% |",
                        f"| mean_probability_margin | {100.0 * stats['mean_probability_margin']:.1f}% |",
                    ]
                )
            lines.append("")
            with open(args.stats_md, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))


if __name__ == "__main__":
    main()
