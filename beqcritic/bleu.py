from __future__ import annotations

import math
import re

from .textnorm import normalize_lean_statement


_TOK_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_'.]*|[0-9]+|[^\s]")


def _tokenize(text: str) -> list[str]:
    return _TOK_RE.findall(text)


def _ngrams(tokens: list[str], n: int) -> dict[tuple[str, ...], int]:
    out: dict[tuple[str, ...], int] = {}
    if n <= 0:
        return out
    for i in range(0, max(0, len(tokens) - n + 1)):
        g = tuple(tokens[i : i + n])
        out[g] = out.get(g, 0) + 1
    return out


def _bleu_score(hyp: list[str], ref: list[str], max_n: int = 4, smooth: float = 1.0) -> float:
    if not hyp and not ref:
        return 1.0
    if not hyp:
        return 0.0

    log_p_sum = 0.0
    for n in range(1, max_n + 1):
        hyp_ngrams = _ngrams(hyp, n)
        ref_ngrams = _ngrams(ref, n)
        overlap = 0
        total = 0
        for g, c in hyp_ngrams.items():
            total += c
            overlap += min(c, ref_ngrams.get(g, 0))
        p_n = (overlap + smooth) / (total + smooth) if total > 0 else 0.0
        log_p_sum += (1.0 / max_n) * math.log(max(1e-12, p_n))

    bp = 1.0
    if len(hyp) < len(ref) and len(hyp) > 0:
        bp = math.exp(1.0 - (len(ref) / len(hyp)))

    return float(bp * math.exp(log_p_sum))


def sym_bleu(a: str, b: str) -> float:
    ta = _tokenize(normalize_lean_statement(a))
    tb = _tokenize(normalize_lean_statement(b))
    return 0.5 * (_bleu_score(ta, tb) + _bleu_score(tb, ta))


def bleu_medoid_index(candidates: list[str]) -> tuple[int, float]:
    """
    Return (medoid_index, centrality) under a symmetric BLEU similarity.

    Centrality is mean similarity to other candidates (in [0,1] approximately).
    """
    if not candidates:
        raise ValueError("No candidates provided")
    if len(candidates) == 1:
        return 0, 1.0

    norm = [normalize_lean_statement(c) for c in candidates]
    toks = [_tokenize(s) for s in norm]
    n = len(candidates)

    best_idx = 0
    best_cent = float("-inf")
    for i in range(n):
        sims: list[float] = []
        for j in range(n):
            if j == i:
                continue
            sims.append(
                0.5 * (_bleu_score(toks[i], toks[j]) + _bleu_score(toks[j], toks[i]))
            )
        cent = sum(sims) / max(1, len(sims))
        if cent > best_cent:
            best_cent = cent
            best_idx = i
        elif cent == best_cent:
            if len(norm[i]) < len(norm[best_idx]):
                best_idx = i
            elif len(norm[i]) == len(norm[best_idx]) and i < best_idx:
                best_idx = i

    return int(best_idx), float(best_cent)

