"""Statistical summaries for BEqCritic selection experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, lgamma, log
from typing import Iterable

DEFAULT_Z = 1.959963984540054


@dataclass(frozen=True)
class ProportionSummary:
    successes: int
    total: int
    rate: float
    ci_low: float
    ci_high: float

    def to_json_dict(self) -> dict[str, float | int]:
        return {
            "successes": self.successes,
            "total": self.total,
            "rate": self.rate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }


@dataclass(frozen=True)
class PairedComparison:
    a_successes: int
    b_successes: int
    both_success: int
    a_only: int
    b_only: int
    neither_success: int
    total: int
    b_minus_a: float
    discordant: int
    exact_sign_p: float

    def to_json_dict(self) -> dict[str, float | int]:
        return {
            "a_successes": self.a_successes,
            "b_successes": self.b_successes,
            "both_success": self.both_success,
            "a_only": self.a_only,
            "b_only": self.b_only,
            "neither_success": self.neither_success,
            "total": self.total,
            "b_minus_a": self.b_minus_a,
            "discordant": self.discordant,
            "exact_sign_p": self.exact_sign_p,
        }


def proportion_summary(
    successes: int, total: int, *, z: float = DEFAULT_Z
) -> ProportionSummary:
    low, high = wilson_interval(successes, total, z=z)
    rate = 0.0 if total == 0 else float(successes) / float(total)
    return ProportionSummary(
        successes=int(successes),
        total=int(total),
        rate=rate,
        ci_low=low,
        ci_high=high,
    )


def wilson_interval(
    successes: int, total: int, *, z: float = DEFAULT_Z
) -> tuple[float, float]:
    """Return a Wilson score confidence interval for a binomial rate."""
    successes = int(successes)
    total = int(total)
    if total < 0:
        raise ValueError("total must be nonnegative")
    if successes < 0 or successes > total:
        raise ValueError("successes must satisfy 0 <= successes <= total")
    if total == 0:
        return (0.0, 0.0)
    if z <= 0 or not isfinite(z):
        raise ValueError("z must be a positive finite value")

    phat = successes / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2.0 * total)) / denom
    half = (
        z
        * ((phat * (1.0 - phat) / total + z * z / (4.0 * total * total)) ** 0.5)
        / denom
    )
    return (max(0.0, center - half), min(1.0, center + half))


def paired_comparison(
    a_success: Iterable[bool], b_success: Iterable[bool]
) -> PairedComparison:
    a_vals = [bool(value) for value in a_success]
    b_vals = [bool(value) for value in b_success]
    if len(a_vals) != len(b_vals):
        raise ValueError("paired inputs must have the same length")

    both = sum(1 for a, b in zip(a_vals, b_vals) if a and b)
    a_only = sum(1 for a, b in zip(a_vals, b_vals) if a and not b)
    b_only = sum(1 for a, b in zip(a_vals, b_vals) if b and not a)
    neither = sum(1 for a, b in zip(a_vals, b_vals) if not a and not b)
    total = len(a_vals)
    discordant = a_only + b_only
    b_minus_a = 0.0 if total == 0 else (b_only - a_only) / float(total)
    p_value = exact_two_sided_sign_test(wins=b_only, losses=a_only)

    return PairedComparison(
        a_successes=both + a_only,
        b_successes=both + b_only,
        both_success=both,
        a_only=a_only,
        b_only=b_only,
        neither_success=neither,
        total=total,
        b_minus_a=b_minus_a,
        discordant=discordant,
        exact_sign_p=p_value,
    )


def exact_two_sided_sign_test(*, wins: int, losses: int) -> float:
    """Exact two-sided sign-test p-value for paired discordant outcomes."""
    wins = int(wins)
    losses = int(losses)
    if wins < 0 or losses < 0:
        raise ValueError("wins and losses must be nonnegative")
    n = wins + losses
    if n == 0:
        return 1.0
    tail_k = min(wins, losses)
    log_probs = [_log_binomial_half_pmf(k, n) for k in range(tail_k + 1)]
    max_log = max(log_probs)
    tail = exp(max_log) * sum(exp(value - max_log) for value in log_probs)
    return min(1.0, 2.0 * tail)


def _log_binomial_half_pmf(k: int, n: int) -> float:
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) - n * log(2.0)
