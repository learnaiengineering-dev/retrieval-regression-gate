"""Compare two systems query by query."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .metrics import METRICS, InputError, score_query, validate_judgments, validate_k, validate_ranking

SCHEMA_VERSION = 1
_EPS = 1e-9


@dataclass(frozen=True)
class QueryResult:
    query: str
    baseline: dict[str, float]
    candidate: dict[str, float]
    regressed: bool

    def delta(self, metric: str) -> float:
        return self.candidate[metric] - self.baseline[metric]


@dataclass(frozen=True)
class GateResult:
    k: int
    metric: str
    tolerance: float
    min_mean: float | None
    queries: list[QueryResult]
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def regressions(self) -> list[QueryResult]:
        return [q for q in self.queries if q.regressed]

    def mean(self, side: str, metric: str | None = None) -> float:
        m = metric or self.metric
        return sum(getattr(q, side)[m] for q in self.queries) / len(self.queries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "k": self.k,
            "metric": self.metric,
            "tolerance": self.tolerance,
            "min_mean": self.min_mean,
            "passed": self.passed,
            "failures": list(self.failures),
            "warnings": list(self.warnings),
            "mean": {
                side: {m: self.mean(side, m) for m in METRICS} for side in ("baseline", "candidate")
            },
            "queries": {
                q.query: {
                    "baseline": q.baseline,
                    "candidate": q.candidate,
                    "delta": {m: q.delta(m) for m in METRICS},
                    "regressed": q.regressed,
                }
                for q in self.queries
            },
        }


def compare(
    baseline: Mapping[str, Sequence[str]],
    candidate: Mapping[str, Sequence[str]],
    judgments: Mapping[str, Mapping[str, int]],
    k: int = 10,
    metric: str = "ndcg",
    tolerance: float = 0.0,
    min_mean: float | None = None,
) -> GateResult:
    """Score both systems and flag every query whose ``metric`` dropped by more than ``tolerance``."""
    validate_k(k)
    if metric not in METRICS:
        raise InputError(f"metric must be one of {', '.join(METRICS)}")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or tolerance < 0:
        raise InputError("tolerance must be a number >= 0")
    if min_mean is not None and (isinstance(min_mean, bool) or not isinstance(min_mean, (int, float))):
        raise InputError("min_mean must be a number")
    for label, data in (("baseline", baseline), ("candidate", candidate), ("judgments", judgments)):
        if not isinstance(data, Mapping):
            raise InputError(f"{label}: expected an object keyed by query")
    names = {"baseline": set(baseline), "candidate": set(candidate), "judgments": set(judgments)}
    if not names["judgments"]:
        raise InputError("no queries found")
    if len({frozenset(v) for v in names.values()}) != 1:
        every = set().union(*names.values())
        detail = "; ".join(
            f"{label} is missing {sorted(every - have)}" for label, have in names.items() if every - have
        )
        raise InputError(f"baseline, candidate and judgments must cover the same queries: {detail}")

    results: list[QueryResult] = []
    warnings: list[str] = []
    for q in sorted(judgments):
        j = validate_judgments(judgments[q], f"judgments[{q!r}]")
        b = score_query(validate_ranking(baseline[q], f"baseline[{q!r}]"), j, k)
        c = score_query(validate_ranking(candidate[q], f"candidate[{q!r}]"), j, k)
        results.append(QueryResult(q, b, c, c[metric] < b[metric] - tolerance - _EPS))
        if not any(g > 0 for g in j.values()):
            warnings.append(f"{q!r} has no relevant judgments, so every score for it is 0")
        if c["unjudged"]:
            warnings.append(
                f"{q!r}: {int(c['unjudged'])} unjudged candidate document(s) in the top {k} score zero"
            )

    failures = [
        f"{r.query!r}: {metric} {r.baseline[metric]:.3f} -> {r.candidate[metric]:.3f} ({r.delta(metric):+.3f})"
        for r in results
        if r.regressed
    ]
    result = GateResult(k, metric, float(tolerance), min_mean, results, failures, warnings)
    if min_mean is not None and result.mean("candidate") < min_mean - _EPS:
        failures.append(f"mean {metric} {result.mean('candidate'):.3f} is below the floor {min_mean:.3f}")
    return result

