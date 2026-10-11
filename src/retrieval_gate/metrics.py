"""Ranking metrics. Definitions follow the Retrieval Evaluation Lab project (see NOTICE)."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

METRICS = ("precision", "recall", "rr", "ndcg")
MAX_GRADE = 3


class InputError(ValueError):
    """The supplied rankings, judgments or options are not usable."""


def validate_k(k: object) -> int:
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise InputError("k must be a positive integer")
    return k


def validate_ranking(ranking: object, label: str = "ranking") -> Sequence[str]:
    if not isinstance(ranking, (list, tuple)):
        raise InputError(f"{label}: expected a list of document ids")
    if any(not isinstance(d, str) or not d for d in ranking):
        raise InputError(f"{label}: document ids must be non-empty strings")
    if len(set(ranking)) != len(ranking):
        raise InputError(f"{label}: document ids must be unique within a ranking")
    return ranking


def validate_judgments(judgments: object, label: str = "judgments") -> Mapping[str, int]:
    if not isinstance(judgments, dict):
        raise InputError(f"{label}: expected an object mapping document id to grade")
    for doc, grade in judgments.items():
        if not isinstance(grade, int) or isinstance(grade, bool) or not 0 <= grade <= MAX_GRADE:
            raise InputError(f"{label}: grade for {doc!r} must be an integer from 0 to {MAX_GRADE}")
    return judgments


def score_query(ranking: Sequence[str], judgments: Mapping[str, int], k: int) -> dict[str, float]:
    """Score one query. Unjudged documents get zero gain and are counted in ``unjudged``."""
    validate_k(k)
    validate_ranking(ranking)
    validate_judgments(judgments)
    top = list(ranking[:k])
    relevant = sum(1 for g in judgments.values() if g > 0)
    hits = sum(1 for d in top if judgments.get(d, 0) > 0)
    first = next((i for i, d in enumerate(top) if judgments.get(d, 0) > 0), None)
    dcg = sum((2 ** judgments.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(top))
    ideal = sum(
        (2**g - 1) / math.log2(i + 2)
        for i, g in enumerate(sorted(judgments.values(), reverse=True)[:k])
    )
    return {
        "precision": hits / k,
        "recall": hits / relevant if relevant else 0.0,
        "rr": 1 / (first + 1) if first is not None else 0.0,
        "ndcg": dcg / ideal if ideal else 0.0,
        "unjudged": sum(1 for d in top if d not in judgments),
    }

