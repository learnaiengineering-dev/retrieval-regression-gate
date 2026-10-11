"""Render a GateResult as text, Markdown or JSON."""
from __future__ import annotations

import json

from .compare import GateResult


def _flag(r) -> str:
    return "REGRESSED" if r.regressed else ""


def to_text(res: GateResult) -> str:
    m = res.metric
    width = max([len("query")] + [len(r.query) for r in res.queries])
    lines = [
        f"Retrieval Regression Gate  k={res.k}  metric={m}  tolerance={res.tolerance:g}",
        f"{'query':<{width}}  {'baseline':>8}  {'candidate':>9}  {'delta':>7}",
    ]
    for r in res.queries:
        lines.append(
            f"{r.query:<{width}}  {r.baseline[m]:>8.3f}  {r.candidate[m]:>9.3f}  {r.delta(m):>+7.3f}  {_flag(r)}".rstrip()
        )
    b, c = res.mean("baseline"), res.mean("candidate")
    lines.append(f"{'mean':<{width}}  {b:>8.3f}  {c:>9.3f}  {c - b:>+7.3f}")
    lines.append(summary_line(res))
    lines += [f"  - {f}" for f in res.failures]
    lines += [f"  warning: {w}" for w in res.warnings]
    return "\n".join(lines)


def summary_line(res: GateResult) -> str:
    if res.passed:
        return "Result: PASS"
    n = len(res.regressions)
    parts = []
    if n:
        parts.append(f"{n} {'query' if n == 1 else 'queries'} regressed")
    if len(res.failures) > n:
        parts.append("mean below floor")
    return f"Result: FAIL ({', '.join(parts)})"


def to_markdown(res: GateResult) -> str:
    m = res.metric
    b, c = res.mean("baseline"), res.mean("candidate")
    out = [
        f"### Retrieval Regression Gate: {'PASS' if res.passed else 'FAIL'}",
        "",
        f"Metric `{m}@{res.k}`, tolerance {res.tolerance:g}. Mean {b:.3f} -> {c:.3f} ({c - b:+.3f}).",
        "",
        "| Query | Baseline | Candidate | Delta | Unjudged in top k |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in res.queries:
        name = r.query.replace("|", "\\|")
        mark = " **REGRESSED**" if r.regressed else ""
        out.append(
            f"| {name} | {r.baseline[m]:.3f} | {r.candidate[m]:.3f} | {r.delta(m):+.3f}{mark} "
            f"| {int(r.baseline['unjudged'])} -> {int(r.candidate['unjudged'])} |"
        )
    if res.failures:
        out += ["", "**Failures**", ""] + [f"- {f}" for f in res.failures]
    if res.warnings:
        out += ["", "**Warnings**", ""] + [f"- {w}" for w in res.warnings]
    return "\n".join(out) + "\n"


def to_json(res: GateResult) -> str:
    return json.dumps(res.to_dict(), indent=2, sort_keys=True) + "\n"

