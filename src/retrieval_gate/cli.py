"""Command line entry point. Exit codes: 0 pass, 1 gate failed, 2 bad input."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .compare import compare
from .metrics import METRICS, InputError
from .report import summary_line, to_json, to_markdown, to_text


def _load(path: str):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as e:
        raise InputError(f"{path}: {e.strerror or e}") from e
    except json.JSONDecodeError as e:
        raise InputError(f"{path}: not valid JSON ({e})") from e


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="retrieval-gate",
        description="Fail when any retrieval query gets worse. Exit 0 pass, 1 gate failed, 2 bad input.",
    )
    p.add_argument("baseline", help="JSON: query -> ranked document ids (current system)")
    p.add_argument("candidate", help="JSON: query -> ranked document ids (new system)")
    p.add_argument("judgments", help="JSON: query -> document id -> grade 0..3")
    p.add_argument("--k", type=int, default=10, help="cutoff (default 10)")
    p.add_argument("--metric", choices=METRICS, default="ndcg", help="gate metric (default ndcg)")
    p.add_argument("--tolerance", type=float, default=0.0, help="allowed per-query drop (default 0)")
    p.add_argument("--min-mean", type=float, default=None, help="also fail if the candidate mean is below this")
    p.add_argument("--json", metavar="PATH", help="write the full JSON report")
    p.add_argument("--markdown", metavar="PATH", help="write a Markdown summary")
    p.add_argument("--github-summary", action="store_true", help="append the Markdown to $GITHUB_STEP_SUMMARY")
    p.add_argument("--quiet", action="store_true", help="print only the result line")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        res = compare(
            _load(args.baseline),
            _load(args.candidate),
            _load(args.judgments),
            k=args.k,
            metric=args.metric,
            tolerance=args.tolerance,
            min_mean=args.min_mean,
        )
        if args.json:
            Path(args.json).write_text(to_json(res), encoding="utf-8")
        if args.markdown:
            Path(args.markdown).write_text(to_markdown(res), encoding="utf-8")
        if args.github_summary:
            target = os.environ.get("GITHUB_STEP_SUMMARY")
            if target:
                with open(target, "a", encoding="utf-8") as fh:
                    fh.write(to_markdown(res))
    except InputError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"error: could not write report: {e.strerror or e}", file=sys.stderr)
        return 2
    print(summary_line(res) if args.quiet else to_text(res))
    return 0 if res.passed else 1

