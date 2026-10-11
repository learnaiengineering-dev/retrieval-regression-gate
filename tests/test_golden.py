"""The Python gate and the browser widget must agree. Both are checked against this golden file."""
import json
from pathlib import Path

import pytest

from retrieval_gate import compare

ROOT = Path(__file__).resolve().parent.parent
CASES = json.loads((ROOT / "tests" / "golden" / "cases.json").read_text())


def load(name):
    return json.loads((ROOT / "examples" / name).read_text())


def close(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            close(a[key], b[key])
    elif isinstance(a, float) or isinstance(b, float):
        assert a == pytest.approx(b, abs=1e-9)
    else:
        assert a == b


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_python_matches_golden(case):
    result = compare(load("baseline.json"), load("candidate.json"), load("judgments.json"), **case["options"])
    close(result.to_dict(), case["expected"])
