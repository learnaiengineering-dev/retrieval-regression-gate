import json
from pathlib import Path

import pytest

from retrieval_gate import InputError, compare

EX = Path(__file__).resolve().parent.parent / "examples"


def load(name):
    return json.loads((EX / name).read_text())


def run(**kw):
    return compare(load("baseline.json"), load("candidate.json"), load("judgments.json"), **kw)


def test_mean_improves_but_one_query_regresses():
    r = run(k=3)
    assert r.mean("candidate") > r.mean("baseline")
    assert [q.query for q in r.regressions] == ["roll back a release"]
    assert not r.passed


def test_known_mean_ndcg_matches_reference():
    r = run(k=3)
    assert r.mean("baseline") == pytest.approx(0.620, abs=1e-3)
    assert r.mean("candidate") == pytest.approx(0.961, abs=1e-3)


def test_tolerance_absorbs_a_small_drop():
    assert run(k=3, tolerance=0.2).passed
    assert not run(k=3, tolerance=0.1).passed


def test_other_metrics_can_gate():
    # At k=3 the regressed query is unchanged on precision, so a precision gate passes.
    assert run(k=3, metric="precision").passed


def test_min_mean_floor_fails_even_without_regressions():
    r = run(k=3, tolerance=0.2, min_mean=0.99)
    assert not r.regressions and not r.passed
    assert any("below the floor" in f for f in r.failures)


def test_identical_systems_pass():
    b = load("baseline.json")
    assert compare(b, b, load("judgments.json"), k=3).passed


def test_mismatched_queries_rejected_with_detail():
    b = load("baseline.json")
    c = dict(b)
    c.pop("api rate limits")
    with pytest.raises(InputError, match="candidate is missing"):
        compare(b, c, load("judgments.json"))


def test_unjudged_and_empty_judgment_warnings():
    r = compare({"q": ["x"]}, {"q": ["x"]}, {"q": {"y": 0}}, k=1)
    assert any("no relevant judgments" in w for w in r.warnings)
    assert any("unjudged" in w for w in r.warnings)


def test_bad_options_rejected():
    with pytest.raises(InputError):
        run(metric="map")
    with pytest.raises(InputError):
        run(tolerance=-0.1)


def test_report_dict_is_json_serialisable_and_versioned():
    d = run(k=3).to_dict()
    assert d["schema_version"] == 1 and d["passed"] is False
    json.dumps(d)
    assert d["queries"]["roll back a release"]["regressed"] is True

