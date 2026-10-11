import pytest

from retrieval_gate.metrics import InputError, score_query

J = {"runbook": 3, "faq": 2, "notes": 1, "blog": 0}


def test_perfect_ranking():
    s = score_query(["runbook", "faq", "notes"], J, 3)
    assert s["ndcg"] == pytest.approx(1.0)
    assert s["precision"] == pytest.approx(1.0)
    assert s["recall"] == pytest.approx(1.0)
    assert s["rr"] == 1.0


def test_known_values_match_reference_definition():
    # Values checked against the Retrieval Evaluation Lab reference implementation.
    s = score_query(["notes", "blog", "runbook", "faq"], J, 3)
    assert s["ndcg"] == pytest.approx(0.479, abs=1e-3)
    assert s["precision"] == pytest.approx(2 / 3)
    assert s["recall"] == pytest.approx(2 / 3)
    assert s["rr"] == 1.0


def test_unjudged_documents_score_zero_and_are_counted():
    s = score_query(["forum", "runbook"], J, 2)
    assert s["unjudged"] == 1
    assert s["precision"] == pytest.approx(0.5)


def test_rr_is_zero_without_a_relevant_hit_in_top_k():
    assert score_query(["blog", "forum", "runbook"], J, 2)["rr"] == 0.0


def test_no_relevant_judgments_scores_zero_without_dividing_by_zero():
    s = score_query(["a"], {"a": 0}, 1)
    assert s["ndcg"] == 0.0 and s["recall"] == 0.0


@pytest.mark.parametrize("k", [0, -1, True, 1.5, "3"])
def test_bad_k_rejected(k):
    with pytest.raises(InputError):
        score_query(["a"], {"a": 1}, k)


def test_duplicate_ids_rejected():
    with pytest.raises(InputError):
        score_query(["a", "a"], {"a": 1}, 2)


@pytest.mark.parametrize("grade", [4, -1, 1.5, True, "2"])
def test_bad_grades_rejected(grade):
    with pytest.raises(InputError):
        score_query(["a"], {"a": grade}, 1)

