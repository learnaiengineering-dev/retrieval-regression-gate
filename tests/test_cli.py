import json
from pathlib import Path

from retrieval_gate.cli import main

EX = Path(__file__).resolve().parent.parent / "examples"
ARGS = [str(EX / "baseline.json"), str(EX / "candidate.json"), str(EX / "judgments.json")]


def test_regression_exits_1_and_names_the_query(capsys):
    assert main(ARGS + ["--k", "3"]) == 1
    out = capsys.readouterr().out
    assert "REGRESSED" in out and "roll back a release" in out and "Result: FAIL" in out


def test_tolerance_makes_it_pass(capsys):
    assert main(ARGS + ["--k", "3", "--tolerance", "0.2"]) == 0
    assert "Result: PASS" in capsys.readouterr().out


def test_quiet_prints_one_line(capsys):
    main(ARGS + ["--k", "3", "--quiet"])
    assert capsys.readouterr().out.strip().startswith("Result: FAIL")


def test_reports_are_written(tmp_path):
    j, m = tmp_path / "r.json", tmp_path / "r.md"
    main(ARGS + ["--k", "3", "--json", str(j), "--markdown", str(m)])
    assert json.loads(j.read_text())["schema_version"] == 1
    assert "REGRESSED" in m.read_text()


def test_github_summary_appends(tmp_path, monkeypatch):
    s = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(s))
    main(ARGS + ["--k", "3", "--github-summary"])
    assert "Retrieval Regression Gate" in s.read_text()


def test_missing_file_exits_2(capsys):
    assert main(["nope.json"] + ARGS[1:]) == 2
    assert "error:" in capsys.readouterr().err


def test_invalid_json_exits_2(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert main([str(bad)] + ARGS[1:]) == 2
    assert "not valid JSON" in capsys.readouterr().err


def test_unwritable_report_path_exits_2(tmp_path, capsys):
    assert main(ARGS + ["--json", str(tmp_path / "missing-dir" / "r.json")]) == 2
    assert "could not write report" in capsys.readouterr().err

