// Run with: node --test tests/js
// The widget's gate logic must match the Python implementation, via the shared golden file.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { compare, scoreQuery, InputError } from "../../docs/retrieval-gate-widget.js";

const read = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url), "utf8"));
const baseline = read("../../examples/baseline.json");
const candidate = read("../../examples/candidate.json");
const judgments = read("../../examples/judgments.json");
const cases = read("../golden/cases.json");

function close(actual, expected, path = "") {
  if (expected !== null && typeof expected === "object" && !Array.isArray(expected)) {
    assert.deepEqual(Object.keys(actual).sort(), Object.keys(expected).sort(), `keys at ${path}`);
    for (const key of Object.keys(expected)) close(actual[key], expected[key], `${path}.${key}`);
  } else if (typeof expected === "number") {
    assert.ok(Math.abs(actual - expected) < 1e-9, `${path}: ${actual} vs ${expected}`);
  } else if (Array.isArray(expected)) {
    // Failure and warning wording differs slightly between languages; the count must match.
    assert.equal(actual.length, expected.length, `length at ${path}`);
  } else {
    assert.equal(actual, expected, path);
  }
}

for (const c of cases) {
  test(`matches Python: ${c.name}`, () => {
    const { k, metric, tolerance, min_mean: minMean } = c.options;
    const result = compare(baseline, candidate, judgments, { k, metric, tolerance, minMean: minMean ?? null });
    close(result, c.expected);
  });
}

test("a regression is flagged even though the mean improves", () => {
  const r = compare(baseline, candidate, judgments, { k: 3 });
  assert.equal(r.passed, false);
  assert.ok(r.mean.candidate.ndcg > r.mean.baseline.ndcg);
  assert.deepEqual(Object.entries(r.queries).filter(([, v]) => v.regressed).map(([q]) => q), ["roll back a release"]);
});

test("unjudged documents score zero and are counted", () => {
  const s = scoreQuery(["forum", "runbook"], { runbook: 3 }, 2);
  assert.equal(s.unjudged, 1);
  assert.equal(s.precision, 0.5);
});

test("an id like 'constructor' is not mistaken for a judged document", () => {
  const s = scoreQuery(["constructor", "toString"], { a: 3 }, 2);
  assert.equal(s.unjudged, 2);
  assert.equal(s.precision, 0);
});

test("bad input raises InputError", () => {
  assert.throws(() => compare(baseline, candidate, judgments, { k: 0 }), InputError);
  assert.throws(() => compare(baseline, candidate, judgments, { metric: "map" }), InputError);
  assert.throws(() => compare(baseline, { ...candidate, extra: [] }, judgments), /same queries/);
  assert.throws(() => scoreQuery(["a", "a"], { a: 1 }, 2), InputError);
  assert.throws(() => scoreQuery(["a"], { a: 4 }, 1), InputError);
});
