/*
 * <retrieval-gate>: embeddable retrieval regression gate.
 * Copyright (c) 2026 Learn AI Engineering. MIT License.
 * Metric definitions follow the Retrieval Evaluation Lab project in
 * https://github.com/rohitg00/ai-engineering-from-scratch (MIT). See NOTICE.
 *
 * One file, no dependencies, no network access, no inline styles or scripts.
 * Usage:
 *   <script type="module" src="retrieval-gate-widget.js"></script>
 *   <retrieval-gate k="3" tolerance="0.02">
 *     <script type="application/json">{"baseline":{},"candidate":{},"judgments":{}}</script>
 *   </retrieval-gate>
 */

export const METRICS = ["precision", "recall", "rr", "ndcg"];
export const SCHEMA_VERSION = 1;
const MAX_GRADE = 3;
const EPS = 1e-9;

export class InputError extends Error {}

const isObject = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const has = (o, k) => Object.prototype.hasOwnProperty.call(o, k);

function validateK(k) {
  if (!Number.isInteger(k) || k < 1) throw new InputError("k must be a positive integer");
  return k;
}

function validateRanking(r, label) {
  if (!Array.isArray(r)) throw new InputError(`${label}: expected a list of document ids`);
  if (r.some((d) => typeof d !== "string" || d === "")) throw new InputError(`${label}: document ids must be non-empty strings`);
  if (new Set(r).size !== r.length) throw new InputError(`${label}: document ids must be unique within a ranking`);
  return r;
}

function validateJudgments(j, label) {
  if (!isObject(j)) throw new InputError(`${label}: expected an object mapping document id to grade`);
  for (const [doc, g] of Object.entries(j)) {
    if (!Number.isInteger(g) || g < 0 || g > MAX_GRADE) throw new InputError(`${label}: grade for '${doc}' must be an integer from 0 to ${MAX_GRADE}`);
  }
  return j;
}

/** Score one query. Unjudged documents get zero gain and are counted in `unjudged`. */
export function scoreQuery(ranking, judgments, k) {
  validateK(k);
  validateRanking(ranking, "ranking");
  validateJudgments(judgments, "judgments");
  const top = ranking.slice(0, k);
  const grade = (d) => (has(judgments, d) ? judgments[d] : 0);
  const grades = Object.values(judgments);
  const relevant = grades.filter((g) => g > 0).length;
  const hits = top.filter((d) => grade(d) > 0).length;
  const first = top.findIndex((d) => grade(d) > 0);
  const dcg = top.reduce((s, d, i) => s + (2 ** grade(d) - 1) / Math.log2(i + 2), 0);
  const ideal = grades
    .slice()
    .sort((a, b) => b - a)
    .slice(0, k)
    .reduce((s, g, i) => s + (2 ** g - 1) / Math.log2(i + 2), 0);
  return {
    precision: hits / k,
    recall: relevant ? hits / relevant : 0,
    rr: first >= 0 ? 1 / (first + 1) : 0,
    ndcg: ideal ? dcg / ideal : 0,
    unjudged: top.filter((d) => !has(judgments, d)).length,
  };
}

/** Compare two systems query by query. Returns the same shape as the Python report. */
export function compare(baseline, candidate, judgments, { k = 10, metric = "ndcg", tolerance = 0, minMean = null } = {}) {
  validateK(k);
  if (!METRICS.includes(metric)) throw new InputError(`metric must be one of ${METRICS.join(", ")}`);
  if (typeof tolerance !== "number" || !(tolerance >= 0)) throw new InputError("tolerance must be a number >= 0");
  if (minMean !== null && (typeof minMean !== "number" || Number.isNaN(minMean))) throw new InputError("min_mean must be a number");
  for (const [label, data] of [["baseline", baseline], ["candidate", candidate], ["judgments", judgments]]) {
    if (!isObject(data)) throw new InputError(`${label}: expected an object keyed by query`);
  }
  const names = { baseline: Object.keys(baseline), candidate: Object.keys(candidate), judgments: Object.keys(judgments) };
  if (!names.judgments.length) throw new InputError("no queries found");
  const every = new Set([...names.baseline, ...names.candidate, ...names.judgments]);
  const missing = Object.entries(names)
    .map(([label, list]) => [label, [...every].filter((q) => !list.includes(q))])
    .filter(([, m]) => m.length);
  if (missing.length) {
    const detail = missing.map(([label, m]) => `${label} is missing ${JSON.stringify(m.sort())}`).join("; ");
    throw new InputError(`baseline, candidate and judgments must cover the same queries: ${detail}`);
  }

  const queries = {};
  const failures = [];
  const warnings = [];
  const rows = [];
  for (const q of names.judgments.slice().sort()) {
    const j = validateJudgments(judgments[q], `judgments['${q}']`);
    const b = scoreQuery(validateRanking(baseline[q], `baseline['${q}']`), j, k);
    const c = scoreQuery(validateRanking(candidate[q], `candidate['${q}']`), j, k);
    const regressed = c[metric] < b[metric] - tolerance - EPS;
    rows.push({ b, c });
    queries[q] = {
      baseline: b,
      candidate: c,
      delta: Object.fromEntries(METRICS.map((m) => [m, c[m] - b[m]])),
      regressed,
    };
    if (!Object.values(j).some((g) => g > 0)) warnings.push(`'${q}' has no relevant judgments, so every score for it is 0`);
    if (c.unjudged) warnings.push(`'${q}': ${c.unjudged} unjudged candidate document(s) in the top ${k} score zero`);
    if (regressed) failures.push(`'${q}': ${metric} ${b[metric].toFixed(3)} -> ${c[metric].toFixed(3)} (${(c[metric] - b[metric]).toFixed(3)})`);
  }
  const mean = (side) => Object.fromEntries(METRICS.map((m) => [m, rows.reduce((s, r) => s + r[side][m], 0) / rows.length]));
  const means = { baseline: mean("b"), candidate: mean("c") };
  if (minMean !== null && means.candidate[metric] < minMean - EPS) {
    failures.push(`mean ${metric} ${means.candidate[metric].toFixed(3)} is below the floor ${minMean.toFixed(3)}`);
  }
  return { schema_version: SCHEMA_VERSION, k, metric, tolerance, min_mean: minMean, passed: failures.length === 0, failures, warnings, mean: means, queries };
}

/* ------------------------------------------------------------------ widget */

const CSS = `
:host{
  --_bg:var(--rg-bg,#ffffff); --_ink:var(--rg-ink,#1a1a1a); --_muted:var(--rg-muted,#6b6b66);
  --_line:var(--rg-line,#e4e4dc); --_accent:var(--rg-accent,#3553ff);
  --_good:var(--rg-good,#2f9e44); --_bad:var(--rg-bad,#d9480f);
  display:block; color:var(--_ink); background:var(--_bg); border:1px solid var(--_line);
  border-radius:var(--rg-radius,12px); padding:14px; font:15px/1.5 var(--rg-font,system-ui,-apple-system,"Segoe UI",sans-serif);
  box-sizing:border-box;
}
@media (prefers-color-scheme:dark){
  :host(:not([theme="light"])){
    --_bg:var(--rg-bg,#1a1a1f); --_ink:var(--rg-ink,#ececf0); --_muted:var(--rg-muted,#9a9aa5);
    --_line:var(--rg-line,#2c2c34); --_accent:var(--rg-accent,#8097ff);
    --_good:var(--rg-good,#5fd47a); --_bad:var(--rg-bad,#ff8a5c);
  }
}
:host([theme="dark"]){
  --_bg:var(--rg-bg,#1a1a1f); --_ink:var(--rg-ink,#ececf0); --_muted:var(--rg-muted,#9a9aa5);
  --_line:var(--rg-line,#2c2c34); --_accent:var(--rg-accent,#8097ff);
  --_good:var(--rg-good,#5fd47a); --_bad:var(--rg-bad,#ff8a5c);
}
*{box-sizing:border-box}
h2{margin:0 0 8px;font-size:1rem}
.banner{border:1px solid var(--_line);border-radius:10px;padding:10px 12px;margin-bottom:10px}
.banner.fail{border-color:var(--_bad)}
.banner.pass{border-color:var(--_good)}
.banner b{display:block}
.banner small{color:var(--_muted)}
.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:.9em}
th,td{padding:6px 8px;text-align:right;border-bottom:1px solid var(--_line);white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal;min-width:7em}
th{font-size:.75em;text-transform:uppercase;letter-spacing:.05em;color:var(--_muted);font-weight:600}
tr.mean td{font-weight:700;border-top:2px solid var(--_line)}
.up{color:var(--_good)} .down{color:var(--_bad)} .flat{color:var(--_muted)}
.tag{margin-left:6px;font-size:.75em;font-weight:700;color:var(--_bad)}
ul{margin:10px 0 0;padding-left:18px;color:var(--_muted);font-size:.85em}
details{margin-top:10px}
summary{cursor:pointer;color:var(--_muted);font-size:.85em}
label{display:block;font-size:.8em;font-weight:600;margin:8px 0 2px}
textarea{width:100%;height:110px;font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--_ink);background:var(--_bg);border:1px solid var(--_line);border-radius:8px;padding:6px;resize:vertical}
textarea:focus-visible{outline:2px solid var(--_accent);outline-offset:1px}
.error{color:var(--_bad);font-size:.85em;margin:6px 0 0}
`;

let sheet = null;
function applyStyles(root) {
  // Constructed stylesheets are not blocked by a strict style-src Content-Security-Policy.
  if (typeof CSSStyleSheet !== "undefined" && "adoptedStyleSheets" in root) {
    if (!sheet) {
      sheet = new CSSStyleSheet();
      sheet.replaceSync(CSS);
    }
    root.adoptedStyleSheets = [sheet];
  } else {
    const style = document.createElement("style");
    style.textContent = CSS;
    root.appendChild(style);
  }
}

const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};
const fmt = (v) => v.toFixed(3);
function deltaCell(d) {
  const td = el("td");
  if (Math.abs(d) < 0.0005) { td.className = "flat"; td.textContent = "±0.000"; }
  else { td.className = d > 0 ? "up" : "down"; td.textContent = (d > 0 ? "+" : "") + d.toFixed(3); }
  return td;
}

if (typeof HTMLElement !== "undefined" && typeof customElements !== "undefined") {
  class RetrievalGate extends HTMLElement {
    static observedAttributes = ["k", "metric", "tolerance", "min-mean", "editable", "theme", "heading"];

    #data = null;
    #result = null;
    #initError = null;
    #root;
    #host = null;
    #editorError = null;

    constructor() {
      super();
      this.#root = this.attachShadow({ mode: "open" });
      applyStyles(this.#root);
    }

    connectedCallback() {
      if (!this.#data) {
        const script = this.querySelector('script[type="application/json"]');
        if (script) {
          try { this.#data = JSON.parse(script.textContent); }
          catch (e) { this.#initError = `Could not read the embedded JSON: ${e.message}`; }
        }
      }
      this.#build();
    }

    attributeChangedCallback() { if (this.isConnected) this.#build(); }

    /** Set {baseline, candidate, judgments} from JavaScript. */
    set data(value) { this.#data = value; this.#initError = null; if (this.isConnected) this.#build(); }
    get data() { return this.#data; }
    /** The last result, in the same shape as the Python report. */
    get result() { return this.#result; }

    #options() {
      const num = (name) => Number(this.getAttribute(name));
      return {
        k: this.hasAttribute("k") ? num("k") : 10,
        metric: this.getAttribute("metric") || "ndcg",
        tolerance: this.hasAttribute("tolerance") ? num("tolerance") : 0,
        minMean: this.hasAttribute("min-mean") ? num("min-mean") : null,
      };
    }

    // Builds the static shell (heading, result area, optional editor) once per attribute change.
    #build() {
      for (const n of [...this.#root.childNodes]) if (n.nodeType === 1 && n.tagName !== "STYLE") n.remove();
      const wrap = el("div");
      const heading = this.getAttribute("heading");
      if (heading) wrap.appendChild(el("h2", "", heading));
      this.#host = el("div");
      wrap.appendChild(this.#host);
      if (this.hasAttribute("editable")) wrap.appendChild(this.#buildEditor());
      this.#root.appendChild(wrap);
      this.#update();
    }

    // Re-renders only the result area, so editor textareas keep focus while typing.
    #update() {
      this.#host.replaceChildren();
      this.#result = null;
      let error = this.#initError;
      if (!error) {
        if (!this.#data) error = 'No data. Add a <script type="application/json"> child or set the .data property.';
        else {
          try { this.#result = compare(this.#data.baseline, this.#data.candidate, this.#data.judgments, this.#options()); }
          catch (e) { if (e instanceof InputError) error = e.message; else throw e; }
        }
      }
      if (error) {
        const p = el("p", "error", error);
        p.setAttribute("role", "alert");
        this.#host.appendChild(p);
      } else {
        this.#renderResult(this.#host, this.#result);
      }
      this.dispatchEvent(new CustomEvent("gate-result", { detail: this.#result, bubbles: true, composed: true }));
    }

    #renderResult(wrap, res) {
      const m = res.metric;
      const mb = res.mean.baseline[m], mc = res.mean.candidate[m];
      const regs = Object.values(res.queries).filter((v) => v.regressed).length;
      const banner = el("div", `banner ${res.passed ? "pass" : "fail"}`);
      banner.setAttribute("role", "status");
      banner.appendChild(el("b", "", res.passed ? "Gate: PASS" : "Gate: FAIL"));
      banner.appendChild(el("small", "", res.passed
        ? `No query got worse on ${m}@${res.k}. Mean ${fmt(mb)} to ${fmt(mc)}.`
        : `${regs ? `${regs} ${regs === 1 ? "query" : "queries"} regressed. ` : ""}Mean ${fmt(mb)} to ${fmt(mc)}.`));
      wrap.appendChild(banner);

      const scroll = el("div", "scroll");
      const table = el("table");
      const head = el("tr");
      for (const h of [`Query (${m}@${res.k})`, "Before", "After", "Change"]) head.appendChild(el("th", "", h));
      table.appendChild(head);
      for (const [q, v] of Object.entries(res.queries)) {
        const tr = el("tr");
        const name = el("td", "", q);
        if (v.regressed) name.appendChild(el("span", "tag", "REGRESSED"));
        tr.append(name, el("td", "", fmt(v.baseline[m])), el("td", "", fmt(v.candidate[m])), deltaCell(v.delta[m]));
        table.appendChild(tr);
      }
      const mean = el("tr", "mean");
      mean.append(el("td", "", "Mean"), el("td", "", fmt(mb)), el("td", "", fmt(mc)), deltaCell(mc - mb));
      table.appendChild(mean);
      scroll.appendChild(table);
      wrap.appendChild(scroll);

      const notes = [...res.failures.filter((f) => f.includes("below the floor")), ...res.warnings];
      if (notes.length) {
        const ul = el("ul");
        for (const n of notes) ul.appendChild(el("li", "", n));
        wrap.appendChild(ul);
      }
    }

    #buildEditor() {
      const details = el("details");
      details.appendChild(el("summary", "", "Edit the data"));
      const areas = {};
      for (const key of ["baseline", "candidate", "judgments"]) {
        const id = `rg-${key}`;
        const label = el("label", "", key);
        label.htmlFor = id;
        const ta = el("textarea");
        ta.id = id;
        ta.spellcheck = false;
        ta.value = JSON.stringify(this.#data ? this.#data[key] : {}, null, 2);
        areas[key] = ta;
        details.append(label, ta);
      }
      this.#editorError = el("p", "error");
      this.#editorError.setAttribute("role", "alert");
      details.appendChild(this.#editorError);
      const onInput = () => {
        try {
          this.#data = Object.fromEntries(Object.entries(areas).map(([k, ta]) => [k, JSON.parse(ta.value)]));
          this.#initError = null;
          this.#editorError.textContent = "";
          this.#update();
        } catch (e) {
          this.#editorError.textContent = `Invalid JSON: ${e.message}`;
        }
      };
      for (const ta of Object.values(areas)) ta.addEventListener("input", onInput);
      return details;
    }
  }

  if (!customElements.get("retrieval-gate")) customElements.define("retrieval-gate", RetrievalGate);
}
