"""Build the HTML sweep report from the tables DeformationSweepReport.ipynb writes.

    jupyter nbconvert --to notebook --execute --inplace DeformationSweepReport.ipynb
    python make_sweep_report.py            # -> sweep_results/DeformationSweepReport.html

Every number on the page comes from sweep_results/*.csv / summary.json. The prose
that interprets them lives in FINDINGS / RECOMMENDATION_NOTES below.
"""

import html
import json
import os

import numpy as np
import pandas as pd

SWEEP_DIR = "sweep_results"
OUT = os.path.join(SWEEP_DIR, "DeformationSweepReport.html")
FAMILIES = ["translation", "rotation", "rotation+translation", "affine (shear)", "nonrigid", "breath hold"]
TREES = ["non-planar", "planar"]
GENERATIONS = list(range(6))
PASS_TLD = 95.0
SHORT = {"maximum_pairing_distance": "Max pairing distance", "minimum_landmark_distance": "Min landmark distance",
         "sensitivity_multiplier": "Sensitivity multiplier", "projection_neighbors": "Projection neighbors"}
FLAG = {"maximum_pairing_distance": "--maximumPairingDistance", "minimum_landmark_distance": "--minimumLandmarkDistance",
        "sensitivity_multiplier": "--sensitivityMultiplier", "projection_neighbors": "--projectionNeighbors"}
MAIN_EFFECT_NAME = {"pairing": "maximum_pairing_distance", "min landmark": "minimum_landmark_distance",
                    "sensitivity": "sensitivity_multiplier", "neighbors": "projection_neighbors"}

# Interpretation written after reading the results; numbers are filled from the data.
FINDINGS = []  # list of (title, html) set in main() from the data
RECOMMENDATION_NOTES = []


def esc(text):
    return html.escape(str(text), quote=True)


def fmt(value, digits=1, suffix=""):
    return "–" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.{digits}f}{suffix}"


# -------------------------------------------------------------------------
# SVG helpers (static charts; the page script only adds tooltips and toggles)
# -------------------------------------------------------------------------

class Scale:
    def __init__(self, domain, range_):
        self.d0, self.d1 = domain
        self.r0, self.r1 = range_

    def __call__(self, v):
        return self.r0 + (v - self.d0) / (self.d1 - self.d0) * (self.r1 - self.r0)


def nice_ticks(lo, hi, count=4):
    span = hi - lo
    step = 10 ** np.floor(np.log10(span / count))
    for m in (1, 2, 2.5, 5, 10):
        if span / (m * step) <= count:
            step *= m
            break
    start = np.ceil(lo / step) * step
    return [round(t, 6) for t in np.arange(start, hi + step * 1e-9, step)]


def line_panel(title, series, x_label, y_domain=(0, 100), reference=None, width=320, height=210, x_ticks=None):
    """series: list of dict(name, cls, points=[(x, y, tip)], dash=bool). y is a percentage."""
    m = dict(l=40, r=12, t=26, b=38)
    xs = [p[0] for s in series for p in s["points"]]
    x_hi = max(xs) if xs else 1
    x = Scale((0, x_hi * 1.04 or 1), (m["l"], width - m["r"]))
    y = Scale(y_domain, (height - m["b"], m["t"]))
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
             f'<text class="panel-title" x="{m["l"]}" y="15">{esc(title)}</text>']
    for t in nice_ticks(*y_domain, count=4):
        parts.append(f'<line class="grid" x1="{m["l"]}" x2="{width - m["r"]}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>'
                     f'<text class="tick" x="{m["l"] - 6}" y="{y(t) + 3.5:.1f}" text-anchor="end">{t:g}</text>')
    for t in (x_ticks or nice_ticks(0, x_hi, count=4)):
        if t <= x_hi * 1.04:
            parts.append(f'<text class="tick" x="{x(t):.1f}" y="{height - m["b"] + 14}" text-anchor="middle">{t:g}</text>')
    parts.append(f'<line class="axis" x1="{m["l"]}" x2="{width - m["r"]}" y1="{height - m["b"]}" y2="{height - m["b"]}"/>')
    parts.append(f'<text class="axis-label" x="{(m["l"] + width - m["r"]) / 2}" y="{height - 6}" text-anchor="middle">{esc(x_label)}</text>')
    if reference is not None:
        parts.append(f'<line class="reference" x1="{m["l"]}" x2="{width - m["r"]}" y1="{y(reference):.1f}" y2="{y(reference):.1f}"/>')
    for s in series:
        pts = [(x(px), y(py), tip) for px, py, tip in s["points"] if py is not None and not np.isnan(py)]
        if len(pts) > 1:
            d = " ".join(f"{'M' if i == 0 else 'L'}{px:.1f},{py:.1f}" for i, (px, py, _) in enumerate(pts))
            parts.append(f'<path class="line {s["cls"]}{" dashed" if s.get("dash") else ""}" d="{d}"/>')
        for px, py, tip in pts:
            parts.append(f'<circle class="dot {s["cls"]}" cx="{px:.1f}" cy="{py:.1f}" r="3.2"/>'
                         f'<circle class="hit" cx="{px:.1f}" cy="{py:.1f}" r="9" data-tip="{esc(tip)}"/>')
        for px, py, tip in s.get("errors", []):
            parts.append(f'<g class="err-mark" data-tip="{esc(tip)}"><path d="M{x(px) - 4:.1f},{y(py) - 4:.1f}l8,8m0,-8l-8,8"/>'
                         f'<circle class="hit" cx="{x(px):.1f}" cy="{y(py):.1f}" r="9"/></g>')
    parts.append("</svg>")
    return "".join(parts)


def category_panel(title, categories, series, y_domain, y_format, width=250, height=200):
    """Points at categorical x positions; series: list of dict(name, cls, values, tips)."""
    m = dict(l=40, r=12, t=26, b=30)
    step = (width - m["l"] - m["r"]) / len(categories)
    xs = [m["l"] + step * (i + 0.5) for i in range(len(categories))]
    y = Scale(y_domain, (height - m["b"], m["t"]))
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
             f'<text class="panel-title" x="{m["l"]}" y="15">{esc(title)}</text>']
    for t in nice_ticks(*y_domain, count=4):
        parts.append(f'<line class="grid" x1="{m["l"]}" x2="{width - m["r"]}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>'
                     f'<text class="tick" x="{m["l"] - 6}" y="{y(t) + 3.5:.1f}" text-anchor="end">{y_format(t)}</text>')
    for cx, c in zip(xs, categories):
        parts.append(f'<text class="tick" x="{cx:.1f}" y="{height - m["b"] + 14}" text-anchor="middle">{esc(c)}</text>')
    parts.append(f'<line class="axis" x1="{m["l"]}" x2="{width - m["r"]}" y1="{height - m["b"]}" y2="{height - m["b"]}"/>')
    for s in series:
        pts = [(cx, y(v), tip) for cx, v, tip in zip(xs, s["values"], s["tips"])]
        d = " ".join(f"{'M' if i == 0 else 'L'}{px:.1f},{py:.1f}" for i, (px, py, _) in enumerate(pts))
        parts.append(f'<path class="line {s["cls"]}" d="{d}"/>')
        for px, py, tip in pts:
            parts.append(f'<circle class="dot {s["cls"]}" cx="{px:.1f}" cy="{py:.1f}" r="4"/>'
                         f'<circle class="hit" cx="{px:.1f}" cy="{py:.1f}" r="10" data-tip="{esc(tip)}"/>')
    parts.append("</svg>")
    return "".join(parts)


RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
        "#1c5cab", "#184f95", "#104281", "#0d366b"]


def ramp_color(value, lo=50.0, hi=100.0):
    if value is None or np.isnan(value):
        return None
    t = float(np.clip((value - lo) / (hi - lo), 0, 1))
    return RAMP[int(round(t * (len(RAMP) - 1)))]


def heatmap(title, rows, width=430):
    """rows: list of (label, [values per generation], [tips])."""
    m = dict(l=170, r=8, t=30, b=8)
    cell_w = (width - m["l"] - m["r"]) / len(GENERATIONS)
    cell_h = 15
    height = m["t"] + cell_h * len(rows) + m["b"]
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
             f'<text class="panel-title" x="0" y="13">{esc(title)}</text>']
    for g in GENERATIONS:
        parts.append(f'<text class="tick" x="{m["l"] + cell_w * (g + 0.5):.1f}" y="{m["t"] - 5}" text-anchor="middle">gen {g}</text>')
    for i, (label, values, tips) in enumerate(rows):
        top = m["t"] + i * cell_h
        parts.append(f'<text class="row-label" x="{m["l"] - 6}" y="{top + cell_h - 4}" text-anchor="end">{esc(label)}</text>')
        for g, (v, tip) in enumerate(zip(values, tips)):
            left = m["l"] + g * cell_w
            color = ramp_color(v)
            if color is None:
                parts.append(f'<rect class="cell-error" x="{left + 1:.1f}" y="{top + 1}" width="{cell_w - 2:.1f}" height="{cell_h - 2}" rx="2" data-tip="{esc(tip)}"/>'
                             f'<text class="cell-text muted" x="{left + cell_w / 2:.1f}" y="{top + cell_h - 4}" text-anchor="middle">err</text>')
                continue
            parts.append(f'<rect x="{left + 1:.1f}" y="{top + 1}" width="{cell_w - 2:.1f}" height="{cell_h - 2}" rx="2" fill="{color}" data-tip="{esc(tip)}"/>')
            if v < 99.95:  # label only the cells that lost a branch
                ink = "#ffffff" if v >= 75 else "#0b0b0b"
                parts.append(f'<text class="cell-text" fill="{ink}" x="{left + cell_w / 2:.1f}" y="{top + cell_h - 4}" text-anchor="middle">{v:.0f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def ramp_legend():
    stops = "".join(f'<span style="background:{c}"></span>' for c in RAMP)
    return (f'<div class="ramp-legend"><span class="ramp-label">≤50%</span><span class="ramp">{stops}</span>'
            f'<span class="ramp-label">100% of the generation\'s branches detected</span></div>')


def table(frame, formats=None, classes="", highlight=None):
    formats = formats or {}
    head = "".join(f"<th>{esc(c)}</th>" for c in frame.columns)
    body = []
    for _, row in frame.iterrows():
        cells = []
        for c in frame.columns:
            v = row[c]
            text = formats[c](v) if c in formats else ("–" if pd.isna(v) else str(v))
            cls = highlight(c, v) if highlight else ""
            cells.append(f'<td class="{cls}">{esc(text)}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="table-wrap"><table class="{classes}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


# -------------------------------------------------------------------------
# Page
# -------------------------------------------------------------------------

CSS = """
/* Layout: one reading column, figures break wider on large screens; data in tables and small multiples. */
:root {
  --surface: #fcfcfb; --plane: #f4f5f3; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --hairline: rgba(11,11,11,0.10);
  --s-best: #2a78d6; --s-default: #eb6834; --s-oracle: #1baf7a; --s-none: #898781;
  --pass-bg: #e3f1e3; --pass-ink: #006300; --fail-bg: #fbe4e3; --fail-ink: #a12a2a;
  --accent: #184f95;
  --font-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-data: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface: #1a1a19; --plane: #121211; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #9a988f;
    --grid: #2c2c2a; --axis: #383835; --hairline: rgba(255,255,255,0.10);
    --s-best: #3987e5; --s-default: #d95926; --s-oracle: #199e70; --s-none: #9a988f;
    --pass-bg: #16301a; --pass-ink: #5fd35f; --fail-bg: #3a1c1c; --fail-ink: #ff9b9b; --accent: #86b6ef;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface: #1a1a19; --plane: #121211; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #9a988f;
  --grid: #2c2c2a; --axis: #383835; --hairline: rgba(255,255,255,0.10);
  --s-best: #3987e5; --s-default: #d95926; --s-oracle: #199e70; --s-none: #9a988f;
  --pass-bg: #16301a; --pass-ink: #5fd35f; --fail-bg: #3a1c1c; --fail-ink: #ff9b9b; --accent: #86b6ef;
}
body { background: var(--plane); color: var(--ink); font: 15px/1.6 var(--font-body); padding-inline: 20px; }
main { max-width: 1080px; margin: 0 auto; padding-block: 40px 72px; }
.prose { max-width: 70ch; }
h1 { font-size: 2.1rem; line-height: 1.15; margin: 0 0 10px; font-weight: 600; text-wrap: balance; letter-spacing: -0.01em; }
h2 { font-size: 1.35rem; margin: 52px 0 10px; font-weight: 600; text-wrap: balance; }
h3 { font-size: 1.02rem; margin: 26px 0 6px; font-weight: 600; }
p { margin: 0 0 12px; }
.eyebrow { font: 500 0.72rem/1 var(--font-data); letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-2); margin-bottom: 14px; }
.lede { font-size: 1.08rem; color: var(--ink-2); max-width: 68ch; }
.meta { font: 0.78rem/1.5 var(--font-data); color: var(--muted); margin-top: 14px; }
code, .mono { font-family: var(--font-data); font-size: 0.86em; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 28px; }
.tile { background: var(--surface); border: 1px solid var(--hairline); border-radius: 8px; padding: 14px 16px; display: flex; flex-direction: column; gap: 4px; }
.tile .label { font-size: 0.8rem; color: var(--ink-2); }
.tile .value { font-size: 1.55rem; font-weight: 600; line-height: 1.2; }
.tile .from { font: 0.78rem/1.4 var(--font-data); color: var(--muted); }
.figure { background: var(--surface); border: 1px solid var(--hairline); border-radius: 8px; padding: 16px; margin: 16px 0 8px; }
.figure-head { display: flex; flex-wrap: wrap; gap: 10px 20px; align-items: center; justify-content: space-between; margin-bottom: 8px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 0.82rem; color: var(--ink-2); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 18px; height: 0; border-top: 2px solid; display: inline-block; }
.swatch.dashed { border-top-style: dashed; }
.caption { font-size: 0.84rem; color: var(--ink-2); margin-top: 8px; max-width: 80ch; }
.panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 8px 14px; }
.panels.four { grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }
.panels.two { grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); }
svg { width: 100%; height: auto; display: block; overflow: visible; }
svg text { font-family: var(--font-body); }
.panel-title { font-size: 12px; font-weight: 600; fill: var(--ink); }
.tick { font-size: 10px; fill: var(--muted); font-family: var(--font-data); font-variant-numeric: tabular-nums; }
.axis-label, .row-label { font-size: 10px; fill: var(--ink-2); }
.grid { stroke: var(--grid); stroke-width: 1; }
.axis { stroke: var(--axis); stroke-width: 1; }
.reference { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 2 3; }
.line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.line.dashed { stroke-dasharray: 4 3; }
.dot { stroke: var(--surface); stroke-width: 1.5; }
.best { stroke: var(--s-best); } .dot.best { fill: var(--s-best); stroke: var(--surface); }
.default { stroke: var(--s-default); } .dot.default { fill: var(--s-default); stroke: var(--surface); }
.oracle { stroke: var(--s-oracle); } .dot.oracle { fill: var(--s-oracle); stroke: var(--surface); }
.none { stroke: var(--s-none); } .dot.none { fill: var(--s-none); stroke: var(--surface); }
.swatch.best { border-color: var(--s-best); } .swatch.default { border-color: var(--s-default); }
.swatch.oracle { border-color: var(--s-oracle); } .swatch.none { border-color: var(--s-none); }
.hit { fill: transparent; cursor: default; }
.err-mark path { stroke: var(--ink); stroke-width: 1.6; fill: none; }
.cell-text { font-size: 9px; font-family: var(--font-data); pointer-events: none; }
.cell-text.muted { fill: var(--muted); }
.cell-error { fill: var(--grid); }
.ramp-legend { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; font-size: 0.8rem; color: var(--ink-2); margin-top: 6px; }
.ramp { display: inline-flex; } .ramp span { width: 16px; height: 10px; }
.toggle { display: inline-flex; border: 1px solid var(--hairline); border-radius: 6px; overflow: hidden; }
.toggle button { font: 500 0.8rem var(--font-body); color: var(--ink-2); background: transparent; border: 0; padding: 6px 12px; cursor: pointer; }
.toggle button[aria-pressed="true"] { background: var(--accent); color: var(--surface); }
.toggle button:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.table-wrap { overflow-x: auto; margin: 12px 0; background: var(--surface); border: 1px solid var(--hairline); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 0.84rem; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--hairline); white-space: nowrap; }
th { font-weight: 600; color: var(--ink-2); font-size: 0.76rem; letter-spacing: 0.02em; }
td { font-variant-numeric: tabular-nums; }
tbody tr:last-child td { border-bottom: 0; }
td.pass { background: var(--pass-bg); color: var(--pass-ink); font-weight: 600; }
td.fail { background: var(--fail-bg); color: var(--fail-ink); }
td.strong { font-weight: 600; }
tr.is-default td { background: color-mix(in srgb, var(--s-default) 9%, transparent); }
tr.is-best td { background: color-mix(in srgb, var(--s-best) 11%, transparent); }
.findings { display: grid; gap: 14px; margin-top: 12px; }
.finding { border-left: 3px solid var(--accent); padding: 2px 0 2px 14px; max-width: 78ch; }
.finding h3 { margin: 0 0 4px; }
.recommend { background: var(--surface); border: 1px solid var(--hairline); border-radius: 8px; padding: 18px 20px; }
.flags { font-family: var(--font-data); font-size: 0.88rem; background: var(--plane); border-radius: 6px; padding: 10px 12px; overflow-x: auto; white-space: pre; }
#tip { position: fixed; pointer-events: none; background: var(--ink); color: var(--surface); font: 0.76rem/1.45 var(--font-body); padding: 6px 9px; border-radius: 6px; max-width: 280px; white-space: pre-line; z-index: 10; }
[data-tree][hidden] { display: none !important; }
@media (prefers-reduced-motion: no-preference) { .toggle button { transition: background 120ms; } }
"""

SCRIPT = """
(() => {
  const tip = document.getElementById('tip');
  const show = (el, x, y) => {
    tip.textContent = el.getAttribute('data-tip');
    tip.hidden = false;
    const r = tip.getBoundingClientRect();
    tip.style.left = Math.min(x + 14, innerWidth - r.width - 8) + 'px';
    tip.style.top = Math.max(8, y - r.height - 10) + 'px';
  };
  document.addEventListener('pointermove', e => {
    const el = e.target.closest('[data-tip]');
    if (el) show(el, e.clientX, e.clientY); else tip.hidden = true;
  });
  document.addEventListener('pointerleave', () => { tip.hidden = true; });
  document.querySelectorAll('.toggle').forEach(group => {
    group.addEventListener('click', e => {
      const button = e.target.closest('button');
      if (!button) return;
      const tree = button.dataset.value;
      const scope = document.getElementById(group.dataset.scope);
      group.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
      scope.querySelectorAll('[data-tree]').forEach(el => { el.hidden = el.dataset.tree !== tree; });
    });
  });
})();
"""


def tree_toggle(scope):
    buttons = "".join(f'<button type="button" data-value="{t}" aria-pressed="{str(t == TREES[0]).lower()}">{t} tree</button>'
                      for t in TREES)
    return f'<div class="toggle" role="group" aria-label="Tree" data-scope="{scope}">{buttons}</div>'


def main():
    summary = json.load(open(os.path.join(SWEEP_DIR, "summary.json")))
    curves = pd.read_csv(os.path.join(SWEEP_DIR, "default_vs_best.csv"))
    limits = pd.read_csv(os.path.join(SWEEP_DIR, "recoverable_limits.csv"))
    effects = pd.read_csv(os.path.join(SWEEP_DIR, "main_effects.csv"))
    ranking = pd.read_csv(os.path.join(SWEEP_DIR, "parameter_ranking.csv"), header=[0, 1], index_col=[0, 1, 2, 3, 4])
    generations = pd.read_csv(os.path.join(SWEEP_DIR, "generation_bd.csv"))
    generation_summary = pd.read_csv(os.path.join(SWEEP_DIR, "generation_summary.csv"))
    thresholds = pd.read_csv(os.path.join(SWEEP_DIR, "detection_threshold.csv"))
    raw = pd.read_csv(os.path.join(SWEEP_DIR, "results.csv"))
    deformations = pd.read_csv(os.path.join(SWEEP_DIR, "deformations.csv"))
    notes = json.load(open(os.path.join(SWEEP_DIR, "report_notes.json")))

    best = summary["best_params"]
    rank = ranking.reset_index()
    rank.columns = [a if not b or b.startswith("Unnamed") else f"{a}|{b}" for a, b in rank.columns]
    best_row = rank[rank["combo"] == summary["best_combo"]].iloc[0]
    default_row = rank[rank["combo"] == summary["default_combo"]].iloc[0]
    n_deformations = int((deformations["family"] != "identity").sum() / len(TREES))

    def curve_rows(tree):
        return curves[curves["tree"] == tree]

    np_curves = curve_rows("non-planar")
    passes_default = int(np_curves["pass_default"].sum())
    passes_best = int(np_curves["pass_best"].sum())

    # ---------------- header + tiles ----------------
    parts = [f"<title>Airway Deformation Sweep</title><style>{CSS}</style>",
             '<main>',
             '<div class="eyebrow">PCCT airway segmentation · deformable evaluation</div>',
             '<h1>Airway Deformation Sweep</h1>',
             f'<p class="lede">How much of a synthetic 6-generation airway tree the <code>AirwayEvaluation</code> '
             f'CLI recovers after {n_deformations} known deformations per tree, on a planar and a non-planar phantom, '
             f'across {summary["n_combos"]} combinations of its four registration hyperparameters.</p>',
             f'<p class="meta">{summary["n_runs"]:,} CLI runs · {summary["n_errors"]:,} CLI errors · '
             f'CLI built {esc(notes["cli_built"])} · report generated {esc(notes["generated"])}</p>']

    tiles = [
        ("Mean TLD after registration, non-planar", f'{best_row["mean TLD|non-planar"]:.1f}%',
         f'defaults {default_row["mean TLD|non-planar"]:.1f}%'),
        ("Deformations passing, non-planar", f"{passes_best} of {len(np_curves)}",
         f"defaults {passes_default} of {len(np_curves)}"),
        ("Mean gen-5 branch detection, non-planar", f'{best_row["mean gen-5 BD|non-planar"]:.1f}%',
         f'defaults {default_row["mean gen-5 BD|non-planar"]:.1f}%'),
        ("CLI errors (both trees)", f'{int(best_row["errors|non-planar"] + best_row["errors|planar"])}',
         f'defaults {int(default_row["errors|non-planar"] + default_row["errors|planar"])}'),
    ]
    parts.append('<div class="tiles">' + "".join(
        f'<div class="tile"><span class="label">{esc(a)}</span><span class="value">{esc(b)}</span>'
        f'<span class="from">{esc(c)} · recommended setting shown</span></div>' for a, b, c in tiles) + "</div>")

    # ---------------- setup ----------------
    parts.append('<h2>What was tested</h2><div class="prose">')
    parts.append(notes["setup_html"])
    parts.append("</div>")
    grid_table = pd.DataFrame([
        {"Parameter": SHORT[k], "CLI flag": FLAG[k],
         "Values swept": ", ".join(f"{v:g}" for v in sorted(raw[k].unique())) if k in raw else "",
         "CLI default": {"maximum_pairing_distance": "5", "minimum_landmark_distance": "5",
                         "sensitivity_multiplier": "2", "projection_neighbors": "5"}[k]}
        for k in SHORT])
    parts.append(table(grid_table))
    families = (deformations[deformations["tree"] == "non-planar"].groupby("family", sort=False)
                .agg(levels=("level", lambda s: ", ".join(s)), peak=("peak_disp", "max")).reset_index())
    families = families[families["family"] != "identity"]
    families.columns = ["Deformation family", "Levels", "Largest peak displacement (voxels)"]
    parts.append(table(families, {"Largest peak displacement (voxels)": lambda v: f"{v:.0f}"}))

    # ---------------- findings ----------------
    parts.append('<h2>Findings</h2><div class="findings">')
    for title, body in notes["findings"]:
        parts.append(f'<div class="finding"><h3>{esc(title)}</h3><p>{body}</p></div>')
    parts.append("</div>")

    # ---------------- recovery curves ----------------
    parts.append('<h2>How much is recoverable</h2>')
    parts.append('<div class="prose"><p>Tree length detected after registration, against the peak displacement the '
                 'deformation applies to the airway. The dotted line is the 95% pass level. The per-deformation best '
                 'is the highest TLD any of the 135 settings reached for that deformation, an upper bound that no '
                 'single setting reaches everywhere.</p></div>')
    legend = ('<div class="legend">'
              '<span><i class="swatch none dashed"></i>unregistered</span>'
              '<span><i class="swatch default"></i>CLI defaults</span>'
              '<span><i class="swatch best"></i>recommended setting</span>'
              '<span><i class="swatch oracle dashed"></i>best setting per deformation</span>'
              '<span>✕ CLI error</span></div>')
    parts.append(f'<div class="figure" id="fig-recovery"><div class="figure-head">{legend}{tree_toggle("fig-recovery")}</div>')
    for tree in TREES:
        panels = []
        for family in FAMILIES:
            rows = curve_rows(tree)
            rows = rows[rows["family"] == family].sort_values("magnitude")

            def pts(column, label):
                out, errs = [], []
                for _, r in rows.iterrows():
                    tip = f"{family} {r['level']} (peak {r['peak_disp']:.1f} vox)\n{label}: "
                    if pd.isna(r[column]):
                        errs.append((r["peak_disp"], 2, tip + "CLI error"))
                    else:
                        out.append((r["peak_disp"], r[column], tip + f"TLD {r[column]:.1f}%"))
                return out, errs

            none_pts, _ = pts("TLD_unregistered", "unregistered")
            default_pts, default_err = pts("TLD_default", "CLI defaults")
            best_pts, best_err = pts("TLD_best", "recommended")
            oracle_pts, _ = pts("TLD_oracle", "best per deformation")
            panels.append(line_panel(family, [
                dict(cls="none", points=none_pts, dash=True),
                dict(cls="oracle", points=oracle_pts, dash=True),
                dict(cls="default", points=default_pts, errors=default_err),
                dict(cls="best", points=best_pts, errors=best_err),
            ], "peak displacement (voxels)", reference=PASS_TLD))
        parts.append(f'<div class="panels" data-tree="{tree}"{" hidden" if tree != TREES[0] else ""}>{"".join(panels)}</div>')
    parts.append('<p class="caption">TLD (%) on the y axis. Hover a point for its deformation level and value.</p></div>')

    lim = limits.copy()
    lim = lim[["tree", "family", "passes (default)", "largest passing (default)", "passes (best)",
               "largest passing (best)", "mean TLD default", "mean TLD best", "mean TLD per-deformation best"]]
    lim.columns = ["Tree", "Family", "Passing, defaults", "Largest passing, defaults", "Passing, recommended",
                   "Largest passing, recommended", "Mean TLD, defaults", "Mean TLD, recommended",
                   "Mean TLD, best per deformation"]
    pct = lambda v: f"{v:.1f}%"
    parts.append('<h3>Largest deformation that still passes</h3>')
    parts.append(table(lim, {c: pct for c in lim.columns if c.startswith("Mean")}))

    # ---------------- generations ----------------
    parts.append('<h2>Where the remaining loss is</h2>')
    parts.append(f'<div class="prose">{notes["generations_html"]}</div>')
    parts.append(f'<div class="figure" id="fig-gen"><div class="figure-head">{ramp_legend()}{tree_toggle("fig-gen")}</div>')
    for tree in TREES:
        maps = []
        for label, key in (("CLI defaults", "defaults"), ("Recommended setting", "recommended")):
            rows = generations[(generations["tree"] == tree) & (generations["parameters"] == key)
                               & (generations["family"] != "identity")].sort_values("deformation_id")
            entries = []
            for _, r in rows.iterrows():
                values, tips = [], []
                for g in GENERATIONS:
                    v = r[f"g{g}_BD"]
                    values.append(v)
                    if pd.isna(v):
                        tips.append(f"{r['family']} {r['level']}, gen {g}: CLI error")
                    else:
                        tips.append(f"{r['family']} {r['level']}, gen {g}\n{int(r[f'g{g}_detected'])} of "
                                    f"{int(r[f'g{g}_total'])} branches detected ({v:.0f}%)\nTLD {r['TLD']:.1f}%")
                entries.append((f"{r['family']} · {r['level']}", values, tips))
            maps.append(heatmap(label, entries))
        parts.append(f'<div class="panels two" data-tree="{tree}"{" hidden" if tree != TREES[0] else ""}>{"".join(maps)}</div>')
    parts.append('<p class="caption">Cells with a number lost at least one branch in that generation; unlabeled cells '
                 'are 100%. Generation 0 is the trachea.</p></div>')
    gs = generation_summary.copy()
    gs.columns = ["Parameters", "Tree"] + [f"Gen {g}" for g in GENERATIONS]
    parts.append('<h3>Mean branch detection per generation across all deformations</h3>')
    parts.append(table(gs, {f"Gen {g}": pct for g in GENERATIONS}))

    # ---------------- parameter effects ----------------
    parts.append('<h2>What each hyperparameter does</h2>')
    parts.append(f'<div class="prose">{notes["effects_html"]}</div>')
    effect_legend = ('<div class="legend"><span><i class="swatch best"></i>non-planar tree</span>'
                     '<span><i class="swatch oracle"></i>planar tree</span></div>')
    parts.append(f'<div class="figure"><div class="figure-head">{effect_legend}</div>')
    for metric, label, domain, fmt_axis in (("mean TLD", "Mean TLD (%)", None, lambda t: f"{t:g}"),
                                            ("pass rate", "Pass rate", (0, None), lambda t: f"{t:.0%}"),
                                            ("error rate", "CLI error rate", (0, None), lambda t: f"{t:.0%}")):
        values = effects[metric]
        lo = float(np.floor(values.min() / 5) * 5) if domain is None else domain[0]
        hi = float(np.ceil(values.max() / 5) * 5) if domain is None else max(0.05, float(np.ceil(values.max() * 20) / 20))
        panels = []
        for short, name in MAIN_EFFECT_NAME.items():
            rows = effects[effects["parameter"] == short]
            categories = [f"{v:g}" for v in sorted(rows["value"].unique())]
            series = []
            for tree, cls in (("non-planar", "best"), ("planar", "oracle")):
                t = rows[rows["tree"] == tree].sort_values("value")
                series.append(dict(cls=cls, values=list(t[metric]), tips=[
                    f"{SHORT[name]} = {v:g}, {tree} tree\n{label}: " + (f"{m:.1f}" if metric == "mean TLD" else f"{m:.1%}")
                    for v, m in zip(t["value"], t[metric])]))
            panels.append(category_panel(f"{SHORT[name]} · {label.lower()}", categories, series, (lo, hi), fmt_axis))
        parts.append(f'<div class="panels four">{"".join(panels)}</div>')
    parts.append('<p class="caption">Each point averages every deformation and every value of the other three '
                 'parameters. CLI errors count as TLD 0.</p></div>')

    # ---------------- ranking ----------------
    parts.append('<h2>Best parameter combinations</h2>')
    parts.append('<div class="prose"><p>Ranked by mean TLD over the non-planar tree\'s deformations; the CLI defaults '
                 f'rank {summary["default_rank"]} of {summary["n_combos"]}.</p></div>')
    top = pd.concat([rank.head(10), rank[rank["combo"] == summary["default_combo"]]]).drop_duplicates("combo")
    ranked = pd.DataFrame({
        "Rank": [int(np.flatnonzero(rank["combo"] == c)[0]) + 1 for c in top["combo"]],
        "Pairing": top["maximum_pairing_distance"], "Min landmark": top["minimum_landmark_distance"],
        "Sensitivity": top["sensitivity_multiplier"], "Neighbors": top["projection_neighbors"],
        "Mean TLD, non-planar": top["mean TLD|non-planar"], "Passes, non-planar": top["passes|non-planar"],
        "Errors, non-planar": top["errors|non-planar"], "Worst TLD, non-planar": top["worst TLD|non-planar"],
        "Mean TLD, planar": top["mean TLD|planar"], "Passes, planar": top["passes|planar"],
        "Errors, planar": top["errors|planar"],
    })
    num = lambda v: f"{v:g}"
    table_html = table(ranked, {"Mean TLD, non-planar": pct, "Worst TLD, non-planar": pct, "Mean TLD, planar": pct,
                                "Passes, non-planar": lambda v: f"{int(v)}", "Errors, non-planar": lambda v: f"{int(v)}",
                                "Passes, planar": lambda v: f"{int(v)}", "Errors, planar": lambda v: f"{int(v)}",
                                "Pairing": num, "Min landmark": num, "Sensitivity": num,
                                "Neighbors": lambda v: f"{int(v)}"})
    rows_html = table_html.split("<tbody>")[1].split("</tbody>")[0].split("</tr>")
    marked = []
    for row, combo in zip(rows_html, top["combo"]):
        cls = "is-best" if combo == summary["best_combo"] else "is-default" if combo == summary["default_combo"] else ""
        marked.append(row.replace("<tr>", f'<tr class="{cls}">', 1) + "</tr>")
    parts.append(table_html.split("<tbody>")[0] + "<tbody>" + "".join(marked) + "</tbody></table></div>")
    parts.append('<p class="caption">Blue row: recommended. Orange row: CLI defaults.</p>')

    # ---------------- scoring sensitivity ----------------
    parts.append('<h2>Scoring sensitivity</h2>')
    parts.append(f'<div class="prose">{notes["threshold_html"]}</div>')
    th = thresholds.copy()
    th.columns = ["Tree", "Family"] + [c.replace("BD @", "Mean BD at") for c in th.columns[2:]]
    parts.append(table(th, {c: pct for c in th.columns[2:]}))

    # ---------------- recommendation ----------------
    parts.append('<h2>Recommended parameters</h2><div class="recommend">')
    flags = " ".join(f"{FLAG[k]} {v:g}" for k, v in best.items())
    parts.append(f'<p>For the best recovery through deformation, run the CLI with:</p><div class="flags">'
                 f'AirwayEvaluation --deformableEvaluation {esc(flags)}</div>')
    rec = pd.DataFrame([{"Parameter": SHORT[k], "CLI default": {"maximum_pairing_distance": 5, "minimum_landmark_distance": 5,
                                                              "sensitivity_multiplier": 2, "projection_neighbors": 5}[k],
                         "Recommended": best[k]} for k in SHORT])
    parts.append(table(rec, {"CLI default": num, "Recommended": num},
                       highlight=lambda c, v: "strong" if c == "Recommended" else ""))
    parts.append(f'<div class="prose">{notes["recommendation_html"]}</div></div>')

    parts.append('<p class="meta">Source: sweep_results/results.csv (DeformationSweep.py), analyzed in '
                 'DeformationSweepReport.ipynb; this page is built by make_sweep_report.py.</p>')
    parts.append(f'</main><div id="tip" hidden></div><script>{SCRIPT}</script>')
    parts.insert(1, '<link rel="preconnect" href="https://fonts.googleapis.com">'
                    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
                    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500'
                    '&family=IBM+Plex+Sans:wght@400;500;600&display=swap">')
    with open(OUT, "w") as f:
        f.write("\n".join(parts))
    print("wrote", OUT, f"{os.path.getsize(OUT) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
