#!/usr/bin/env python3
"""Render an adaptive cut load dump as an interactive standalone SVG.

Each step of each pass is colored by cut area per unit distance of travel, relative to the
algorithm's target (optimalCutAreaPD). Radio buttons choose the variant (computed / output points);
clicking a step shows its details in the right panel. Settings persist in localStorage. Scroll to
zoom, drag to pan, double click to reset.

Usage: render_svg.py OUT.svg DUMP.bin [MORE_DUMP.bin ...]
Variants from later dumps are added to (or replace) those from earlier ones; all dumps must hold
the same passes (same computed points).
"""

import html
import json
import sys

import numpy as np

from cutload_data import load, loadPerDistance, polygons

VARIANTS = [
    ("computed", "computed"),
    ("output", "output (gcode)"),
]
DEFAULT_VARIANT = "output"

# Color scale over (area per distance) / target. Clamped to [0, MAX_RATIO]. The legend gives each
# interval between stops equal width, so the interesting range near 1x gets most of the room.
MAX_RATIO = 2.0
# computed steps above this cut load (× target) are listed as over tolerance: 1 + AREA_ERROR_FACTOR
OVER_TOLERANCE = 1.05
# computed steps listed as large turns: at least this many degrees from the previous step (or, for a
# pass's first step, from the tool direction it starts with)
MIN_TURN = 8.0
STOPS = [
    (0.0, (200, 200, 200)),
    (0.5, (150, 185, 235)),
    (0.9, (40, 100, 210)),
    (1.0, (25, 155, 75)),
    (1.05, (165, 200, 0)),
    (1.1, (240, 180, 0)),
    (1.2, (240, 110, 0)),
    (1.5, (215, 25, 25)),
    (2.0, (120, 0, 120)),
]
# Line width in model units, as a fraction of the step over (the spacing between passes)
LINE_WIDTH_FRACTION = 0.35


def legendX(r, width):
    """Position of ratio r on the piecewise linear legend of the given width."""
    r = min(max(r, 0.0), MAX_RATIO)
    seg = width / (len(STOPS) - 1)
    for i, ((a, _), (b, _)) in enumerate(zip(STOPS, STOPS[1:])):
        if r <= b:
            return seg * (i + (r - a) / (b - a))
    return width


def color(r):
    r = min(max(r, 0.0), MAX_RATIO)
    for (a, ca), (b, cb) in zip(STOPS, STOPS[1:]):
        if r <= b:
            t = (r - a) / (b - a)
            return "#%02x%02x%02x" % tuple(round(ca[i] + (cb[i] - ca[i]) * t) for i in range(3))
    return "#%02x%02x%02x" % STOPS[-1][1]


def distToPolyline(P, Q, signed=False):
    """Distance from each point of P (m, 2) to the polyline Q (n, 2). If signed: positive when the
    point is to the right of Q (in Q's direction of travel), negative to the left."""
    if len(Q) == 1:
        return np.hypot(*(P - Q[0]).T)
    A, B = Q[:-1], Q[1:]
    AB = B - A
    L2 = np.maximum((AB * AB).sum(axis=1), 1e-30)
    out = np.empty(len(P))
    for c in range(0, len(P), 256):
        p = P[c : c + 256, None, :]
        t = np.clip(((p - A) * AB).sum(axis=2) / L2, 0, 1)
        d = p - (A + t[..., None] * AB)
        dist = np.sqrt((d * d).sum(axis=2))
        j = dist.argmin(axis=1)
        rows = np.arange(len(j))
        out[c : c + 256] = dist[rows, j]
        if signed:
            cross = AB[j, 0] * d[rows, j, 1] - AB[j, 1] * d[rows, j, 0]  # > 0: left of travel
            out[c : c + 256] *= np.where(cross > 0, -1.0, 1.0)
    return out


def merge(paths):
    dumps = [load(p) for p in paths]
    base = dumps[0]
    for d in dumps[1:]:
        if len(d.passes) != len(base.passes):
            raise SystemExit(f"dumps differ in pass count: {len(base.passes)} vs {len(d.passes)}")
        for a, b in zip(base.passes, d.passes):
            ca, cb = a.variants["computed"], b.variants["computed"]
            if ca.shape != cb.shape or np.abs(ca[:, :2] - cb[:, :2]).max() > 1e-9:
                raise SystemExit(f"dumps differ in pass {a.index}")
            a.variants.update(b.variants)
    return base


def render(out, dumpPaths):
    d = merge(dumpPaths)
    opt = d.header["optimalCutAreaPD"]
    available = [k for k, _ in VARIANTS if all(k in p.variants for p in d.passes)]

    allPts = np.concatenate([p.variants["computed"][:, :2] for p in d.passes])
    xmin, ymin = allPts.min(axis=0)
    xmax, ymax = allPts.max(axis=0)
    pad = d.header["toolDiameter"] / 2 + 1.0  # tool radius plus a small buffer around the path
    xmin, ymin, xmax, ymax = xmin - pad, ymin - pad, xmax + pad, ymax + pad
    w, h = xmax - xmin, ymax - ymin
    lineWidth = LINE_WIDTH_FRACTION * d.header["toolDiameter"] * d.header["stepOverFactor"]

    def pathData(arr):
        return " ".join(
            "M" + " L".join(f"{x:.3f} {y:.3f}" for x, y in poly) + "Z" for poly in polygons(arr)
        )

    # cleared area per "world": computed (the algorithm's; previous passes swept along their
    # computed points) and output (previous passes swept along their gcode points)
    def world(prefix):
        if not all(prefix + "cleared_start" in p.variants for p in d.passes):
            return None
        return {"start": [pathData(p.variants[prefix + "cleared_start"]) for p in d.passes]}

    cleared = {"computed": world(""), "output": world("output_")}
    hasCleared = cleared["computed"] is not None
    if cleared["output"] is None:
        cleared["output"] = cleared["computed"]  # older dumps: only the algorithm's cleared area
    if not hasCleared:
        cleared = None

    # per point data for tooltips: ratio of the step ending at the point, and the point's distance
    # (micrometers) to the other version of the pass (output <-> computed)
    steps = {k: [] for k in available}
    for p in d.passes:
        comp = p.variants["computed"][:, :2]
        outArr = p.variants["output"]
        outVerts = outArr[outArr[:, 3] == 1][:, :2]
        for k in available:
            arr = p.variants[k]
            pd, _ = loadPerDistance(arr)
            ref = outVerts if k.startswith("computed") else comp
            dev = distToPolyline(arr[:, :2], ref, signed=True)
            entry = {
                "r": [0] + [round(float(x), 4) for x in pd / opt],
                "dev": [round(float(x) * 1000, 2) for x in dev],
            }
            # cut region perimeter split (lagging variants): [prev arc, current arc, other, polygon area]
            per = {"computed": "computed_perimeter", "output": "output_perimeter"}.get(k)
            if per and per in p.variants:
                entry["per"] = [[round(float(v), 5) for v in row] for row in p.variants[per]]
            # search exit tags of the computed step (output points: their corresponding computed step)
            if "computed_search" in p.variants:
                entry["search"] = [[int(r[0]), int(r[1]), int(r[2])] for r in p.variants["computed_search"]]
            # exit 2 steps: every try of the search, from the computed step start:
            # {point: [[x, y, ratio, angle deg, conventional, iteration, best deg, bound deg], ...]}
            if "search_tries_pos" in p.variants and len(p.variants["search_tries_pos"]):
                comp = p.variants["computed"][:, :2]
                tries = {}
                bounds = p.variants.get("search_tries_bounds", np.full((len(p.variants["search_tries_pos"]), 4), np.nan))
                deg = lambda v: None if np.isnan(v) else round(float(np.degrees(v)), 3)
                for (x, y, area, kp), (ang, conv, it, _), (bmin, bmax, _, _) in zip(
                    p.variants["search_tries_pos"], p.variants["search_tries_info"], bounds
                ):
                    kp = int(kp)
                    dist = float(np.hypot(x - comp[kp - 1, 0], y - comp[kp - 1, 1]))
                    ratio = area / dist / opt if dist > 0 else 0.0
                    tries.setdefault(str(kp), []).append(
                        [round(float(x), 5), round(float(y), 5), round(ratio, 4), round(float(np.degrees(ang)), 3),
                         int(conv), int(it), deg(bmin), deg(bmax)]
                    )
                entry["tries"] = tries
            if "fail" in p.variants and len(p.variants["fail"]):
                f = p.variants["fail"][0]
                entry["fail"] = [int(f[0]), int(f[1]), int(f[2]), int(f[3])]
            if "lead_out" in p.variants and len(p.variants["lead_out"]):
                entry["leadOut"] = [[round(float(x), 5), round(float(y), 5)] for x, y in p.variants["lead_out"][:, :2]]
            steps[k].append(entry)

    # largest turns: each computed step's heading relative to the previous step's (degrees, + = left),
    # leaving out steps that were rotated back inside the tool boundary
    turns = []
    initialHeading = []  # per pass: heading (degrees) of the tool direction before the first step, or None
    for i, p in enumerate(d.passes):
        v = np.diff(p.variants["computed"][:, :2], axis=0)
        heading = np.arctan2(v[:, 1], v[:, 0])
        h0 = None
        if "initial_dir" in p.variants and len(p.variants["initial_dir"]):
            dx, dy = p.variants["initial_dir"][0, :2]
            h0 = float(np.arctan2(dy, dx))
        initialHeading.append(None if h0 is None else round(float(np.degrees(h0)), 4))
        prev = np.concatenate([[h0 if h0 is not None else np.nan], heading[:-1]])
        rel = np.degrees((heading - prev + np.pi) % (2 * np.pi) - np.pi)  # step ending at point j + 1
        rotated = p.variants["computed_search"][:, 1] if "computed_search" in p.variants else np.zeros(len(v) + 1)
        turns += [(abs(t), i, j + 1, t) for j, t in enumerate(rel) if not np.isnan(t) and abs(t) >= MIN_TURN and not rotated[j + 1]]
    topTurns = [[i, k, round(float(t), 2)] for _, i, k, t in sorted(turns, reverse=True)]

    layers = []
    for i, p in enumerate(d.passes):
        groups = []
        for k in available:
            arr = p.variants[k]
            pd, dist = loadPerDistance(arr)
            ratio = pd / opt
            # one element per step (step j+1 goes from point j to point j+1), colored by its exact value
            lines = []
            for j in range(len(ratio)):
                (x0, y0), (x1, y1) = arr[j, :2], arr[j + 1, :2]
                lines.append(
                    f'<polyline points="{x0:.4f},{y0:.4f} {x1:.4f},{y1:.4f}" stroke="{color(ratio[j])}" '
                    f'data-r="{ratio[j]:.3f}-{ratio[j]:.3f}" data-s="{j + 1}"/>'
                )
            # drawn in reverse: earlier steps on top of later ones
            groups.append(f'<g class="v v-{k}">' + "".join(reversed(lines)) + "</g>")
        layers.append(f'<g class="pass" data-i="{i}">' + "".join(groups) + "</g>")

    # Legend gradient
    grad = "".join(
        f'<stop offset="{legendX(r, 1):.4f}" stop-color="{color(r)}"/>'
        for r in sorted(set(np.round(np.concatenate([np.linspace(a, b, 6) for (a, _), (b, _) in zip(STOPS, STOPS[1:])]), 6)))
    )
    ticks = [t for t, _ in STOPS]

    variantRadios = "".join(
        f'<label class="{"" if k in available else "na"}"><input type="radio" name="variant" value="{k}"'
        f'{"" if k in available else ' disabled="disabled"'}/> {html.escape(label)}'
        f'{"" if k in available else " (not in this dump)"}</label>'
        for k, label in VARIANTS
    )
    nPasses = len(d.passes)
    title = " + ".join(dumpPaths)
    meta = {
        "nPasses": nPasses,
        "available": available,
        "default": DEFAULT_VARIANT if DEFAULT_VARIANT in available else available[0],
        "opt": opt,
        "header": d.header,
        "cleared": cleared,
        "steps": steps,
        "stops": [[t, list(c)] for t, c in STOPS],
        "overTolerance": OVER_TOLERANCE,
        "topTurns": topTurns,
        "minTurn": MIN_TURN,
        "initialHeading": initialHeading,
        "lineWidth": lineWidth,
    }

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xhtml="http://www.w3.org/1999/xhtml"
     width="100%" height="100%" viewBox="0 0 1400 1000" preserveAspectRatio="xMidYMid meet"
     style="background:#fff;font-family:system-ui,sans-serif">
<title>Adaptive cut load</title>
<defs><linearGradient id="scale" x1="0" x2="1" y1="0" y2="0">{grad}</linearGradient></defs>
<style>
  .pass polyline {{ fill: none; stroke-width: {lineWidth:.4f}px;
                   stroke-linecap: round; stroke-linejoin: round; }}
  .v {{ display: none; }}
  .v-{meta["default"]} {{ display: inline; }}  /* without script: show the default variant */
  .hidden {{ display: none; }}
  #panel {{ font-size: 13px; color: #222; }}
  #panel label {{ display: block; margin: 2px 0; cursor: pointer; }}
  #panel label.na {{ color: #999; cursor: default; }}
  #panel input[type=range] {{ width: 100%; }}
  #panel .row {{ margin: 8px 0; }}
  #panel .stat {{ font-variant-numeric: tabular-nums; }}
  #panel td {{ padding: 0 10px 0 0; }}
  #tip {{ pointer-events: none; }}
</style>
<rect x="0" y="0" width="1400" height="1000" fill="#fff"/>
<svg id="view" x="0" y="0" width="1000" height="1000"
     viewBox="{xmin:.3f} {-ymax:.3f} {w:.3f} {h:.3f}" preserveAspectRatio="xMidYMid meet">
  <rect x="{xmin - 10 * w:.3f}" y="{-ymax - 10 * h:.3f}" width="{21 * w:.3f}" height="{21 * h:.3f}" fill="#fafafa"/>
  <g transform="scale(1,-1)" id="layers">
  <path id="clearedBg" d="" fill="#e3ddd0" fill-rule="evenodd" stroke="none"/>
{chr(10).join(reversed(layers))}
  <g id="markLayer"></g>
  </g>
  <g id="grid" pointer-events="none"></g>
</svg>
<line x1="1000" y1="0" x2="1000" y2="1000" stroke="#ccc"/>
<g transform="translate(1020,20)">
  <text x="0" y="12" font-size="15" font-weight="600">Cut area per distance / target</text>
  <rect x="0" y="22" width="360" height="16" fill="url(#scale)" stroke="#888" stroke-width="0.5"/>
  {"".join(f'<line x1="{legendX(t, 360):.1f}" x2="{legendX(t, 360):.1f}" y1="38" y2="44" stroke="#444"/><text x="{legendX(t, 360):.1f}" y="56" font-size="11" text-anchor="middle">{t:g}</text>' for t in ticks)}
  {"".join(f'<text x="{legendX(t, 360):.1f}" y="70" font-size="9" fill="#666" text-anchor="middle">{t * opt:.3g}</text>' for t in ticks)}
  <text x="0" y="86" font-size="10" fill="#666">× target (top), mm²/mm (bottom); target {opt:.4g} mm²/mm</text>
  <foreignObject x="0" y="96" width="370" height="880">
    <div xmlns="http://www.w3.org/1999/xhtml" id="panel">
      <div class="row"><b>Variant</b>{variantRadios}</div>
      <div class="row"><button id="resetView">Reset zoom</button>
        <button id="gotoWorst">Goto worst</button></div>
      <div class="row"><b id="overTitle"></b> <button id="overNext">Next</button>
        <div id="overList" style="max-height:130px;overflow-y:auto;font-size:12px;border:1px solid #ddd;margin-top:4px"></div></div>
      <div class="row"><b id="turnTitle"></b> <button id="turnNext">Next</button>
        <div id="turnList" style="max-height:130px;overflow-y:auto;font-size:12px;border:1px solid #ddd;margin-top:4px"></div></div>
      <div class="row stat" id="gridInfo" style="color:#666;font-size:12px"></div>
      <div class="row stat" id="markInfo"></div>
      <div class="row" style="color:#666;font-size:12px">Click a point for details. Background: cleared area at the start of the marked pass (else the initial cleared area).</div>
    </div>
  </foreignObject>
</g>
<g id="tip" visibility="hidden"><rect id="tipBg" fill="#fff" stroke="#888" rx="3"/><text id="tipText" font-size="12"></text></g>
<script><![CDATA[
(function () {{
  const META = {json.dumps(meta)};
  const SVGNS = "http://www.w3.org/2000/svg";
  // same color scale as the steps (STOPS in render_svg.py), from a cut load ratio
  function colorOf(r) {{
    const S = META.stops;
    r = Math.min(Math.max(r, 0), S[S.length - 1][0]);
    for (let i = 1; i < S.length; i++) {{
      if (r <= S[i][0]) {{
        const t = (r - S[i - 1][0]) / (S[i][0] - S[i - 1][0]);
        return "rgb(" + [0, 1, 2].map(c => Math.round(S[i - 1][1][c] + (S[i][1][c] - S[i - 1][1][c]) * t)).join(",") + ")";
      }}
    }}
    return "rgb(" + S[S.length - 1][1].join(",") + ")";
  }}
  const OTHER = v => v.startsWith("computed") ? "output" : "computed";
  let mark = null;  // marked [pass, point]
  let markTry = null;  // index of the selected search try of the marked step, if any
  const KEY = "adaptiveCutLoad:" + location.pathname + ":";
  const store = {{
    get(k) {{ try {{ return localStorage.getItem(KEY + k); }} catch (e) {{ return null; }} }},
    set(k, v) {{ try {{ localStorage.setItem(KEY + k, v); }} catch (e) {{}} }},
  }};
  // indexed by pass number (passes are drawn in reverse order, earlier passes on top)
  const passes = [];
  document.querySelectorAll(".pass").forEach(g => {{ passes[+g.dataset.i] = g; }});
  const radios = Array.from(document.querySelectorAll("input[name=variant]"));
  let variant = META.default;
  const clearedBg = document.getElementById("clearedBg");
  // background: the variant's cleared area at the start of the marked pass, else pass 0's
  function updateBg() {{
    if (!META.cleared) return;
    const w = META.cleared[variant.startsWith("output") ? "output" : "computed"];
    clearedBg.setAttribute("d", w.start[mark ? mark[0] : 0]);
  }}

  const clampInt = (v, d) => {{ const n = parseInt(v, 10); return isNaN(n) ? d : Math.max(0, Math.min(META.nPasses - 1, n)); }};
  const sv = store.get("variant");
  if (sv && META.available.includes(sv)) variant = sv;
  // URL hash overrides the stored settings: #variant=output&view=x,y,w,h&mark=pass:point&try=n
  const hash = new URLSearchParams(location.hash.slice(1));
  if (hash.has("variant") && META.available.includes(hash.get("variant"))) variant = hash.get("variant");

  let styleEl = document.createElementNS("http://www.w3.org/2000/svg", "style");
  document.documentElement.appendChild(styleEl);

  let update = function (changed) {{
    styleEl.textContent = ".v {{ display: none !important; }} .v-" + variant + " {{ display: inline !important; }}";
    radios.forEach(r => r.checked = r.value === variant);
    updateBg();
    store.set("variant", variant);
  }}
  radios.forEach(r => r.addEventListener("change", () => {{ if (r.checked) {{ variant = r.value; update(); }} }}));
  update();

  // zoom / pan
  const view = document.getElementById("view");
  const vb0 = view.getAttribute("viewBox").split(" ").map(Number);
  let vb = vb0.slice();
  const vbKey = "viewBox";
  try {{ const s = JSON.parse(store.get(vbKey)); if (Array.isArray(s) && s.length === 4 && s.every(isFinite)) vb = s; }} catch (e) {{}}
  if (hash.has("view")) {{
    // x, y in model coordinates (y up) of the lower left corner, then width, height
    const v = hash.get("view").split(",").map(Number);
    if (v.length === 4 && v.every(isFinite)) vb = [v[0], -(v[1] + v[3]), v[2], v[3]];
  }}
  // Clipper grid: lines at every integer clipper coordinate (1 unit = 1/scaleFactor mm), drawn
  // when they are at least GRID_MIN_PX screen pixels apart; every 10th line darker.
  const GRID_MIN_PX = 6;
  const gridEl = document.getElementById("grid");
  const gridInfo = document.getElementById("gridInfo");
  function drawGrid() {{
    gridEl.textContent = "";
    const m = view.getScreenCTM();
    if (!m) return;
    const unit = 1 / META.header.scaleFactor;  // mm
    const px = unit * m.a;  // screen pixels per clipper unit
    if (px < GRID_MIN_PX) {{
      gridInfo.textContent = "";
      return;
    }}
    gridInfo.textContent = "Grid: 1 clipper unit = " + (unit * 1000).toFixed(3) + " µm";
    // visible region in view coordinates (y down; model y = -view y), with a margin for aspect
    // visible region: the view's viewport corners (root units 0..1000) mapped into view coordinates
    const root = document.documentElement, rm = root.getScreenCTM(), inv = m.inverse();
    const corner = (x, y) => {{ const p = root.createSVGPoint(); p.x = x; p.y = y; return p.matrixTransform(rm).matrixTransform(inv); }};
    const ca = corner(0, 0), cb = corner(1000, 1000);
    const x0 = Math.min(ca.x, cb.x), x1 = Math.max(ca.x, cb.x), y0 = Math.min(ca.y, cb.y), y1 = Math.max(ca.y, cb.y);
    const add = (xa, ya, xb, yb, major) => {{
      const l = document.createElementNS(SVGNS, "line");
      l.setAttribute("x1", xa); l.setAttribute("y1", ya); l.setAttribute("x2", xb); l.setAttribute("y2", yb);
      l.setAttribute("stroke", major ? "rgba(0,0,0,0.45)" : "rgba(0,0,0,0.18)");
      l.setAttribute("stroke-width", "1"); l.setAttribute("vector-effect", "non-scaling-stroke");
      gridEl.appendChild(l);
    }};
    for (let i = Math.ceil(x0 / unit); i <= Math.floor(x1 / unit); i++) add(i * unit, y0, i * unit, y1, i % 10 === 0);
    // horizontal lines at model y = j units, i.e. view y = -j units
    for (let j = Math.ceil(-y1 / unit); j <= Math.floor(-y0 / unit); j++) add(x0, -j * unit, x1, -j * unit, j % 10 === 0);
  }}
  const setVB = () => {{ view.setAttribute("viewBox", vb.join(" ")); store.set(vbKey, JSON.stringify(vb)); drawGrid(); }};
  setVB();
  // Map a mouse event to the view's user (model) coordinates via the view's screen transform.
  // (getBoundingClientRect on a nested svg measures its content, not its viewport, so not that.)
  function toUser(evt) {{
    const m = view.getScreenCTM();
    const pt = document.documentElement.createSVGPoint();
    pt.x = evt.clientX; pt.y = evt.clientY;
    const p = pt.matrixTransform(m.inverse());
    return [p.x, p.y, 1 / m.a];  // last: user units per screen pixel
  }}
  view.addEventListener("wheel", e => {{
    e.preventDefault();
    const [ux, uy] = toUser(e);
    const f = Math.exp((e.deltaY > 0 ? 1 : -1) * 0.15);
    vb = [ux - (ux - vb[0]) * f, uy - (uy - vb[1]) * f, vb[2] * f, vb[3] * f];
    setVB();
  }}, {{ passive: false }});
  let drag = null;
  view.addEventListener("mousedown", e => {{ e.preventDefault(); drag = [e.clientX, e.clientY, vb.slice(), toUser(e)[2]]; }});
  window.addEventListener("mousemove", e => {{
    if (!drag) return;
    const [x0, y0, v0, sc] = drag;
    vb = [v0[0] - (e.clientX - x0) * sc, v0[1] - (e.clientY - y0) * sc, v0[2], v0[3]];
    setVB();
  }});
  window.addEventListener("mouseup", () => drag = null);
  const reset = () => {{ vb = vb0.slice(); setVB(); }};
  view.addEventListener("dblclick", reset);
  document.getElementById("resetView").addEventListener("click", reset);

  // step marker: tool circles at the start and end of one step of the current variant
  const markLayer = document.getElementById("markLayer");
  const toolR = META.header.toolDiameter / 2;
  const ptsOf = pl => pl.getAttribute("points").trim().split(/\\s+/).map(q => q.split(",").map(Number));
  // turn of step st (from a to b) relative to the previous step of the same variant, or for the
  // pass's first step, to the tool direction the pass starts with; as text in degrees
  function relTurn(pi, st, a, b) {{
    let prev = null;
    if (st > 1) {{
      const pl = passes[pi].querySelector(".v-" + variant + " polyline[data-s='" + (st - 1) + "']");
      if (pl) {{ const [p0, p1] = ptsOf(pl); prev = Math.atan2(p1[1] - p0[1], p1[0] - p0[0]) * 180 / Math.PI; }}
    }}
    else if (META.initialHeading[pi] !== null) prev = META.initialHeading[pi];
    if (prev === null) return "?°";
    let t = Math.atan2(b[1] - a[1], b[0] - a[0]) * 180 / Math.PI - prev;
    t = ((t + 180) % 360 + 360) % 360 - 180;
    return (t > 0 ? "+" : "") + t.toFixed(1) + "°";
  }}
  function drawMark() {{
    store.set("mark", JSON.stringify(mark));
    store.set("markTry", JSON.stringify(markTry));
    updateBg();
    markLayer.textContent = "";
    const info = document.getElementById("markInfo");
    info.textContent = "";
    if (!mark) return;
    const [pi, st] = mark;
    const g = passes[pi] && passes[pi].querySelector(".v-" + variant);
    if (!g) return;
    const pl = g.querySelector("polyline[data-s='" + st + "']");
    if (!pl) return;
    const [a, b] = ptsOf(pl);
    const add = (tag, attrs) => {{
      const e = document.createElementNS(SVGNS, tag);
      for (const k in attrs) e.setAttribute(k, attrs[k]);
      markLayer.appendChild(e);
      return e;
    }};
    const thin = {{ stroke: "#000", "stroke-width": "1.5", "vector-effect": "non-scaling-stroke", fill: "none" }};
    // search tries (exit 2 steps), from the computed step's start
    // search tries: of this step (from the computed step's start), and, on the last step of the
    // pass, of the failed search that ended the pass (from the last computed point)
    const cs = META.steps.computed[pi];
    const cpts = ptsOf(passes[pi].querySelector(".v-computed polyline[data-s='" + st + "']"));
    const isLast = !passes[pi].querySelector(".v-computed polyline[data-s='" + (st + 1) + "']");
    const groups = [];
    if (cs.tries && cs.tries[st]) groups.push({{ id: "", tries: cs.tries[st], origin: cpts[0] }});
    if (isLast && cs.tries && cs.tries[st + 1]) groups.push({{ id: "f", tries: cs.tries[st + 1], origin: cpts[1] }});
    let tr = null, trOrigin = null;
    for (const gr of groups) gr.tries.forEach((t, idx) => {{ if (markTry === gr.id + idx) {{ tr = t; trOrigin = gr.origin; }} }});
    // tool circles: the selected try if any, else the accepted step
    const [ca, cb] = tr ? [trOrigin, [tr[0], tr[1]]] : [a, b];
    add("circle", {{ cx: ca[0], cy: ca[1], r: toolR, ...thin, "stroke-dasharray": "4 3" }});
    add("circle", {{ cx: cb[0], cy: cb[1], r: toolR, ...thin }});
    add("line", {{ x1: a[0], y1: a[1], x2: b[0], y2: b[1], ...thin, "stroke-width": "2" }});
    // outline of the accepted step as drawn: the line's width around the segment, round ends
    {{
      const h = META.lineWidth / 2;
      const base = Math.atan2(b[1] - a[1], b[0] - a[0]), pts = [];
      for (let q = 0; q <= 32; q++) {{ const t = base + Math.PI / 2 - Math.PI * q / 32; pts.push([b[0] + h * Math.cos(t), b[1] + h * Math.sin(t)]); }}
      for (let q = 0; q <= 32; q++) {{ const t = base - Math.PI / 2 - Math.PI * q / 32; pts.push([a[0] + h * Math.cos(t), a[1] + h * Math.sin(t)]); }}
      add("polygon", {{ points: pts.map(p => p[0] + "," + p[1]).join(" "), ...thin }});
    }}
    // lead out after the pass (last step only), thin black
    if (isLast && cs.leadOut) {{
      add("polyline", {{ points: [cpts[1], ...cs.leadOut].map(p => p[0] + "," + p[1]).join(" "), ...thin, "pointer-events": "none" }});
    }}
    for (const gr of groups) gr.tries.forEach((t, idx) => {{
      const l = add("line", {{ x1: gr.origin[0], y1: gr.origin[1], x2: t[0], y2: t[1], stroke: colorOf(t[2]),
        "stroke-width": META.lineWidth / 4, "stroke-linecap": "round", class: "try", "data-t": gr.id + idx }});
      l.style.cursor = "pointer";
    }});
    if (tr) add("line", {{ x1: trOrigin[0], y1: trOrigin[1], x2: tr[0], y2: tr[1], ...thin, "stroke-width": "2", "pointer-events": "none" }});
    const sd = META.steps[variant][pi];
    const L = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const H = "<div style='font-weight:600;margin-top:10px'>";
    let html = H + "Pass " + pi + ", point " + st + ": step " + (st - 1) + "→" + st + " (" + variant + ")</div>" +
      "Cut load: " + sd.r[st].toFixed(3) + "× target (" + (sd.r[st] * META.opt).toFixed(3) + " mm²/mm)<br/>" +
      "Step " + L.toFixed(3) + " mm at " + relTurn(pi, st, a, b) + "<br/>" +
      "Smoothing offset " + sd.dev[st - 1].toFixed(2) + " / " + sd.dev[st].toFixed(2) + " µm (start/end)";
    if (sd.search && sd.search[st]) {{
      const [ex, rot, its] = sd.search[st];
      const EXIT = {{ 1: "within tolerance", 2: "adjacent integers", 3: "iteration limit", 4: "no cut area" }};
      html += "<br/>Search exit " + ex + " (" + EXIT[ex] + "), " + its + " iterations" + (rot ? ", rotated into boundary" : "");
    }}
    if (sd.per && sd.per[st]) {{
      const [lp, lc, lo] = sd.per[st];
      const u = 1 / META.header.scaleFactor;  // 1 clipper unit, mm
      const k = u / (L * META.opt);  // ratio change per mm of boundary moved by 1 unit
      const f = x => "±" + (x * k).toFixed(4) + "×";
      html += H + "Cut region perimeter</div>" +
        "<table style='border-collapse:collapse;font-size:12px'>" +
        "<tr style='color:#666'><td></td><td>length</td><td>1 unit shift</td></tr>" +
        "<tr><td>previous tool arc</td><td>" + lp.toFixed(3) + " mm</td><td>" + f(lp) + "</td></tr>" +
        "<tr><td>current tool arc</td><td>" + lc.toFixed(3) + " mm</td><td>" + f(lc) + "</td></tr>" +
        "<tr><td>cleared edge</td><td>" + lo.toFixed(3) + " mm</td><td>" + f(lo) + "</td></tr>" +
        "<tr><td>all</td><td>" + (lp + lc + lo).toFixed(3) + " mm</td><td>" + f(lp + lc + lo) + "</td></tr></table>";
    }}
    const R = "<td style='text-align:right'>";
    const fmtA = v => v === null ? "–" : v.toFixed(2) + "°";
    const triesTable = (gr, acceptedAt) =>
      "<table style='border-collapse:collapse;font-size:12px;white-space:nowrap'>" +
      "<tr style='color:#666'><td>#</td>" + R + "angle</td>" + R + "load</td>" + R + "best</td>" + R + "bound</td><td></td></tr>" +
      gr.tries.map((t, idx) => {{
        const acc = acceptedAt && Math.abs(t[0] - acceptedAt[0]) < 1e-4 && Math.abs(t[1] - acceptedAt[1]) < 1e-4;
        const sel = gr.id + idx === markTry ? " style='cursor:pointer;background:#eef;font-weight:600'" : " style='cursor:pointer'";
        return "<tr data-t='" + gr.id + idx + "'" + sel + "><td>" + t[5] + "</td>" + R + t[3].toFixed(2) + "°</td>" +
          R + t[2].toFixed(4) + "×</td>" + R + fmtA(t[6]) + "</td>" + R + fmtA(t[7]) + "</td><td>" +
          [acc ? "accepted" : "", t[4] ? "conv." : ""].filter(x => x).join(", ") + "</td></tr>";
      }}).join("") + "</table>";
    for (const gr of groups) {{
      if (gr.id === "") {{
        html += H + "Search tries</div>" + triesTable(gr, cpts[1]);
      }}
      else {{
        const EXIT = {{ 1: "within tolerance", 2: "adjacent integers", 3: "iteration limit", 4: "no cut area" }};
        const FAIL = [[1, "area &lt; 1"], [2, "conventional"], [4, "over 2× target"], [8, "can't rotate into boundary"]];
        const f = cs.fail || [0, 0, 0, 0];
        html += H + "Failed search ending the pass</div>" +
          "Exit " + f[0] + " (" + (EXIT[f[0]] || "?") + "), " + f[3] + " iterations" + (f[1] ? ", rotated" : "") + "<br/>" +
          "Failed: " + (FAIL.filter(([b]) => f[2] & b).map(([, n]) => n).join(", ") || "–") +
          triesTable(gr, null);
      }}
    }}
    if (groups.length) html += "<div style='color:#666;font-size:11px'>best / bound: interpolation points (interp.m_best / m_bound) after the try</div>";
    info.innerHTML = html;
  }}
  try {{
    const sm = JSON.parse(store.get("mark"));
    if (Array.isArray(sm) && sm.length === 2 && sm.every(Number.isInteger) && sm[0] >= 0 && sm[0] < META.nPasses) mark = sm;
  }} catch (e) {{}}
  try {{ const t = JSON.parse(store.get("markTry")); if (typeof t === "string") markTry = t; }} catch (e) {{}}
  if (hash.has("mark")) {{
    const m = hash.get("mark").split(":").map(Number);
    if (m.length === 2 && m.every(isFinite)) {{ mark = m; markTry = hash.has("try") ? hash.get("try") : null; }}
  }}
  let downAt = null;
  view.addEventListener("mousedown", e => {{ downAt = [e.clientX, e.clientY]; }});
  view.addEventListener("click", e => {{
    if (downAt && Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4) return;  // was a drag
    const t = e.target;
    if (t.classList && t.classList.contains("try")) {{ markTry = t.dataset.t; drawMark(); return; }}
    markTry = null;
    if (t.tagName !== "polyline") {{ mark = null; drawMark(); return; }}
    const [pi, i] = pointAt(t, e);
    mark = [pi, i];
    drawMark();
  }});
  // select the step with the highest cut load (current variant) and center on it
  // select a step and center the view on it (keeping the zoom)
  function gotoStep(pi, k) {{
    mark = [pi, k];
    markTry = null;
    const pl = passes[pi].querySelector(".v-" + variant + " polyline[data-s='" + k + "']");
    const [a, b] = ptsOf(pl);
    const cx = (a[0] + b[0]) / 2, cy = -(a[1] + b[1]) / 2;  // view coordinates (y down)
    vb = [cx - vb[2] / 2, cy - vb[3] / 2, vb[2], vb[3]];
    setVB();
    drawMark();
    highlightOver();
    highlightTurns();
  }}
  document.getElementById("gotoWorst").addEventListener("click", () => {{
    let best = null;
    for (let pi = 0; pi < META.nPasses; pi++) {{
      const r = META.steps[variant][pi].r;
      for (let k = 1; k < r.length; k++) if (!best || r[k] > best[2]) best = [pi, k, r[k]];
    }}
    if (best) gotoStep(best[0], best[1]);
  }});
  // computed steps over tolerance: count, list, next
  const over = [];
  for (let pi = 0; pi < META.nPasses; pi++) {{
    const r = META.steps.computed[pi].r;
    for (let k = 1; k < r.length; k++) if (r[k] > META.overTolerance) over.push([pi, k, r[k]]);
  }}
  const overList = document.getElementById("overList");
  document.getElementById("overTitle").textContent =
    over.length + " computed step" + (over.length === 1 ? "" : "s") + " over " + META.overTolerance + "× target";
  overList.innerHTML = over.map(([pi, k, r], i) =>
    "<div data-o='" + i + "' style='cursor:pointer;padding:0 4px'>pass " + pi + ", point " + k + ": " + r.toFixed(3) + "×</div>").join("");
  function highlightOver() {{
    overList.querySelectorAll("div[data-o]").forEach(el => {{
      const [pi, k] = over[+el.dataset.o];
      const on = mark && mark[0] === pi && mark[1] === k;
      el.style.background = on ? "#eef" : "";
      el.style.fontWeight = on ? "600" : "";
      if (on) el.scrollIntoView({{ block: "nearest" }});
    }});
  }}
  overList.addEventListener("click", e => {{
    const el = e.target.closest && e.target.closest("div[data-o]");
    if (el) {{ const [pi, k] = over[+el.dataset.o]; gotoStep(pi, k); }}
  }});
  document.getElementById("overNext").addEventListener("click", () => {{
    if (!over.length) return;
    // the first one after the marked step (in pass, point order), wrapping around
    let i = over.findIndex(([pi, k]) => !mark || pi > mark[0] || (pi === mark[0] && k > mark[1]));
    if (i < 0) i = 0;
    gotoStep(over[i][0], over[i][1]);
  }});
  highlightOver();
  // largest turns from the previous step
  const turnList = document.getElementById("turnList");
  document.getElementById("turnTitle").textContent = META.topTurns.length + " computed turn" + (META.topTurns.length === 1 ? "" : "s") +
    " of " + META.minTurn + "° or more (not rotated into boundary)";
  turnList.innerHTML = META.topTurns.map(([pi, k, t], i) =>
    "<div data-u='" + i + "' style='cursor:pointer;padding:0 4px'>pass " + pi + ", point " + k + ": " + (t > 0 ? "+" : "") + t.toFixed(1) + "°</div>").join("");
  function highlightTurns() {{
    turnList.querySelectorAll("div[data-u]").forEach(el => {{
      const [pi, k] = META.topTurns[+el.dataset.u];
      const on = mark && mark[0] === pi && mark[1] === k;
      el.style.background = on ? "#eef" : "";
      el.style.fontWeight = on ? "600" : "";
      if (on) el.scrollIntoView({{ block: "nearest" }});
    }});
  }}
  turnList.addEventListener("click", e => {{
    const el = e.target.closest && e.target.closest("div[data-u]");
    if (el) {{ const [pi, k] = META.topTurns[+el.dataset.u]; gotoStep(pi, k); }}
  }});
  document.getElementById("turnNext").addEventListener("click", () => {{
    if (!META.topTurns.length) return;
    // the entry after the marked step if it is in the list, else the first
    const cur = META.topTurns.findIndex(([pi, k]) => mark && mark[0] === pi && mark[1] === k);
    const [pi, k] = META.topTurns[(cur + 1) % META.topTurns.length];
    gotoStep(pi, k);
  }});
  highlightTurns();
  document.getElementById("markInfo").addEventListener("click", e => {{
    const row = e.target.closest && e.target.closest("tr[data-t]");
    if (!row) return;
    markTry = row.dataset.t === markTry ? null : row.dataset.t;
    drawMark();
  }});
  const origUpdate = update;
  update = function (c) {{ origUpdate(c); drawMark(); }};
  radios.forEach(r => r.addEventListener("change", drawMark));
  drawMark();

  // tooltip
  const tip = document.getElementById("tip"), tipText = document.getElementById("tipText"), tipBg = document.getElementById("tipBg");
  const root = document.documentElement;
  function pointAt(t) {{
    // each step is its own element: the browser's hit test (z order, round ends) picks the step.
    // Returns [pass, point index at the end of that step].
    return [+t.closest(".pass").dataset.i, +t.dataset.s];
  }}
  view.addEventListener("mousemove", e => {{
    const t = e.target;
    if (t.tagName !== "polyline") return;
    const [pi, i] = pointAt(t, e);
    const sd = META.steps[variant][pi];
    const r = sd.r[i], dev = sd.dev[i];
    tipText.textContent = "pass " + pi + ", point " + i + ": step " + (i - 1) + "→" + i + " " + r.toFixed(3) +
      "× target; " + (dev >= 0 ? "+" : "") + dev.toFixed(2) + " µm from the " + OTHER(variant) + " path" +
      (sd.search && sd.search[i] ? "; exit " + sd.search[i][0] + (sd.search[i][1] ? ", rotated" : "") : "");
    tip.setAttribute("visibility", "visible");
  }});
  view.addEventListener("mouseout", e => {{ if (e.target.tagName === "polyline") tip.setAttribute("visibility", "hidden"); }});
  root.addEventListener("mousemove", e => {{
    if (tip.getAttribute("visibility") !== "visible") return;
    const pt = root.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    const p = pt.matrixTransform(root.getScreenCTM().inverse());
    tipText.setAttribute("x", p.x + 14); tipText.setAttribute("y", p.y - 10);
    const bb = tipText.getBBox();
    tipBg.setAttribute("x", bb.x - 4); tipBg.setAttribute("y", bb.y - 2);
    tipBg.setAttribute("width", bb.width + 8); tipBg.setAttribute("height", bb.height + 4);
  }});
}})();
]]></script>
</svg>
"""
    with open(out, "w") as f:
        f.write(svg)
    print(f"wrote {out}: {nPasses} passes, variants {available}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    render(sys.argv[1], sys.argv[2:])
