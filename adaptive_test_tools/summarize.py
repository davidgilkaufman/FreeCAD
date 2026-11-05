#!/usr/bin/env python3
"""Summary statistics of cut load dumps: per variant, percentiles of (area per distance)/target
weighted by travel distance, travel above thresholds, and the worst steps.

Usage: summarize.py DUMP.bin [DUMP.bin ...]
"""
import sys

import numpy as np

from cutload_data import load, loadPerDistance

ORDER = ["computed", "computed_replay", "output"]


def wpercentile(v, w, q):
    i = np.argsort(v)
    c = np.cumsum(w[i]) / w.sum()
    return [float(v[i][np.searchsorted(c, x / 100)]) for x in q]


for path in sys.argv[1:]:
    d = load(path)
    h = d.header
    opt = h["optimalCutAreaPD"]
    print(f"== {path}: tool {h['toolDiameter']:g} mm, step over {100 * h['stepOverFactor']:g}%, "
          f"{len(d.passes)} passes, target {opt:.4g} mm^2/mm")
    print(f"  {'variant':16s} {'p50':>6s} {'p95':>6s} {'p99':>6s} {'p99.9':>6s} {'max':>6s}  "
          f"{'mm>1.05':>8s} {'mm>1.1':>7s} {'mm>1.2':>7s} {'total mm':>9s}   (cutting steps only, distance weighted)")
    for k in ORDER:
        if k not in d.passes[0].variants:
            continue
        R, W = [], []
        for p in d.passes:
            pd, dist = loadPerDistance(p.variants[k])
            R.append(pd / opt)
            W.append(dist)
        r, w = np.concatenate(R), np.concatenate(W)
        cut = r > 0.05  # ignore travel through (nearly) cleared area
        pc = wpercentile(r[cut], w[cut], [50, 95, 99, 99.9])
        print(f"  {k:16s} " + " ".join(f"{x:6.3f}" for x in pc) + f" {r.max():6.3f}  "
              f"{w[r > 1.05].sum():8.1f} {w[r > 1.1].sum():7.1f} {w[r > 1.2].sum():7.1f} {w[cut].sum():9.0f}")
    # worst computed steps, with the algorithm's own search error for the step
    rows = []
    for p in d.passes:
        a = p.variants["computed"]
        pd, dist = loadPerDistance(a)
        for j in np.where(pd / opt > 1.15)[0]:
            rows.append((pd[j] / opt, p.index, j + 1, len(pd), a[j + 1, 0], a[j + 1, 1], dist[j], a[j + 1, 3]))
    rows.sort(reverse=True)
    print(f"  computed steps above 1.15x target: {len(rows)}")
    for r in rows[:8]:
        print(f"    {r[0]:.3f}x pass {r[1]} step {r[2]}/{r[3]} at ({r[4]:.2f}, {r[5]:.2f}) step len {r[6]:.3f} mm, "
              f"search error {abs(r[7]):.3f}{' (iteration limit)' if r[7] < 0 else ''}")
