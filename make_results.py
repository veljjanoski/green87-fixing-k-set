"""Build data/results.csv and data/table.md from the raw run outputs.

k <= 80 : data/results_fixed.txt (CPU, 128-bit fixed point, rigorous bound)
k >= 81 : data/results_gpu.txt   (GPU, float64, bound 2^-53 (4 merges + 2))
Also prints the monotonicity check and a few summary statistics.
"""
import math
import os
import re
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def load(fn):
    d = {}
    for line in open(os.path.join(DATA, fn)):
        m = re.match(r"k=\s*(\d+) p\(k\)=([0-9.]+) err<=([0-9.e+-]+) max_states=(\d+)", line)
        if m:
            d[int(m.group(1))] = (m.group(2), float(m.group(3)), int(m.group(4)))
    return d


fixed = load("results_fixed.txt")
gpu = load("results_gpu.txt")
rows = []
for k in sorted(set(fixed) | set(gpu)):
    if k in fixed:
        p, err, st = fixed[k]
        rows.append((k, p, err, st, "cpu-fixed128"))
    else:
        p, err, st = gpu[k]
        rows.append((k, p, err, st, "gpu-float64"))

with open(os.path.join(DATA, "results.csv"), "w", newline="\n") as f:
    f.write("k,p_k,abs_error_bound,peak_states,source\n")
    for k, p, err, st, src in rows:
        f.write(f"{k},{p},{err:.1e},{st},{src}\n")

delta = 1 - (1 + math.log(math.log(2))) / math.log(2)


def fmt_bound(x):
    """2 significant digits, rounded UP (a bound must not shrink in display)."""
    e = math.floor(math.log10(x))
    m = math.ceil(x / 10 ** e * 10 - 1e-9) / 10
    if m >= 10:
        m, e = 1.0, e + 1
    return f"{m:.1f}e{e}"


with open(os.path.join(DATA, "table.md"), "w", newline="\n") as f:
    f.write("| k | p(k) | bound | peak states | k | p(k) | bound | peak states |\n")
    f.write("|--:|:--|:--|--:|--:|:--|:--|--:|\n")
    half = (len(rows) + 1) // 2
    for a in range(half):
        cells = []
        for idx in (a, a + half):
            if idx < len(rows):
                k, p, err, st, src = rows[idx]
                cells.append(f"{k} | {p[:14]} | {fmt_bound(err)} | {st:,}")
            else:
                cells.append(" | | | ")
        f.write("| " + " | ".join(cells) + " |\n")

# checks and summary
inc = [k for (k, p, *_), (k2, p2, *_) in zip(rows, rows[1:]) if Fraction(p2) >= Fraction(p)]
print("values:", len(rows), "k =", rows[0][0], "..", rows[-1][0])
print("increases:", inc if inc else "none")
steps = [(k2, float(p) - float(p2)) for (k, p, *_), (k2, p2, *_) in zip(rows, rows[1:])]
kmin, smin = min(steps, key=lambda t: t[1])
print(f"smallest step p(k-1)-p(k): {smin:.3e} at k={kmin}")
for k in (10, 30, 50, 80, 100, 121):
    p = float(dict((r[0], r[1]) for r in rows)[k])
    pk1 = float(dict((r[0], r[1]) for r in rows)[k - 1])
    print(f"k={k:3d} p={p:.6f} rel.step={(pk1 - p) / p:.5f} delta/k={delta / k:.5f} "
          f"ratio p/(k^-d (log k)^-3/2)={p / (k ** -delta * math.log(k) ** -1.5):.3f}")
