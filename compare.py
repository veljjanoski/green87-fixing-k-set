"""Compare result files: BW table (k<=30), float vs fixed per k, monotonicity."""
import os, re, sys
def load(fn):
    d = {}
    for l in open(fn):
        m = re.match(r"k=\s*(\d+) p\(k\)=([0-9.]+)", l)
        if m: d[int(m.group(1))] = m.group(2)
    return d
fixed = load(sys.argv[1])
bw = {int(l.split()[0]): float(l.split()[1]) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'bw_table.txt'))}
bad = [k for k in bw if k in fixed and abs(float(fixed[k]) - bw[k]) > 5.1e-9]
print(f"BW k<=30: {sum(k in fixed for k in bw)} compared, mismatches: {bad}")
if len(sys.argv) > 2:
    fl = load(sys.argv[2])
    worst = max(abs(float(fixed[k]) - float(fl[k])) for k in fixed if k in fl)
    print(f"float vs fixed: {sum(k in fl for k in fixed)} compared, worst diff {worst:.1e}")
ks = sorted(fixed)
from fractions import Fraction
inc = [(k) for a, k in zip(ks, ks[1:]) if k == a + 1 and Fraction(fixed[k]) >= Fraction(fixed[a])]
print(f"k range {ks[0]}..{ks[-1]}, increases at: {inc if inc else 'none'}")
gaps = [(k, float(fixed[a]) - float(fixed[k])) for a, k in zip(ks, ks[1:]) if k == a + 1]
print("smallest decrement p(k-1)-p(k):", min(gaps, key=lambda t: t[1]))
