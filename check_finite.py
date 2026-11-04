"""Finite-n check of the reading of the problem: i(n,k) = proportion of
permutations of [n] with an invariant set of size k = proportion whose cycle
type has a sub-partition of size k.  Exact rationals via partitions of n.
Compare with Britnell-Wildon Appendix C (5 decimals), e.g. i(30,9) = .38452,
i(30,10) = .38548 (the first finite-n non-monotonic pair), i(10,3) = .49467.
"""
from fractions import Fraction
import math
import sys


def partitions(n, maxpart=None):
    if maxpart is None:
        maxpart = n
    if n == 0:
        yield []
        return
    for p in range(min(n, maxpart), 0, -1):
        for rest in partitions(n - p, p):
            yield [p] + rest


def has_subpartition(parts, k):
    sums = {0}
    for x in parts:
        sums = sums | {s + x for s in sums if s + x <= k}
    return k in sums


def i_n_k(n, k):
    tot = Fraction(0)
    for parts in partitions(n):
        if has_subpartition(parts, k):
            # 1/|centralizer| = 1/prod(i^{m_i} m_i!)
            c = 1
            from collections import Counter
            for i, m in Counter(parts).items():
                c *= i ** m * math.factorial(m)
            tot += Fraction(1, c)
    return tot


if __name__ == "__main__":
    checks = [(10, 3, 0.49467), (30, 9, 0.38452), (30, 10, 0.38548), (20, 10, 0.25881),
              (25, 12, 0.32342), (30, 15, 0.23153), (28, 7, 0.40802)]
    for n, k, ref in checks:
        v = i_n_k(n, k)
        print(f"i({n},{k}) = {float(v):.5f}  BW: {ref:.5f}  {'OK' if abs(float(v) - ref) < 6e-6 else 'MISMATCH'}")
