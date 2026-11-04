"""Independent definition-based checker for q(k) = 1 - p(k), small k only.

Enumerates every capped count vector (b_1..b_k), b_i <= m_i = k // i, builds
the multiset of parts explicitly, computes ALL subset sums by a plain set
construction (no bitmasks, no pruning), and adds the probability of the
vector (Poisson masses, with the capped value carrying the Poisson tail).
Compares against pk_dp.q_of_k for each k.
"""
import itertools
import os
import math
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pk_dp import q_of_k  # noqa: E402


def subset_sums(parts):
    sums = {0}
    for x in parts:
        sums = sums | {s + x for s in sums}
    return sums


def pois_capped(i, m):
    lam = 1.0 / i
    probs = [math.exp(-lam) * lam ** j / math.factorial(j) for j in range(m)]
    probs.append(1.0 - sum(probs))
    return probs


def q_brute(k):
    caps = [k // i for i in range(1, k + 1)]
    tables = [pois_capped(i, caps[i - 1]) for i in range(1, k + 1)]
    q = 0.0
    n_vec = 0
    for b in itertools.product(*[range(c + 1) for c in caps]):
        n_vec += 1
        parts = []
        for i, bi in enumerate(b, start=1):
            parts.extend([i] * bi)
        if k in subset_sums(parts):
            continue
        pr = 1.0
        for i, bi in enumerate(b, start=1):
            pr *= tables[i - 1][bi]
        q += pr
    return q, n_vec


if __name__ == "__main__":
    kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    worst = 0.0
    for k in range(1, kmax + 1):
        qb, nv = q_brute(k)
        qd, _, _ = q_of_k(k, mode="float", reduce=True)
        qn, _, _ = q_of_k(k, mode="float", reduce=False)
        d1 = abs(qb - qd)
        d2 = abs(qb - qn)
        worst = max(worst, d1, d2)
        print(f"k={k:2d} vectors={nv:8d} brute={qb:.15f} dp={qd:.15f} dp_noreduce={qn:.15f} "
              f"diff={d1:.1e}/{d2:.1e}")
    print("worst diff", worst, "OK" if worst < 1e-12 else "MISMATCH")
