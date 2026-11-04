"""Exact-ish computation of p(k) (Green problem 87 / Cameron's conjecture).

p(k) = lim_{n->inf} P(random permutation of [n] has an invariant set of size k)
     = P(the random partition with Pois(1/i) parts of size i, i=1..k, has a
         sub-partition of total size k)            (Britnell-Wildon, Prop. 1)

q(k) = 1 - p(k) = P(no sub-multiset of the cycle lengths sums to k).

DP: process lengths i = 1..k.  State = set of achievable subset sums in [0,k]
(an int bitmask).  The count b_i ~ Pois(1/i) is capped at m_i = k // i, since
more copies cannot change the truncated set; the capped tail gets the Poisson
tail mass.  States containing k are dead (they fix a k-set) and are dropped.
Reduction (--reduce, default on): after processing length i, a sum s with
k - s in [1, i] can never be completed by lengths > i, so bits s >= k - i are
cleared, merging states.  --noreduce keeps every bit (for cross-checking).

Arithmetic modes:
  --float      doubles (fast, exploration)
  --dps D      mpmath with D digits (slow)
  --fixed F    fixed-point integers with F fractional bits (default, F=128).
               Every truncation (constants, products) loses < 2^-F and all
               weights are <= 1, so |error in q| <= (#truncations) * 2^-F.
               The reported bound is that count times 2^-F.
"""
import argparse
import math
import sys
import time


def count_probs(i, m, one, E, fact, div):
    """P(min(Pois(1/i), m) = j) for j = 0..m."""
    ei = E(i)
    probs = []
    acc = 0
    for j in range(m):
        pj = div(ei, i ** j * fact(j))
        probs.append(pj)
        acc += pj
    probs.append(one - acc)
    return probs


def q_of_k(k, mode="fixed", fixed_bits=128, reduce=True, verbose=False):
    """Returns (q, max_states, error_bound).  q is a float / mpf / Fraction."""
    ntrunc = 0
    if mode == "float":
        one = 1.0
        E = lambda i: math.exp(-1.0 / i)
        fact = math.factorial
        div = lambda a, b: a / b
        mul = lambda a, b: a * b
    elif mode == "mp":
        from mpmath import mpf, exp, factorial
        one = mpf(1)
        E = lambda i: exp(-mpf(1) / i)
        fact = factorial
        div = lambda a, b: a / b
        mul = lambda a, b: a * b
    else:  # fixed point
        from mpmath import mp, mpf, exp, floor
        F = fixed_bits
        one = 1 << F
        # Evaluate every constant INSIDE the high-precision context (a lazily
        # called closure would run at the default 15 digits: review bug 11-04).
        with mp.workdps(F // 3 + 30):
            etab = [None] + [int(floor(exp(-mpf(1) / i) * (1 << F))) for i in range(1, k + 1)]
        E = lambda i: etab[i]
        fact = math.factorial
        div = lambda a, b: a // b
        mul = lambda a, b: (a * b) >> F
    full = (1 << (k + 1)) - 1
    kbit = 1 << k
    states = {1: one}  # achievable sums = {0}
    max_states = 1
    for i in range(1, k + 1):
        m = k // i
        probs = count_probs(i, m, one, E, fact, div)
        ntrunc += 2 * m + 3  # E(i), the m divisions (<2 ulp each), the tail
        keep = ((1 << (k - i)) - 1) if reduce else full
        new = {}
        for S, pr in states.items():
            T = S
            for j in range(m + 1):
                if j > 0:
                    T = (T | (T << i)) & full
                if T & kbit:
                    break  # k reachable with j copies -> also with more copies
                U = T & keep
                w = mul(pr, probs[j])
                ntrunc += 1
                if U in new:
                    new[U] += w
                else:
                    new[U] = w
        states = new
        if len(states) > max_states:
            max_states = len(states)
        if verbose:
            print(f"  i={i:3d} states={len(states)}", file=sys.stderr)
    q = sum(states.values())
    if mode == "fixed":
        from fractions import Fraction
        q = Fraction(q, 1 << fixed_bits)
        bound = Fraction(ntrunc, 1 << fixed_bits)
    elif mode == "float":
        bound = ntrunc * 2.0 ** -52
    else:
        from mpmath import mpf, mp
        bound = ntrunc * mpf(10) ** (-mp.dps + 1)
    return q, max_states, bound


def fmt(p, mode, digits=30):
    if mode == "float":
        return f"{p:.16f}"
    if mode == "mp":
        from mpmath import nstr
        return nstr(p, digits)
    # Fraction -> decimal string with `digits` decimals, truncated
    from fractions import Fraction
    sign = "-" if p < 0 else ""
    p = abs(p)
    ip = p.numerator // p.denominator
    frac = p - ip
    digs = (frac.numerator * 10 ** digits) // frac.denominator
    return f"{sign}{ip}.{digs:0{digits}d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kmax", type=int)
    ap.add_argument("--kmin", type=int, default=1)
    ap.add_argument("--float", action="store_true")
    ap.add_argument("--dps", type=int, default=0, help="use mpmath with this many digits")
    ap.add_argument("--fixed", type=int, default=128, help="fixed-point fractional bits")
    ap.add_argument("--noreduce", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.float:
        mode = "float"
    elif a.dps:
        mode = "mp"
        from mpmath import mp
        mp.dps = a.dps
    else:
        mode = "fixed"
    out = open(a.out, "a") if a.out else None
    prev = None
    for k in range(a.kmin, a.kmax + 1):
        t0 = time.time()
        q, ms, bound = q_of_k(k, mode=mode, fixed_bits=a.fixed, reduce=not a.noreduce,
                              verbose=a.verbose)
        p = 1 - q
        dt = time.time() - t0
        flag = ""
        if prev is not None and p >= prev:
            flag = "  <-- NOT DECREASING"
        line = (f"k={k:3d} p(k)={fmt(p, mode)} err<={float(bound):.1e} "
                f"max_states={ms} time={dt:.1f}s{flag}")
        print(line, flush=True)
        if out:
            out.write(line + "\n")
            out.flush()
        prev = p


if __name__ == "__main__":
    main()
