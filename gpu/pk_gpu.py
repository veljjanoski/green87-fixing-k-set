"""GPU (CuPy) version of pk_dp.py: p(k) = lim P(random permutation fixes a k-set),
with checkpoints and resume.  Same DP as pk_dp.py (see its docstring):

  q(k) = 1 - p(k) = P(no sub-multiset of {Pois(1/i) copies of i, i<=k} sums to k)

State = bitmask of achievable sums (bits 0..k, so k <= 127 with two uint64 words
lo/hi), value = probability (float64).  Step i: counts capped at m = k // i, the
cumulative union T_j = S u (S+i) u ... u (S+j i) is computed in a kernel, states
reaching k die, survivors emit (T_j & keep, val * P(count=j)) where keep clears
bits >= k - i (sums no later part can complete).  Emitted pairs are merged by
sort + segmented (Kahan-Babuska) sum.  After step i only k - i bits are live, so
at the heavy steps (i ~ k/2) keys fit one uint64 and cupy.argsort is used;
early steps (k - i > 64) use lexsort on (lo, hi) and are small.

Error bound (float64, u = 2^-53): P(count=j) are rounded from 40-digit mpmath
values (<= u relative), each product adds <= u, each Kahan-Babuska segment sum
adds <= 2u * (segment mass) + O(n u^2); every weight is <= 1 and later factors
are <= 1, so |error in q| <= u * (4 * #merges + 2), which the run reports.

Checkpoints: after a step, if --ckpt-minutes elapsed (or --ckpt-every-step),
the state (lo, hi, vals, meta) is saved to <ckpt-dir>/k<K>.npz atomically.  A
file named STOP in the run directory, Ctrl-C, or a CUDA out-of-memory error
also save a checkpoint and exit.  On start, finished k's are read from the
results file and skipped; an existing k<K>.npz resumes k at its next step.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import cupy as cp

U = 2.0 ** -53
MASK64 = (1 << 64) - 1

KERNELS = r"""
extern "C" {
// T <- (T | (T << i)) & full ; mark dead if bit kpos set.  128-bit as (lo, hi).
__global__ void step_T(unsigned long long* tlo, unsigned long long* thi,
                       unsigned char* alive, const long long n, const int i,
                       const unsigned long long full_lo, const unsigned long long full_hi,
                       const int kpos)
{
    long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (t >= n || !alive[t]) return;
    unsigned long long lo = tlo[t], hi = thi[t], slo, shi;
    if (i < 64) { slo = lo << i; shi = (hi << i) | (lo >> (64 - i)); }
    else        { slo = 0ULL;    shi = lo << (i - 64); }
    lo = (lo | slo) & full_lo;
    hi = (hi | shi) & full_hi;
    tlo[t] = lo; thi[t] = hi;
    unsigned long long kb = (kpos < 64) ? ((lo >> kpos) & 1ULL) : ((hi >> (kpos - 64)) & 1ULL);
    if (kb) alive[t] = 0;
}
// Segmented Kahan-Babuska (Neumaier) sums over sorted runs.  One thread per
// segment, sequential and deterministic.
__global__ void segsum(const double* vals, const long long* starts, const long long nseg,
                       const long long n, double* out, int* maxseg)
{
    long long s = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    long long a = starts[s], b = (s + 1 < nseg) ? starts[s + 1] : n;
    double sum = 0.0, c = 0.0;
    for (long long t = a; t < b; ++t) {
        double x = vals[t];
        double tt = sum + x;
        if (fabs(sum) >= fabs(x)) c += (sum - tt) + x; else c += (x - tt) + sum;
        sum = tt;
    }
    out[s] = sum + c;
    atomicMax(maxseg, (int)(b - a));
}
}
"""
_mod = cp.RawModule(code=KERNELS)
_step_T = _mod.get_function("step_T")
_segsum = _mod.get_function("segsum")
THREADS = 256


def grid(n):
    return ((n + THREADS - 1) // THREADS,)


def split_mask(nbits):
    """(lo, hi) uint64 masks for bits 0..nbits-1."""
    if nbits <= 0:
        return 0, 0
    if nbits <= 64:
        return (1 << nbits) - 1, 0
    return MASK64, (1 << (nbits - 64)) - 1


def count_probs(i, m):
    """P(min(Pois(1/i), m) = j), j = 0..m, correctly rounded from 40 digits."""
    from mpmath import mp, mpf, exp, factorial
    with mp.workdps(40):
        ei = exp(-mpf(1) / i)
        ps = [ei / (mpf(i) ** j * factorial(j)) for j in range(m)]
        tail = 1 - sum(ps)
        return [float(x) for x in ps] + [float(tail)]


def reduce_states(box, stats):
    """Sort by key and sum values of equal keys.  box = [lo, hi, w] (hi may be
    None); the box is emptied so the unsorted arrays are freed after sorting."""
    lo, hi, w = box
    box.clear()
    n = len(w)
    if hi is None:
        order = cp.argsort(lo)
    else:
        order = cp.lexsort(cp.stack([lo, hi]))
    lo = lo[order]
    w = w[order]
    if hi is not None:
        hi = hi[order]
    del order
    flag = cp.empty(n, dtype=cp.bool_)
    flag[0] = True
    cp.not_equal(lo[1:], lo[:-1], out=flag[1:])
    if hi is not None:
        flag[1:] |= hi[1:] != hi[:-1]
    starts = cp.flatnonzero(flag).astype(cp.int64)
    del flag
    nseg = len(starts)
    out = cp.empty(nseg, dtype=cp.float64)
    maxseg = cp.zeros(1, dtype=cp.int32)
    _segsum(grid(nseg), (THREADS,), (w, starts, np.int64(nseg), np.int64(n), out, maxseg))
    stats["merges"] += 1
    stats["maxseg"] = max(stats["maxseg"], int(maxseg[0]))
    lo2 = lo[starts]
    hi2 = hi[starts] if hi is not None else None
    return lo2, hi2, out


def merge(pending, stats):
    """pending: list of (lo, hi, w) pieces (the old accumulator may be one of
    them).  The list is emptied and no reference to the pieces or to the
    concatenated arrays is kept here, so reduce_states can free them early
    (review 11-04: ~50-70 B/state saved at the peak merge)."""
    hi_none = pending[0][1] is None
    box = [cp.concatenate([p[0] for p in pending]),
           None if hi_none else cp.concatenate([p[1] for p in pending]),
           cp.concatenate([p[2] for p in pending])]
    pending.clear()
    return reduce_states(box, stats)


def run_step(k, i, S_lo, S_hi, vals, budget, stats):
    n = len(vals)
    m = k // i
    probs = count_probs(i, m)
    T_lo = S_lo.copy()
    T_hi = S_hi.copy() if S_hi is not None else cp.zeros(n, dtype=cp.uint64)
    alive = cp.ones(n, dtype=cp.uint8)
    full_lo, full_hi = split_mask(k + 1)
    nb_next = k - i
    keep_lo, keep_hi = split_mask(nb_next)
    acc = None
    pending, pend_n = [], 0
    for j in range(m + 1):
        if j > 0:
            _step_T(grid(n), (THREADS,), (T_lo, T_hi, alive, np.int64(n), np.int32(i),
                                          np.uint64(full_lo), np.uint64(full_hi), np.int32(k)))
        idx = cp.flatnonzero(alive)
        if len(idx) == 0:
            break
        U_lo = T_lo[idx] & np.uint64(keep_lo)
        U_hi = (T_hi[idx] & np.uint64(keep_hi)) if nb_next > 64 else None
        w = vals[idx] * probs[j]
        del idx
        pending.append((U_lo, U_hi, w))
        pend_n += len(w)
        del U_lo, U_hi, w
        if pend_n >= budget:
            if acc is not None:
                pending.insert(0, acc)
            acc = None
            acc = merge(pending, stats)
            pend_n = 0
    del T_lo, T_hi, alive
    if pending:
        if acc is not None:
            pending.insert(0, acc)
        acc = None
        acc = merge(pending, stats)
    return acc


def save_ckpt(path, k, i, S_lo, S_hi, vals, stats, t_used):
    tmp = path + ".tmp.npz"
    meta = dict(k=k, i=i, stats=stats, t_used=t_used, n=int(len(vals)))
    np.savez(tmp, lo=cp.asnumpy(S_lo), hi=(cp.asnumpy(S_hi) if S_hi is not None else np.zeros(0, np.uint64)),
             vals=cp.asnumpy(vals), meta=json.dumps(meta))
    os.replace(tmp, path)


def load_ckpt(path):
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    S_lo = cp.asarray(z["lo"])
    S_hi = cp.asarray(z["hi"]) if len(z["hi"]) else None
    vals = cp.asarray(z["vals"])
    return meta, S_lo, S_hi, vals


class Stop(Exception):
    pass


def run_k(k, args, log):
    ckpt = os.path.join(args.ckpt_dir, f"k{k}.npz")
    t0 = time.time()
    if os.path.exists(ckpt):
        meta, S_lo, S_hi, vals = load_ckpt(ckpt)
        assert meta["k"] == k
        i0 = meta["i"] + 1
        stats = meta["stats"]
        t_used = meta["t_used"]
        log(f"k={k}: resumed from step {meta['i']} ({meta['n']} states, {t_used:.0f}s used)")
    else:
        S_lo = cp.ones(1, dtype=cp.uint64)  # {0}
        S_hi = None
        vals = cp.ones(1, dtype=cp.float64)
        i0 = 1
        stats = dict(merges=0, maxseg=0, max_states=1)
        t_used = 0.0
    last_ckpt = time.time()
    for i in range(i0, k + 1):
        ts = time.time()
        try:
            if args.interrupt_at_step and i == args.interrupt_at_step:
                raise KeyboardInterrupt  # debug: simulate Ctrl-C inside a step
            S_lo, S_hi, vals = run_step(k, i, S_lo, S_hi, vals, args.budget, stats)
        except (cp.cuda.memory.OutOfMemoryError, MemoryError, KeyboardInterrupt) as e:
            # run_step did not return, so S_lo/S_hi/vals still hold the state after step i-1
            # (thrust raises a plain MemoryError when it cannot allocate)
            why = ("interrupted (Ctrl-C)" if isinstance(e, KeyboardInterrupt)
                   else "CUDA out of memory (rerun with a smaller --budget)")
            log(f"k={k} step {i}: {why}; saving checkpoint and exiting")
            save_ckpt(ckpt, k, i - 1, S_lo, S_hi, vals, stats, t_used + time.time() - t0)
            raise Stop()
        n = int(len(vals))
        stats["max_states"] = max(stats["max_states"], n)
        if args.verbose:
            log(f"  k={k} i={i:3d} states={n} step={time.time() - ts:.1f}s merges={stats['merges']}")
        stop = os.path.exists(args.stop_file) or (args.stop_at_step and i == args.stop_at_step)
        if i == k:
            # a stop request on the last step: finish k, write its result, then stop
            args.stop_after_k = args.stop_after_k or stop
            break
        want_ckpt = args.ckpt_every_step or stop or time.time() - last_ckpt > 60 * args.ckpt_minutes
        if want_ckpt:
            save_ckpt(ckpt, k, i, S_lo, S_hi, vals, stats, t_used + time.time() - t0)
            last_ckpt = time.time()
            if args.verbose:
                log(f"  checkpoint k={k} i={i} saved")
        if stop:
            if os.path.exists(args.stop_file):
                os.remove(args.stop_file)
            log(f"k={k}: stopped after step {i} (checkpoint saved)")
            raise Stop()
    assert len(vals) == 1
    q = float(vals[0])
    p = 1.0 - q
    bound = U * (4 * stats["merges"] + 2)
    # the checkpoint is removed by main() after the result line is written
    return p, bound, stats, t_used + time.time() - t0, ckpt


def done_ks(path):
    import re
    done = set()
    if os.path.exists(path):
        for line in open(path):
            m = re.match(r"k=\s*(\d+) p\(k\)=", line)
            if m:
                done.add(int(m.group(1)))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kmax", type=int)
    ap.add_argument("--kmin", type=int, default=1)
    ap.add_argument("--out", default="results_gpu.txt")
    ap.add_argument("--ckpt-dir", default="ckpt")
    ap.add_argument("--ckpt-minutes", type=float, default=10.0)
    ap.add_argument("--ckpt-every-step", action="store_true")
    ap.add_argument("--stop-file", default="STOP")
    ap.add_argument("--stop-at-step", type=int, default=0, help="debug: stop k after this step")
    ap.add_argument("--interrupt-at-step", type=int, default=0,
                    help="debug: raise KeyboardInterrupt at the start of this step")
    ap.add_argument("--budget", type=float, default=5e7, help="pending elements before a merge")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    assert args.kmax <= 127
    args.budget = int(args.budget)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    logf = open(os.path.splitext(args.out)[0] + ".log", "a")

    def log(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    done = done_ks(args.out)
    args.stop_after_k = False
    prev = None
    for k in range(args.kmin, args.kmax + 1):
        if k in done:
            continue
        try:
            p, bound, stats, dt, ckpt = run_k(k, args, log)
        except Stop:
            return 0
        except KeyboardInterrupt:  # outside a step (inside a step run_k saves a checkpoint)
            log("interrupted between steps; the last checkpoint (if any) is kept; rerun to resume")
            return 1
        flag = "  <-- NOT DECREASING" if (prev is not None and p >= prev) else ""
        line = (f"k={k:3d} p(k)={p:.17f} err<={bound:.1e} max_states={stats['max_states']} "
                f"maxseg={stats['maxseg']} merges={stats['merges']} time={dt:.1f}s{flag}")
        log(line)
        with open(args.out, "a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        if os.path.exists(ckpt):
            os.remove(ckpt)
        prev = p
        cp.get_default_memory_pool().free_all_blocks()
        if args.stop_after_k:
            if os.path.exists(args.stop_file):
                os.remove(args.stop_file)
            log(f"k={k}: stop requested; result written, stopping")
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
