# Permutations fixing a k-set: p(k) for k ≤ 121

Let p(k) be the limit, as n → ∞, of the probability that a uniformly random
permutation of {1, …, n} has an invariant set of size k.  Peter Cameron
conjectured (2010) that p(k) is a decreasing function of k; this is the first
half of Problem 87 in Ben Green's *100 open problems* and Question 2 of
Eberhard–Ford–Green.  Britnell and Wildon computed p(k) for k ≤ 30 (2015) and
found it decreasing.

This repository computes p(k) exactly (up to a stated rounding bound) for all
**k ≤ 121** with a much smaller dynamic program, and finds

> **p(k) is strictly decreasing for 1 ≤ k ≤ 121.**

This is a numerical verification, not a proof: it extends the range in which
Cameron's conjecture has been checked from 30 to 121 (with derived error bounds)
and provides the values themselves.
Green and Sawhney (2026) proved p(k) = (1 + o(1)) f({log₂ k}) k^(−δ) (log k)^(−3/2)
with δ = 1 − (1 + log log 2)/log 2 ≈ 0.0861 (called α in Green's list) and an
almost constant f, which makes an increase at large k
implausible, but the monotonicity question itself remains open.

## The quantity

The cycle counts of a random permutation converge to independent Poisson
variables with means 1/i, so (this follows from Britnell–Wildon, Proposition 1)

p(k) = P( the random multiset with Pois(1/i) copies of i, i = 1..k, has a
sub-multiset with sum exactly k ).

Equivalently 1 − p(k) is the sum, over all tuples (a₁, …, a_k) of non-negative
integers for which no choice a'ᵢ ≤ aᵢ gives Σ i·a'ᵢ = k, of
Π e^(−1/i) (1/i)^(aᵢ) / aᵢ!.

## Method

`pk_dp.py` processes the cycle lengths i = 1, …, k.  The state is the set of
sums achievable so far, as a bitmask over {0, …, k}, with its probability.
For length i the count is capped at ⌊k/i⌋ (more copies cannot change the
truncated set; the cap carries the Poisson tail mass).  A state in which k
becomes achievable is discarded (it fixes a k-set).  After length i, a sum s
with k − s ≤ i can never be completed by the remaining lengths > i, so those
bits are cleared, which merges states.  The number of states is small:

| k | 30 | 50 | 80 | 100 | 115 | 121 |
|--:|--:|--:|--:|--:|--:|--:|
| peak states | 2,040 | 61,207 | 3.4 M | 33 M | 165 M | 312 M |

(the Britnell–Wildon enumeration has 12 M rows at k = 30, and their table for
k ≤ 30 took about 72 hours; all k ≤ 30 together take under a second here).  The
growth is about 1.11–1.14× per unit of k for k ≥ 50.  No
further merging is possible in this formulation: two distinct states are told
apart by a single future cycle of the right length.

**Arithmetic.**  The CPU program uses 128-bit fixed point: every constant and
product is truncated, each truncation loses < 2^(−128), all weights are ≤ 1,
so |error in p(k)| ≤ (number of truncations) · 2^(−128), which is reported per
k (≤ 2.5·10^(−31) at k = 80).  The GPU program (`gpu/pk_gpu.py`, CuPy) uses
float64 with Poisson masses rounded from 40-digit values, products, and a
segmented Kahan–Babuska sum kernel; its reported bound is 2^(−53)·(4·merges + 2)
(≤ 7.4·10^(−14) at k = 121).  The GPU merge is a deterministic sort-and-reduce,
so results are bit-reproducible, and the program checkpoints and resumes
(time-based, a `STOP` file, Ctrl-C or CUDA out-of-memory).

## Validation

* `check_brute.py`: for k ≤ 12 the DP (with and without the bit-clearing
  reduction) agrees to 5·10^(−16) with a brute force over all capped count
  vectors that builds the subset sums explicitly.
* All 30 values of Britnell–Wildon (`data/bw_table.txt`, their Appendix B)
  are reproduced to the 8 decimals they give (`compare.py`).
* `check_finite.py` computes the finite-n proportions i(n,k) exactly from the
  partitions of n and matches seven entries of their Appendix C, confirming the reading of
  "fixes a k-set" (including their non-monotone finite-n pairs such as
  i(30,9) < i(30,10)).
* CPU fixed point, CPU float64 and GPU float64 agree pairwise to 10^(−15)
  for k ≤ 80, and the GPU peak state counts equal the CPU ones for every k.
* A stop after step 35 of k = 70 followed by a resume reproduces p(70)
  bit-for-bit.
* The code was reviewed independently (read-only) before the long runs; the
  review found no algorithmic error, one precision defect in an early version
  of the CPU constants (fixed, the run redone) and two checkpoint windows
  (fixed).

## Results

`data/results.csv` has every value with its error bound, the peak state count
and the program that produced it (CPU fixed point for k ≤ 80, GPU for
81 ≤ k ≤ 121); `data/results_fixed.txt` has 30 decimals for k ≤ 80.

The smallest step is p(119) − p(120) = 3.52·10^(−4), more than nine orders
of magnitude above the error bound.  The relative step (p(k−1) − p(k))/p(k) is
0.0220 at k = 10, 0.0069 at 30, 0.0025 at 80 and 0.0018 at 121.  For comparison,
the relative decrease of k^(−δ) (log k)^(−3/2) itself is (δ + 3/(2 log k))/k ≈ 0.0033
at k = 121, about 1.8 times the observed step, and the ratio
p(k) / (k^(−δ) (log k)^(−3/2)) is 1.61, 2.47, 3.18, 3.46 at k = 10, 30, 80, 121
and still rising: these k are far from the asymptotic regime.

### p(k), 1 ≤ k ≤ 121

Values truncated to 12 decimals; bound = absolute error bound (2 significant
digits, rounded up); peak states = largest number of DP states during that k.

| k | p(k) | bound | peak states | k | p(k) | bound | peak states |
|--:|:--|:--|--:|--:|:--|:--|--:|
| 1 | 0.632120558828 | 1.8e-38 | 1 | 62 | 0.250753312906 | 2.2e-32 | 343,044 |
| 2 | 0.553739679703 | 4.4e-38 | 1 | 63 | 0.249843740392 | 2.6e-32 | 387,658 |
| 3 | 0.496583242764 | 7.6e-38 | 2 | 64 | 0.249069540174 | 3.0e-32 | 453,272 |
| 4 | 0.469557730028 | 1.2e-37 | 3 | 65 | 0.248193756044 | 3.5e-32 | 505,398 |
| 5 | 0.441457704867 | 1.7e-37 | 4 | 66 | 0.247448941405 | 4.0e-32 | 585,028 |
| 6 | 0.425058702263 | 2.4e-37 | 5 | 67 | 0.246608119861 | 4.6e-32 | 665,889 |
| 7 | 0.408481130501 | 3.3e-37 | 8 | 68 | 0.245889775462 | 5.3e-32 | 770,645 |
| 8 | 0.397277707247 | 4.3e-37 | 11 | 69 | 0.245080302716 | 6.1e-32 | 853,253 |
| 9 | 0.385164426958 | 5.8e-37 | 15 | 70 | 0.244388080680 | 6.8e-32 | 974,371 |
| 10 | 0.376871921533 | 7.6e-37 | 18 | 71 | 0.243606787313 | 8.0e-32 | 1,105,352 |
| 11 | 0.367730644609 | 1.0e-36 | 26 | 72 | 0.242939627848 | 9.0e-32 | 1,279,090 |
| 12 | 0.361194148728 | 1.3e-36 | 36 | 73 | 0.242186189755 | 1.0e-31 | 1,427,273 |
| 13 | 0.353960677106 | 1.7e-36 | 47 | 74 | 0.241539939575 | 1.2e-31 | 1,621,009 |
| 14 | 0.348550065838 | 2.2e-36 | 55 | 75 | 0.240812945535 | 1.3e-31 | 1,801,914 |
| 15 | 0.342563310269 | 2.8e-36 | 72 | 76 | 0.240189118923 | 1.5e-31 | 2,112,370 |
| 16 | 0.338072491353 | 3.6e-36 | 100 | 77 | 0.239484853262 | 1.7e-31 | 2,337,460 |
| 17 | 0.332973326458 | 4.8e-36 | 124 | 78 | 0.238881129644 | 1.9e-31 | 2,635,561 |
| 18 | 0.329075876727 | 5.9e-36 | 152 | 79 | 0.238200545199 | 2.2e-31 | 2,982,562 |
| 19 | 0.324729077728 | 7.8e-36 | 202 | 80 | 0.237614988477 | 2.5e-31 | 3,407,120 |
| 20 | 0.321324224635 | 9.6e-36 | 259 | 81 | 0.236956070800 | 3.6e-14 | 3,781,771 |
| 21 | 0.317500645638 | 1.2e-35 | 311 | 82 | 0.236388648508 | 3.7e-14 | 4,255,548 |
| 22 | 0.314498618225 | 1.6e-35 | 394 | 83 | 0.235749193871 | 3.7e-14 | 4,809,410 |
| 23 | 0.311104278302 | 2.0e-35 | 490 | 84 | 0.235199437864 | 3.8e-14 | 5,498,624 |
| 24 | 0.308422803013 | 2.4e-35 | 622 | 85 | 0.234579509675 | 3.8e-14 | 6,069,862 |
| 25 | 0.305389043808 | 3.2e-35 | 739 | 86 | 0.234044505715 | 3.8e-14 | 6,775,030 |
| 26 | 0.302953607038 | 3.9e-35 | 918 | 87 | 0.233443393936 | 3.9e-14 | 7,637,272 |
| 27 | 0.300215079050 | 4.9e-35 | 1,131 | 88 | 0.232924323763 | 3.9e-14 | 8,774,222 |
| 28 | 0.298013397414 | 6.0e-35 | 1,425 | 89 | 0.232339363261 | 4.0e-14 | 9,716,890 |
| 29 | 0.295509151492 | 7.6e-35 | 1,706 | 90 | 0.231834878087 | 4.0e-14 | 10,606,072 |
| 30 | 0.293486109613 | 9.0e-35 | 2,040 | 91 | 0.231266925577 | 4.1e-14 | 12,188,812 |
| 31 | 0.291203650329 | 1.2e-34 | 2,483 | 92 | 0.230775573307 | 4.1e-14 | 13,859,483 |
| 32 | 0.289338500867 | 1.4e-34 | 3,075 | 93 | 0.230223392560 | 4.2e-14 | 15,248,284 |
| 33 | 0.287230628595 | 1.7e-34 | 3,572 | 94 | 0.229745280761 | 4.2e-14 | 16,756,095 |
| 34 | 0.285506759415 | 2.1e-34 | 4,425 | 95 | 0.229207266337 | 4.2e-14 | 18,986,404 |
| 35 | 0.283552440980 | 2.5e-34 | 5,180 | 96 | 0.228742225015 | 4.3e-14 | 21,601,467 |
| 36 | 0.281951526246 | 3.0e-34 | 6,381 | 97 | 0.228218571926 | 4.3e-14 | 24,033,244 |
| 37 | 0.280137461178 | 3.8e-34 | 7,480 | 98 | 0.227764416032 | 4.4e-14 | 26,108,204 |
| 38 | 0.278636061662 | 4.4e-34 | 8,939 | 99 | 0.227254739973 | 4.4e-14 | 29,713,647 |
| 39 | 0.276943738160 | 5.4e-34 | 10,366 | 100 | 0.226812458991 | 4.5e-14 | 33,356,323 |
| 40 | 0.275542865608 | 6.4e-34 | 12,468 | 101 | 0.226314805681 | 4.6e-14 | 37,270,747 |
| 41 | 0.273952012649 | 7.9e-34 | 14,666 | 102 | 0.225883535926 | 4.6e-14 | 40,298,418 |
| 42 | 0.272633734999 | 9.2e-34 | 17,463 | 103 | 0.225398598589 | 4.8e-14 | 46,284,748 |
| 43 | 0.271142933576 | 1.1e-33 | 20,619 | 104 | 0.224977212473 | 5.0e-14 | 51,836,315 |
| 44 | 0.269898272254 | 1.3e-33 | 24,544 | 105 | 0.224504244575 | 5.0e-14 | 56,801,990 |
| 45 | 0.268492143087 | 1.6e-33 | 28,044 | 106 | 0.224092897023 | 5.2e-14 | 62,057,885 |
| 46 | 0.267316836233 | 1.8e-33 | 33,618 | 107 | 0.223630659952 | 5.3e-14 | 71,155,895 |
| 47 | 0.265986016564 | 2.2e-33 | 38,770 | 108 | 0.223229319953 | 5.4e-14 | 79,284,359 |
| 48 | 0.264873959354 | 2.6e-33 | 45,626 | 109 | 0.222778021909 | 5.6e-14 | 88,518,540 |
| 49 | 0.263614520260 | 3.1e-33 | 52,422 | 110 | 0.222385032728 | 5.7e-14 | 94,292,614 |
| 50 | 0.262554533301 | 3.6e-33 | 61,207 | 111 | 0.221944500286 | 5.8e-14 | 108,465,101 |
| 51 | 0.261359911879 | 4.3e-33 | 71,202 | 112 | 0.221560674991 | 5.9e-14 | 120,958,956 |
| 52 | 0.260354287482 | 5.0e-33 | 84,220 | 113 | 0.221129366869 | 6.1e-14 | 135,063,778 |
| 53 | 0.259214434986 | 5.9e-33 | 96,532 | 114 | 0.220754113877 | 6.2e-14 | 143,848,338 |
| 54 | 0.258255169167 | 6.7e-33 | 112,060 | 115 | 0.220332659945 | 6.5e-14 | 165,284,871 |
| 55 | 0.257170977789 | 8.0e-33 | 127,022 | 116 | 0.219965077103 | 6.6e-14 | 183,797,738 |
| 56 | 0.256253569264 | 9.3e-33 | 150,224 | 117 | 0.219552988829 | 6.7e-14 | 204,642,284 |
| 57 | 0.255218152291 | 1.1e-32 | 170,632 | 118 | 0.219193249731 | 6.9e-14 | 218,217,391 |
| 58 | 0.254340975024 | 1.2e-32 | 198,001 | 119 | 0.218789525572 | 7.1e-14 | 252,270,517 |
| 59 | 0.253349318930 | 1.5e-32 | 226,531 | 120 | 0.218437690731 | 7.1e-14 | 275,154,717 |
| 60 | 0.252510038653 | 1.7e-32 | 260,439 | 121 | 0.218042550710 | 7.4e-14 | 312,262,679 |
| 61 | 0.251560949279 | 2.0e-32 | 297,642 |  | | |  |

## Limits

The DP must hold every state (no two are equivalent) and their number grows by
about 1.11–1.14× per unit of k (for k ≥ 50), so memory is the limit: the CPU version was run
to k = 80, the GPU version reaches k = 121 in 24 GB (the k = 122 run failed with
an allocation error in the sort at step 60, starting from 315 M states and heading
for about 340 M).  Two observations from exploratory runs not included here:
splitting the state set by the low key bits, the only invariant of a step, is too
skewed to help (at k = 100 the largest of 4096 buckets held 21% of the states);
and an external-memory version streaming sorted runs through the GPU would, by
our estimate, reach k ≈ 130 from host RAM and k ≈ 150 from NVMe at hours per k.

Running times: CPU fixed point k = 80 in 49 s; GPU k = 80 in 0.3 s, k = 100 in
1.6 s, k = 110 in 4 min and k = 121 in 20 min (once the GPU memory is full the
time is dominated by allocator thrashing, not by arithmetic).

## Reproduce

```
python pk_dp.py 45                      # CPU, fixed point, ~2 s
python check_brute.py 10                # brute-force cross-check
python check_finite.py                  # finite-n check against Britnell–Wildon
python gpu/pk_gpu.py 100 --out results_gpu.txt   # needs CuPy + an NVIDIA GPU
python compare.py data/results_fixed.txt data/results_gpu.txt
python make_results.py                  # rebuilds data/results.csv and data/table.md
```

## References

* B. Green, *100 open problems*, Problem 87.
  https://people.maths.ox.ac.uk/greenbj/papers/open-problems.pdf
* P. Cameron, *Derangements, 2*, blog post, 15 Nov 2010.
* S. Eberhard, K. Ford, B. Green, *Permutations fixing a k-set*,
  arXiv:1507.04465 (Question 2).
* J. R. Britnell, M. Wildon, *Computing derangement probabilities of the
  symmetric group acting on k-sets*, arXiv:1511.04106 (values for k ≤ 30).
* B. Green, M. Sawhney, *The proportion of permutations fixing a k-set*,
  arXiv:2604.28116 (the asymptotic formula).

## License and acknowledgement

MIT License.  Claude (Anthropic) was used as assistant.
