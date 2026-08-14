# A1. Margins and ranking stability

Companion to README §4.4 and §7.2/7.3.

**The question.** Under a bounded perturbation of the scores, when does the
top-k set survive? §4.4 answers with a sufficient condition. This page records
what was measured, and the two ways that answer is easy to misread.

## The certificate

If every score moves by less than `m_k / 2`, the top-k **set** is unchanged.
`perturbation/score_bounds.py::certified_radius` returns two radii:

- `set_radius = m_k / 2`: membership, from the gap *at* the boundary;
- `order_radius = m_min^top / 2`: the internal ordering, from the smallest gap
  *inside* the top-k.

**Neither dominates the other.** An early draft claimed `order_radius ≤
set_radius`; that is false. They are computed from disjoint sets of gaps
(`m_min^top` covers gaps within the top-k, `m_k` the gap at the boundary), so
either can be smaller. `joint_radius` is their minimum, for when both properties
are wanted.

At `k = 1` the internal minimum is over an empty set. It returns `NaN` with
`defined = False`. Returning `+inf` would silently claim "no constraint" and
would pollute any percentile computed over it. ([G16](spec_addenda.md#g16))

## Tight in the worst case, conservative on average

These are different claims and neither implies the other. Conflating them is the
main way A1 gets misreported.

**Worst case: the bound is exact.** There is a perturbation of size
`m_k/2 + δ` that flips the pair: push the rank-k score down and the rank-(k+1)
score up. The witness is built from dyadic rationals so every step is exact in
binary64 and the demonstration has no floating-point content of its own: scores
`0.5` and `0.25`, `m = 0.25`, `ε = 0.125 + 2⁻³⁰`. So `m_k/2` cannot be improved.

**Average case: nothing happens until well past it.** Under *random*
perturbation the adversarial configuration is a measure-zero corner of the
perturbation cube. Measured (`analysis/stability_profile.py::transition_curve`,
k=10, 35 of 40 §7.1 leave-one-out queries × 40 trials; the other five were
excluded for `m_k = 0`):

| ε / (m_k/2) | flip rate |
| --- | --- |
| 0.25 to 1.01 | **0.00%** |
| 1.10 | 0.50% |
| 2.00 | 22.50% |
| 5.00 | 62.86% |
| 20.00 | 92.71% |

A paper reporting only the second would understate the risk; only the first would
suggest rankings are far more fragile than they are.

## Auditing the certificate

`certificate_audit` reports a 2×2 table rather than an accuracy. Accuracy is the
wrong summary: it would reward a certificate that always said "no".

| outcome | top-k unchanged | top-k changed |
| --- | --- | --- |
| **certified stable** | 82 | **0** |
| not certified | 408 | 310 |

- **Sound**: `certified_changed` must be zero. Any other value falsifies §4.4;
  it is a bug or a broken proof, never a statistic. `scripts/run_stability_profile.py`
  **exits non-zero** if it is non-zero, and CI runs it on every push.
- **Conservative**: 56.8% of *uncertified* cases were unchanged anyway. Reporting
  this is what stops "not certified" being read as "will break".

One subtlety decides whether this is a theorem check or a flaky annoyance: the
audit compares the **realised** delta rather than the drawn one. `fl(s + d)`
rounds, so actual movement can exceed `|d|` by half an ulp, and the theorem is
about the movement that happened.

## The boundary with A2

Queries with `m_k = 0` are **excluded** from the transition curve and counted
separately. At an exact tie no perturbation is needed to change the outcome; the
result is decided entirely by the tie-break, which is A2's regime. Averaging
them into an A1 curve would let a tie-break effect be read as a
numerical-stability effect. G3 requires this exclusion for margin distributions;
the same logic applies here, and `test_exact_tie_queries_are_excluded_from_the_a1_curve`
enforces it.

## Why numerical error is not the threat

The measured arithmetic noise floor is `η ≈ 1.665e−16`, giving `τ_floor = 2η ≈
3.3e−16`. The smallest strictly-positive score gap observed is `≈ 7e−10`.

**Six orders of magnitude.** Numerical error never came close to closing a real
gap, so it cannot flip the ranking of two distinctly-scored documents. A1's
"bounded perturbations" are therefore entirely about *semantic* perturbation:
corpus edits, which produce `|Δs|` in the range 1e−4 to 1e−1, some thirteen
orders of magnitude above the noise floor.

This is a positive result for §6's design, and it is what makes A1 and A2 cleanly
separable rather than two names for one effect. ([G23](spec_addenda.md#g23))

## Reproducing

```bash
python scripts/run_stability_profile.py --dataset synthetic_tiny -o reports/
python scripts/make_figures.py --reports reports/     # fig_transition, fig_margins
```

`fig_transition.png` falsifies A1 if the curve rises left of the dashed line.
