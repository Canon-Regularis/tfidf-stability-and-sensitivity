# A2. Tie-breaking and decision discontinuities

Companion to README §4.5 and §7.3/7.4.

**The question.** Deterministic tie-breaking makes the ranking a discontinuous
function of the scores. A2 asks how much that matters, *independent of numerical
error*; the phrase "independent of" is what has to be earned.

## How the independence is earned

The argument is structural rather than statistical. `rank_all_operators` shares
**one `sorted_scores` object** across π, π_score and π_alt, so there is no
arithmetic between the three operators that could differ. A disagreement
therefore has one possible cause.

`scripts/run_tie_break_ablations.py` does not take this on trust: before
reporting anything it checks that all three operators hold the **same score
array object**, and **aborts** if not, because every disagreement rate below
that point would be uninterpretable.

Identity rather than value equality, which is what it checked until it was
corrected. `rank_all_operators` builds one `sorted_scores` tuple and hands it to
every operator, so an elementwise `same_bits` comparison against
`rankings["pi"].sorted_scores` compared that object with itself and was true on
every input -- the abort was unreachable. Nor would it have caught the
regression it exists for: with the sharing removed each operator recomputes the
same sort of the same scores, giving three distinct but bit-identical tuples.
`tests/test_ranking_tie_breaks.py::test_every_operator_sees_one_and_the_same_score_array`
sets the standard, and says why: "two equal arrays computed twice would leave a
second explanation".

The complementary check runs from the other side: when all scores are distinct,
π = π_score = π_alt exactly. That validates A2's premise directly, since any
operator disagreement is then attributable to ties alone.

## The three operators

```text
π       = (popularity, rating, engagement)
π_score = ()                                  the empty-priority special case
π_alt   = (engagement, rating, popularity)
```

`π_score` runs π's code path with an empty priority list. The identifier is
appended implicitly to all three and is never permutable; if it moved, the order
would stop being total.

§4.5 never says *which* reordering π_alt uses. It is pinned to the **reversal**
of π's priority (the antipode, maximising distance from π), with the full 3!
sweep available in `configs/ablations.yaml`. This affects published disagreement
rates, so it must be stated rather than left implicit. ([G15](spec_addenda.md#g15))

## Measured

40 leave-one-out queries on `synthetic_tiny` (§7.1's protocol; see
[G25](spec_addenda.md#g25), the numbers change substantially under a different
one), top-k **set** disagreement:

| pair | k=1 | k=5 | k=10 | k=20 | k=50 |
| --- | --- | --- | --- | --- | --- |
| π vs π_alt | **10.0%** | 0.0% | 0.0% | 7.5% | 2.5% |
| π vs π_score | **50.0%** | 0.0% | 10.0% | 0.0% | 2.5% |

**Disagreement is concentrated at k=1.** The top document frequently sits in an
exact-tie block, so which document is returned first is decided entirely by the
tie-break. That is the decision-level discontinuity A2 names, at the rank where a
recommender's output is most visible.

Stratified by margin band (§7.3), the separation is complete. π vs π_score at
k = 1:

| band | disagreement | n |
| --- | --- | --- |
| `exact_tie` | **100.0%** | 20 |
| `(100·τ, ∞)` | **0.0%** | 20 |

Every query with an exact tie at rank 1 disagreed; no query without one did.
Since the operators provably consume bit-identical scores, ties are the sole
cause of the disagreement.

Every rate is reported with its denominator. A rate without its `n` cannot be
distinguished from noise over three queries.

Unlike A1's transition curve, degenerate queries are **included** here. G3
excludes them from margin distributions but keeps them in ablations, and rightly:
a zero-score query is ranked purely by attributes, which makes it the *most*
informative case for A2 rather than a nuisance.

## Ordering distance

`kendall_fks` implements the FKS generalised Kendall distance `K^(p)` with
`p = 1/2`, normalised to `[0, 1]`.

**It is a near-metric.** An early claim that it satisfies the triangle inequality
at `p = 1/2` was wrong. Violations were measured at *every* `p`, growing with
`p`; the implementation was verified against an independently written reference
(they agree exactly), which puts the error in the claim rather than the code.
Witness: A=[3,1,0], B=[5,3,4], C=[5,4,2] gives d(A,B)=6, d(B,C)=2, d(A,C)=12.
`p = 1/2` is retained on **bias** grounds rather than metric grounds.
([G2](spec_addenda.md#g2))

`kendall_intersection` restricts to shared elements, so it cannot detect
membership change at all; quoting it alone would understate the very effect §7.3
measures. `TopKComparison` reports all measures together for that reason.

## Tie groups, and why there are three

§2.3.3's tie ball is **not transitive**, so it is not a partition and cannot be
used as one. The adversarial witness is the dyadic ladder `sᵢ = i · 2⁻²⁰`, where
every value and every difference is exactly representable, so the demonstration
has no floating-point content: with `τ = 2⁻²⁰`, document 2 is in `ball(1)` but
not in `ball(0)`.

On that ladder one chain swallows the whole thing while cliques see only adjacent
pairs, giving `ρ = n/2`. One ulp below τ the ladder shatters into singletons and
`ρ = 1`: a decision discontinuity *in the diagnostic itself*.
([G1](spec_addenda.md#g1))

`TauExceedsScoreRangeWarning` is emitted once from the index constructor, never
per ball: `pyproject.toml` sets `filterwarnings = ["error"]`, so a per-ball
warning would abort a τ-sweep on its first call.

## §7.4. The near-tie case study

§7.4 says two documents are "**identified** such that `|s_A − s_B| ≤ τ`".
Identification is the only route available, because a fine near-tie **cannot be
manufactured by editing text**. §2.2's `tf = count/L` makes a single-token edit a
`1/(L+1)` *relative* perturbation, so a separation of 1e−9 would need a document
of roughly a billion tokens. The synthetic twin mechanism reaches 1e−3 to 1e−1
and cannot go finer; this is a structural consequence of the specification rather
than a limitation of the generator.

What occurs instead: adjacent gaps are either **exactly zero** (17% to 18% of
pairs) or above ~1e−9. The interval in between is *empty*. So at fine τ the
near-tie regime **is** the exact-tie regime, and §7.3's `m_k ≤ τ` stratum
measures the tie-break rather than numerical error. ([G22](spec_addenda.md#g22))

`datasets.synthetic.find_near_ties` performs the search. On `synthetic_tiny` the
tightest gap it finds is **exactly 0**, a twin pair tying at rank 1, and the
closest strictly-positive gap is 1.2e−5.

## Reproducing

```bash
python scripts/run_stability_profile.py --dataset synthetic_tiny -o reports/  # derives the tau band
python scripts/run_tie_break_ablations.py --dataset synthetic_tiny --tau 4.8e-13 -o reports/
python scripts/make_figures.py --reports reports/                             # fig_ablation
```

`--tau` is required; there is no default anywhere in the repository. Any value in
the derived band gives an identical answer; see [G23](spec_addenda.md#g23).
`fig_ablation.png` falsifies A2 if every bar is zero.
