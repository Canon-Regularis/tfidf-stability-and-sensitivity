# Mathematical formulation, as implemented

Companion to README §2. This page records the places where implementing §2
required a decision the specification did not make, and where the obvious
implementation would have been wrong.

## The pipeline

```text
text → normalise → tokenise → stopwords → lemmatise → n-grams → features
features → vocabulary → df → idf → tf → tf-idf → L2 norm
(query, document) → cosine → score → sort key → ranking → margins → tie groups
```

Every stage is a separate module under `src/tfidf_stability/`, and every
intermediate is retained rather than fused — §1.2 requires them to be
inspectable, and `TfidfModel.intermediates(i)` returns them for any document.

## §2.1 — vocabulary, document frequency, IDF

**Vocabulary order is UTF-8 byte order, not Python string order.** Python
compares `str` by code point; the C++ mirror compares by `memcmp`. Those agree
for ASCII and diverge outside the BMP, so the Python side sorts on the encoded
bytes to make the two literally the same relation. `vocabulary.py::_byte_key`.

**IDF is computed in `Decimal` at 60 digits, then rounded once.** This is the
single most consequential decision in the repository.

`math.log` is *not* correctly rounded. Measured against the correctly-rounded
value, it differs in **15.16% of IDF entries** (44.5% of raw logarithms), and
UCRT, glibc and Apple libm disagree with each other. Left alone, Linux and
Windows would produce different weights, different norms, different scores and
different rankings — the reproducibility claim would simply be false.

So `idf.py` computes each value via `decimal.Decimal.ln()` and rounds once to
binary64, and the C++ core **never sees a logarithm**. It receives IDF as data.
What remains in C++ is only `+ − × ÷ √`, which IEEE-754 requires to be correctly
rounded, so every platform must agree. ([G13](spec_addenda.md#g13))

## §2.2 — the TF-IDF embedding

`tf = count / L` with `L` the document length in tokens. One IEEE division, so
exact given exact inputs.

This normalisation has a consequence the paper does not draw out: **a
single-token edit is a `1/(L+1)` relative perturbation applied to every
coordinate at once**, not a small additive nudge to one. It is why fine near-ties
cannot be built by editing text, and why §7.4 must search for one.
([G22](spec_addenda.md#g22))

n-grams are joined with `\x1f` (unit separator), a character that cannot occur in
normalised text, so a bigram can never collide with a unigram containing the
join character.

## §2.3 — cosine, ranking, and the stability quantities

### Cosine

`dot / (‖u‖ ‖v‖)`, in that order, with no rescaling. A hypot-style scaled
formulation would be more numerically robust but would produce *different
digits*, and §6 is explicit that no stabilising transformations are applied.

The `[0, 1]` guarantee depends on the vectors being non-negative, which
`check_non_negative` enforces rather than assumes.

**Where scale invariance actually breaks.** `l2_norm` squares before summing, so
`cos(αu, v) = cos(u, v)` fails once `|x| ≲ √DBL_MIN ≈ 1.49e−154` and the squares
underflow. The code is correct per §6; a test that claimed unconditional scale
invariance was over-claiming and was corrected. ([G18](spec_addenda.md#g18))

### The ranking operator (§2.3.1)

Sort key: `(−score, rank₁, …, rankₘ, id_rank)`, ascending lexicographic.

**Attributes are rank-encoded to dense `int32` once per corpus.** Direction
(`desc`/`asc`) and missing-value placement are baked into the rank rather than
applied per comparison. Three consequences:

- the comparator is plain lexicographic — no direction branch, no missing branch;
- **floating point is removed from the tie-break entirely**, which is what G8
  actually requires, achieved not by care but by there being none left;
- NaN cannot enter through the tie-break at all — a type-level guarantee.

Ratings use the exact integer pair `(2 × Σ rating, count)` and compare by
cross-multiplication. G8's stated justification — that equal means might compare
unequal — does not survive contact with the data: with half-star ratings the sum
is exact and IEEE mandates correctly-rounded division, so equal means *do* give
identical doubles. The real hazard is the opposite: two **distinct** means
colliding onto one double. The resolution is right; the reason needed restating.

**Why sort stability is irrelevant.** Unique identifiers make the key injective,
so no two elements compare equal, so "stable" quantifies over the empty set. A
finite linear order admits exactly one order isomorphism onto `(0..N−1, <)`, so
the output is unique. The stronger test — that the result is independent of the
*input* order — is what the suite asserts, because it catches a non-total
comparator that `sort == stable_sort` can miss by luck.

### Margins (§2.3.2)

`m_k = s_(k) − s_(k+1)`, and the flip radius is `m_k / 2` — exact, since division
by two only decrements the exponent.

`m_k` depends **only on the sorted score multiset**, so it is identical under π,
π_score and π_alt. That is precisely what makes A1 and A2 independent questions,
and it is what lets a disagreement rate be stratified by margin without
circularity. `rank_all_operators` shares one `sorted_scores` object across the
three operators so this is true by construction rather than by coincidence.

### Tie groups (§2.3.3)

The paper defines one object; the implementation provides three, because the
paper's definition is **not transitive** and so is not a partition.
([G1](spec_addenda.md#g1))

| object | definition | cost | partition? |
| --- | --- | --- | --- |
| `tie_ball(j, τ)` | `{i : \|sᵢ − s_j\| ≤ τ}`, verbatim §2.3.3 | O(log N) | no |
| `tie_chains(τ)` | single-linkage: every *adjacent* gap ≤ τ | O(N) | **yes** |
| `tie_cliques(τ)` | complete-linkage: maximal intervals of *diameter* ≤ τ | O(N) | no |

plus `ρ(τ) = |largest chain| / |largest clique| ≥ 1`, the chain-inflation ratio.

**The ball must not binary-search for `c ± τ`.** Those bounds round, so the
realised predicate would differ from the pinned `|sᵢ − c| ≤ τ` exactly at the
boundary — the only place tie groups are interesting. The search is on the
*difference* instead, which is exact and still O(log N), and works because IEEE
subtraction is monotone.

**Cliques are O(N) and provably complete.** The graph `|sᵢ − s_j| ≤ τ` is an
indifference graph, so every maximal clique is a contiguous interval of the
sorted order, and there are at most `N`. Checked against an O(N²) brute-force
enumerator for `N ≤ 12`.

## What the C++ mirror does and does not contain

Mirrored: `vectorisation/`, `similarity/`, `ranking/`. Not mirrored:
`preprocessing/`, `analysis/`, `perturbation/`, `persistence/`, `cli/`,
`datasets/` — orchestration, not hot paths. `scripts/check_layout.py` enforces
the split so it cannot drift silently.

The C++ side receives IDF and attribute ranks **as data**. It re-derives neither.
Every delicate computation happens once, in Python, in exact arithmetic.
