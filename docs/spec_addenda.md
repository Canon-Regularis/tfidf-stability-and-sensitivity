# Specification Addenda

`README.md` (equivalently `docs/main.pdf`) is the normative specification for this
repository. Implementing it end to end surfaced a number of points that are
**underspecified**: places where two faithful readings of the text produce
different numbers, or where a quantity the paper names is not actually
well-defined in the case that matters.

Each is recorded here with a stable identifier, the ambiguity, and the resolution
adopted. Code that depends on one of these decisions cites its identifier in the
docstring, so the reasoning is never more than one grep away.

Nothing here contradicts the paper. Where the paper is definite, the paper wins;
these entries only fill gaps or make an abstract quantity concrete.

| ID | Area | Summary |
|---|---|---|
| [G1](#g1)  | §2.3.3 | Tie groups are balls, not equivalence classes |
| [G2](#g2)  | §4.5, §7.3 | Kendall distance is undefined when top-*k* sets differ |
| [G3](#g3)  | §2, §4 | Edge cases with no defined behaviour |
| [G4](#g4)  | §4.3 | The Lipschitz constant `C` is left abstract |
| [G5](#g5)  | §4.2 | The three-term bound assumes a fixed vocabulary |
| [G6](#g6)  | §2.1 | `max_features` truncation rule is unspecified |
| [G7](#g7)  | §2 | Lemmatisation is not reproducibly specified |
| [G8](#g8)  | §2.3.1 | Attribute sort directions and missing values |
| [G9](#g9)  | §2.3.3 | Precision of `score(r_j)` in the tie-group test |
| [G10](#g10)| §7.1 | Leave-one-out protocol has five unstated decisions |
| [G11](#g11)| §7.1 | Profile aggregation: text or vectors? |
| [G12](#g12)| §3 | Query IDF mapping (already specified; asserted in code) |
| [G13](#g13)| §2.1 | `log` is not correctly rounded — breaks cross-platform bit-exactness |
| [G14](#g14)| §7.1 | Query and user counts deferred to "dataset configuration" |
| [G15](#g15)| §4.5 | Which reordering `π_alt` uses is unspecified |
| [G16](#g16)| §2.3.2 | `m_min^top` is undefined at *k* = 1 |
| [G17](#g17)| §2.3.1 | Ranking an empty corpus has no named error |
| [G18](#g18)| §6 | Where low-norm cosine instability actually begins |
| [G19](#g19)| §7.1 | The candidate set varies per leave-one-out fold |
| [G20](#g20)| §7.1 | Profile aggregation is order-sensitive |
| [G21](#g21)| §7.1 | `vector_sum` and `vector_mean` give identical similarities |

---

<a id="g1"></a>
## G1 — Tie groups are balls, not equivalence classes

**Paper.** §2.3.3 defines `G_τ(j) = { i : |sᵢ − score(r_j)| ≤ τ }` and states that
"documents within a tie group are indistinguishable at the level of similarity
scores up to numerical tolerance".

**Ambiguity.** The relation `|sᵢ − s_j| ≤ τ` is reflexive and symmetric but **not
transitive**: for scores `{0, τ, 2τ}` the first and second are related, and the
second and third are related, but the first and third are not. Consequently
`G_τ(j)` is a *ball* around `score(r_j)`, not an equivalence class. Balls overlap,
they do not partition the corpus, "the tie group of document *i*" is not
well-defined, and two members of one ball may differ by as much as `2τ` — so the
quoted sentence is not strictly true as written.

**Resolution.** Implement three separately named objects and never conflate them:

1. **`tie_ball(j, τ)`** — verbatim §2.3.3. Two binary searches on the descending
   sorted score array, `O(log N)`. **This remains the primary reported object**,
   because it is what the paper defines.
2. **`tie_chains(τ)`** — the transitive closure: maximal runs whose *adjacent*
   gaps are all `≤ τ` (single-linkage). `O(N)`. This *is* an equivalence relation
   and is the correct object wherever a partition is required.
3. **`tie_cliques(τ)`** — maximal intervals of *diameter* `≤ τ` (complete
   linkage), i.e. the sets in which every pair really is mutually
   indistinguishable. `O(N)`, overlapping.

Report ball statistics (paper-faithful) alongside chain statistics (partition),
plus the diagnostic

> **chain-inflation ratio** `ρ(τ) := |largest chain| / |largest clique|`

which flags when the chosen `τ` is inducing transitive chaining and tie-group
statistics must be read with care.

**Suggested erratum.** Replace the quoted sentence with: *"Documents in a tie
group are each within τ of `score(r_j)`; two such documents may therefore differ
by up to 2τ, so the relation is not transitive and tie groups do not partition
the corpus."*

---

<a id="g2"></a>
## G2 — Ordering distance when the top-*k* sets differ

**Paper.** §4.5 asks for "a distance between orderings restricted to **tie
groups**"; §7.3 asks for "**within-top-k** reordering: an ordering distance
restricted to tie-affected subsets". These are two different problems, and only
one of them is ill-posed.

**Resolution — split them.**

**(a) Restricted to a tie group.** Well-posed. A tie group is defined on the score
vector, which `π`, `π_score` and `π_alt` all share, so both orderings contain
*exactly the same set*. Plain normalised Kendall τ distance applies:
`K(σ,σ′) = #discordant / C(|G|,2)`, computed by merge-sort inversion count in
`O(|G| log |G|)`.

**(b) Restricted to top-*k* where `topk(π) ≠ topk(π′)`.** Ill-posed as stated,
and this is precisely the interesting case. Adopt the **Fagin–Kumar–Sivakumar
generalised Kendall distance** (*Comparing Top k Lists*, SIAM J. Discrete Math.
17(1), 2003). Over the union `U = topk(π) ∪ topk(π′)`, each unordered pair
`{a,b} ⊆ U` contributes:

| case | condition | contribution |
|---|---|---|
| 1 | both appear in both lists | 1 if oppositely ordered, else 0 |
| 2 | both in one list, exactly one in the other | 1 if the list containing both ranks the *absent-from-the-other* element first, else 0 |
| 3 | `a` in one list only, `b` in the other only | 1 |
| 4 | both in one list, neither in the other | `p` |

Use **`p = ½`** — but for the right reason. It is the **neutral** choice: knowing
nothing about the relative order of two elements absent from a list, they
disagree with probability one half, so ½ is the unbiased estimate of the
contribution. `p = 0` assumes the unseen pair agrees, biasing every measurement
*downwards*, which is the wrong direction for a study of instability; `p = 1`
biases upwards.

> **Correction.** An earlier draft of this entry claimed `p = ½` makes `K⁽ᵖ⁾` a
> genuine metric. **That is false.** Measured against this repository's own
> implementation (itself cross-checked against an independently written
> reference), the triangle inequality is violated at *every* penalty tested, and
> the violation grows with `p`:
>
> | `p` | worst observed `d(A,C) − d(A,B) − d(B,C)` |
> |---|---|
> | 0.0 | +2 |
> | 0.5 | +4 |
> | 1.0 | +6 |
>
> Witness at `p = ½`, `k = 3`: `A = [3,1,0]`, `B = [5,3,4]`, `C = [5,4,2]` give
> `d(A,B) = 6`, `d(B,C) = 2`, `d(A,C) = 12`. `A` and `C` are disjoint, so their
> distance is the maximum, while `B` shares an element with each.
>
> `K⁽ᵖ⁾` is a **near-metric**: bounded distortion, not the triangle inequality.
> The observed distortion constant is smallest at `p = 0` (≈ 4/3) and grows with
> `p` (≈ 5/3 at ½). This is sufficient for reporting disagreement rates, which is
> all §7.3 needs, but it means the quantity must never be clustered on or treated
> as a norm — and `p` should be chosen on bias grounds, as above, not on a
> metric property it does not have.

Normalise by the maximum, attained on disjoint top-*k* lists:

```
max K^(1/2) = k²  (case 3)  +  2·C(k,2)·½  (case 4)  =  k(3k − 1)/2
```

so report `K̄ = K^(1/2) / (k(3k−1)/2) ∈ [0,1]`. Computed by direct `O(k²)` pair
enumeration — at `k ≤ 50` that is ≤ 1225 pairs, and obvious correctness matters
more here than asymptotics. Cross-checked in tests against an `O(k log k)`
merge-sort implementation.

**(c) Always reported alongside**, never alone:
`K_int` (plain Kendall on the *intersection*, normalised by `C(|S|,2)`, with
`|S|` always reported and `NaN` when `|S| < 2`); the top-*k* set disagreement
indicator `1[topk(π) ≠ topk(π′)]` that §7.3 explicitly asks for; and Jaccard
distance.

**Headline choice for §7.3:** the indicator for "top-*k* disagreement rate", and
normalised `K^(1/2)` for "within-top-*k* reordering".

---

<a id="g3"></a>
## G3 — Edge cases

Two modes throughout: `strict` (raise) and `lenient` (flag and return `NaN`).
The active mode is recorded in every run manifest.

| Case | Behaviour |
|---|---|
| `k = N` | `m_k` undefined → `NaN` plus a validity flag. Never coerced to 0 or ∞; excluded from margin distributions but counted in `n_undefined`. |
| `k > N` | Clamp to `N`, record `k_effective`, all boundary quantities `NaN`. `strict` raises `KOutOfRangeError`. Never a silent clamp. |
| `score(r_k) == score(r_{k+1})` | `m_k = 0`, flip radius 0. **The interesting case** — membership is decided purely by tie-breaking. Counted, never dropped; `P(m_k = 0)` is a headline statistic. |
| Zero-norm document | `w = 0`, `‖w‖ = 0`, `cos := 0` per §2.3. Remains rankable at score 0. Option `exclude_zero_norm_from_ranking` defaults **False** (paper-faithful); `n_zero_norm_docs` always reported. |
| Zero query vector | All `sᵢ = 0`; ranking degenerates to the pure attribute order; `m_k = 0` ∀k. Flagged `query_degenerate`; excluded from margin distributions, **included** in tie-break ablations (it is the extreme case). |
| Empty vocabulary | `EmptyVocabularyError` in `strict` — a configuration error, not data. |
| All scores equal | `m_k = 0` ∀k; every ball is the whole corpus; `ρ(τ)` fires. |
| `τ >` score range | Every ball is the whole corpus. Emits `TauExceedsScoreRange` carrying `τ/(s_max − s_min)`. **Not** an error — it is a legitimate point in the sweep — but tagged so plots can mark it. |
| `τ = 0` | Legal and required: the exact-tie baseline of the sweep. `≤ 0` ⟺ equality. |
| `N = 0` | Error on ranking. `N = 1`: all `m_k` are `NaN`. |
| Duplicate identifiers | `DuplicateIdentifierError` at load. The strict-total-order guarantee **depends** on unique ids. |
| `NaN`/`Inf` in scores or attributes | Rejected at load and re-checked before sorting. A `NaN` destroys the strict weak ordering, making `std::sort` undefined behaviour — a real out-of-bounds write, not merely a wrong answer. |

---

<a id="g4"></a>
## G4 — An explicit Lipschitz constant for §4.3

**Paper.** §4.3 states `|cos(u′,v′) − cos(u,v)| ≤ C (‖u′−u‖₂ + ‖v′−v‖₂)` "for a
constant `C` depending on lower and upper bounds on the norms", and leaves `C`
unspecified. An unspecified constant cannot be computed, tested, or used.

**Resolution — make it explicit and provable.**

> **Theorem.** Let `u, v, u′, v′ ∈ ℝⁿ \ {0}` and `L := min(‖u‖, ‖v‖, ‖u′‖, ‖v′‖) > 0`.
> Then `|cos(u′,v′) − cos(u,v)| ≤ (1/L)(‖u′−u‖₂ + ‖v′−v‖₂)`, i.e. **`C = 1/L`**.

*Proof.* Write `û = u/‖u‖`.
1. `|⟨û,v̂⟩ − ⟨û′,v̂′⟩| ≤ |⟨û−û′, v̂⟩| + |⟨û′, v̂−v̂′⟩| ≤ ‖û−û′‖ + ‖v̂−v̂′‖`
   by the triangle inequality and Cauchy–Schwarz with unit vectors.
2. In an inner-product space, `‖u/‖u‖ − u′/‖u′‖‖ ≤ 2‖u−u′‖/(‖u‖+‖u′‖)`
   (Dunkl–Williams, with the sharp Hilbert-space constant 2).
3. `‖u‖ + ‖u′‖ ≥ 2L`, hence `‖û−û′‖ ≤ ‖u−u′‖/L`; identically for `v`. ∎

The tighter non-uniform form is also implemented and tested:

```
|Δcos| ≤ 2‖u−u′‖/(‖u‖+‖u′‖) + 2‖v−v′‖/(‖v‖+‖v′‖)
```

**A corpus-level bound requiring no reference to the perturbation.** For TF-IDF
vectors as defined in §2.2:

- `idf(t) ≥ 1` for every `t ∈ V`, since `df(t) ≤ N ⇒ (1+N)/(1+df) ≥ 1 ⇒ log ≥ 0`;
- `‖tfᵢ‖₂ ≥ ‖tfᵢ‖₁/√nnzᵢ = 1/√nnzᵢ` by Cauchy–Schwarz, because `‖tfᵢ‖₁ = 1` exactly;
- hence `‖wᵢ‖₂ ≥ 1/√nnzᵢ`, and `‖wᵢ‖₂ ≤ ‖idf‖_∞ = log((1+N)/2) + 1`.

Therefore **`C ≤ √( max nnz )`** over the documents and queries involved. This
also makes §6's qualitative claim — "cosine similarity becomes unstable for
low-norm vectors" — quantitative, and localises it to *short documents*.

**Verification.** A Hypothesis property test asserts the bound is never violated
over randomised non-negative sparse inputs, evaluated under exact summation, plus
a tightness search reporting the empirical `lhs/rhs` ratio.

---

<a id="g5"></a>
## G5 — The three-term bound assumes a fixed vocabulary

**Paper.** §4.2 bounds `‖wᵢ′−wᵢ‖₂` by
`‖Δtfᵢ‖₂‖idf‖_∞ + ‖tfᵢ‖₂‖Δidf‖_∞ + ‖Δtfᵢ‖₂‖Δidf‖_∞`.

**Ambiguity.** This presupposes `w`, `w′`, `idf`, `idf′` share an index set. But
under a corpus perturbation the **vocabulary itself changes**: tokens appear, and
tokens fall below `min_df`. The paper never says how `Δidf` is defined when
`V ≠ V′`.

**Resolution.** Define everything on the **union vocabulary** `V ∪ V′`, with
coordinates zero outside the respective vocabulary. For `t ∈ V′\V` this gives
`Δidf(t) = idf′(t)`, which is large, so the bound remains **valid but loose**.

Report both: (i) the bound on `V ∪ V′`; and (ii) the restriction to `V ∩ V′`
together with the ℓ₂ mass carried by the symmetric-difference coordinates, so the
decomposition stays exact:

```
‖w′−w‖₂² = ‖(w′−w)|_{V∩V′}‖₂² + ‖w′|_{V′\V}‖₂² + ‖w|_{V\V′}‖₂²
```

**Verification.** A property test generates a corpus plus a perturbation drawn
from `{add_doc, remove_doc, edit_tokens, duplicate_doc}`, rebuilds both models,
aligns on the union vocabulary, and asserts the inequality under exact
arithmetic. It also records **which of the three terms dominates**, producing a
figure for a mechanism the paper currently only asserts qualitatively.

---

<a id="g6"></a>
## G6 — `max_features` truncation rule (`TFIDF-SPEC-01`)

**Paper.** §2.1 mentions "optionally, a maximum-feature constraint" and stops.

**Ambiguity.** Neither the ranking criterion nor its tie-breaking is given. A
naive "first `max_features` encountered" implementation is **not invariant to
document order**, which would destroy the determinism guarantee.

**Resolution.**

> After applying `min_df` (and `max_df` if configured), if the candidate
> vocabulary exceeds `max_features`, retain the `max_features` tokens greatest
> under the strict total order: **(1) descending `df(t)`; (2) descending
> collection frequency `cf(t)`; (3) ascending UTF-8 byte order of `t`.**
> Identifiers are then assigned by re-sorting the retained set in ascending byte
> order.

Criterion (1) is the statistic the paper already privileges; (3) makes the rule a
*total* order, so the retained set is unique and order-invariant. A
`max_features_policy` option additionally offers `"sklearn_compat"` purely so the
scikit-learn differential test can be run with a matching rule.

**Bonus experiment this unlocks.** A token sitting exactly at the truncation
boundary being kept or dropped is *itself* a decision discontinuity of the kind
§4.5 studies, but upstream of scoring. Sweeping `max_features ± 1` and measuring
top-*k* disagreement gives a **vocabulary-boundary sensitivity** result extending
the paper's thesis to vocabulary construction.

---

<a id="g7"></a>
## G7 — Deterministic lemmatisation

**Paper.** §2 requires lemmatisation as part of a "fixed, deterministic
preprocessing map".

**Ambiguity.** NLTK's WordNet lemmatiser needs a downloaded, versioned corpus and
a POS tagger; spaCy needs a statistical model whose output is not stable across
versions. Neither is acceptable in an artefact claiming reproducibility, and
neither ports to C++.

**Resolution — a four-tier `Lemmatiser` protocol.**

- **Tier 0 `none`** — identity.
- **Tier 1 `porter2`** *(default)* — the Snowball English stemmer: a complete
  published algorithmic specification, no data files, portable bit-identically to
  both languages, and — decisively — with an official ~29 000-word
  `voc.txt`/`output.txt` test-vector pair. Vendored with a recorded SHA-256, so
  the preprocessing step is itself machine-verified.
  *Honesty requirement:* Porter2 is a **stemmer**, not a lemmatiser. The paper
  should read "lemmatisation (implemented as Snowball English stemming; see G7)".
- **Tier 2 `lookup`** — a bundled, hash-verified `surface⇥lemma` table consulted
  before the rule-based fallback.
- **Tier 3 `external_cached`** — NLTK/spaCy permitted, but **only through a
  cache**: run once, emit a token stream plus a manifest recording tool version,
  model version, and input/output hashes. The external tool is never invoked
  inside a reproducible run.

**Also pinned here, all previously unspecified:** Unicode normalisation is
**NFKC**, applied first; case folding is `str.lower()` (not `casefold()`, which
needs ICU to reproduce in C++); the tokeniser pattern is a pinned regex hashed
into the manifest; the stopword list is frozen, versioned and hash-verified;
n-gram range defaults to `(1,2)`; the n-gram joiner is `\x1f` (ASCII Unit
Separator, which cannot occur inside a token, keeping the token→n-gram encoding
injective); and **stopword removal precedes n-gram generation, with n-grams
forbidden from spanning a removed token** — otherwise "king of pop" silently
manufactures the bigram "king pop".

---

<a id="g8"></a>
## G8 — Attribute directions and missing values

**Paper.** §2.3.1 gives the tuple `(popularity, rating, engagement, identifier)`
but never a sort direction.

**Resolution.** Pin per attribute in config: `{name, direction, dtype,
missing_policy}`. Defaults: `desc` for popularity/rating/engagement, `asc` for
identifier. Missing values are represented by an explicit `has_value` bit and
sort last — **never** `NaN`, which would break the ordering.

**Float attributes are themselves a determinism hazard.** A mean rating computed
as `sum/count` in binary64 can make two genuinely equal means compare unequal in
a platform-dependent way. Since MovieLens ratings are quantised to 0.5, store the
**exact integer pair `(2·Σrating, count)`** and compare `a.s * b.c` against
`b.s * a.c` in `int64`. This removes floating point from the tie-break entirely.

---

<a id="g9"></a>
## G9 — Precision of `score(r_j)`

The raw computed `double`, never rounded or quantised. The comparison is
`|sᵢ − s_{r_j}| <= τ`, inclusive, exactly as written in §2.3.3.

---

<a id="g10"></a>
## G10 — Leave-one-out protocol

§7.1 names the protocol but leaves five decisions unstated, each of which
materially changes the margin distribution:

1. **Which item is held out** — *every* interacted item in turn (all folds). Any
   subsampling uses a seeded RNG whose seed is in the manifest.
2. **Does the held-out item stay in the corpus?** — **Yes.** It must remain
   scoreable; it is the target.
3. **Are the user's remaining profile items excluded from the candidate set?** —
   **Yes.** Otherwise they trivially occupy the top ranks, since they literally
   contributed the query text, and dominate the margin distribution. *This is the
   single most consequential unstated choice in §7.1.*
4. **Eligibility** — users with ≥ 5 qualifying interactions; resulting counts
   recorded (see G14).
5. **What counts as an interaction** — for MovieLens, `rating ≥ 4.0`, pinned in
   `configs/datasets.yaml`.

---

<a id="g11"></a>
## G11 — Profile aggregation

§7.1 says "aggregating **text** from a user's interacted items". Pinned as
`profile_aggregation = "text_concat"`: concatenate token streams, then apply the
standard embedding.

Note the consequence, which is worth stating in the paper: concatenation makes
`tf` a *length-weighted* average, so verbose items dominate the profile. Because
that is itself an interesting stability axis, `"vector_mean"` and `"vector_sum"`
are also implemented and offered as an ablation.

---

<a id="g12"></a>
## G12 — Query IDF mapping

Already specified by §3 ("embedded using the same vocabulary and IDF mapping as
the corpus"). No recomputation and no vocabulary extension for queries. Listed
here only because it is asserted explicitly in code rather than left implicit.

---

<a id="g13"></a>
## G13 — `log` is not correctly rounded

**This is the most consequential finding in the implementation.**

**Measured on this project's reference machine**, comparing the platform
`math.log` against the correctly-rounded value (`decimal.Decimal.ln()` at 60
digits) for `N ∈ {100, 610, 9742, 20000, 50000}` across all valid `df`:

| quantity | share of entries that differ |
|---|---|
| the raw logarithm `log((1+N)/(1+df))` | **44.5%** |
| the idf value `log((1+N)/(1+df)) + 1` | **15.16%** (12 195 of 80 452) |

Worst absolute difference `1.78e-15`. The two figures differ because adding 1
shifts the value into a coarser binade, rounding away most 1-ulp disagreements.
**The 15.16% figure is the load-bearing one**, since idf is what propagates
downstream; the 44.5% figure is the sharper statement about `log` itself. Both
are quoted so neither is mistaken for the other.

**Why it matters.** IEEE-754 does *not* require `log` to be correctly rounded,
and UCRT, glibc, Apple libm and the various libstdc++ combinations each round
differently. CPython's `math.log` delegates to the platform libm. So `idf` values
differ by ~1 ulp across platforms in ~15% of entries, and that propagates into
every weight, every norm and every score. **Cross-platform bit-reproducibility is
broken by default — by the only transcendental function in the entire pipeline.**

**Resolution — two decisions, both cheap.**

1. **Compute `idf` once in Python and pass it into the native core as data.**
   It is `O(|V|)` values computed once, so there is no performance reason for C++
   to call `log` at all. Everything remaining in the native pipeline is
   `+ − × ÷ √`, all of which IEEE-754 *does* mandate be correctly rounded.
2. **Compute it with a correctly-rounded logarithm**:
   `Decimal(1+N) / Decimal(1+df)` then `.ln()` at 60 digits, rounded once to
   `float`. Standard library only, microseconds of cost, identical on every
   platform. Exposed as `idf_log_impl ∈ {"correctly_rounded" (default),
   "platform"}` and recorded in the manifest, with a test asserting the two
   differ in the expected ~15% of entries so the distinction stays visible.

Together these make the whole system **bit-reproducible across Linux, macOS and
Windows**, which turns the cross-platform snapshot test into a hard CI gate
rather than an aspiration.

**Also pinned:** the expression is `log((1+N)/(1+df))`, division **first**.
Measured: `log(a/b) ≠ log(a) − log(b)` in **94.53%** of cases at `N = 9742`.

---

<a id="g14"></a>
## G14 — Query and user counts

§7.1 defers these to "the dataset configuration". Every run manifest therefore
fixes: `n_users_eligible`, `n_queries`, `n_folds`, `n_docs`, `|V|`, `nnz`, all
filter thresholds, and the dataset SHA-256.

---

## Related

- [`experiments.md`](experiments.md) — the reduction policies, the noise-floor
  measurement, and the derivation of the `τ` band from it.
- [`index.md`](index.md) — the determinism guarantees and how CI enforces them.
- [`mathematical_formulation.md`](mathematical_formulation.md) — the
  floating-point guard and where floating point was removed rather than handled.

---

<a id="g15"></a>
## G15 — Which reordering does `π_alt` use?

**Paper.** §4.5 defines the alternate tie-break ranking as
`π_alt = Sort(sᵢ, aᵢ with reordered priority)` and says nothing further.

**Ambiguity.** With three attributes there are 3! = 6 orderings, and §7.3's
top-*k* disagreement rate is a different number for each. The choice is not
cosmetic: it directly determines a published result.

**Resolution.** Pin `π_alt` to the **reversal** of `π`'s priority:

> `π` = (popularity, rating, engagement) → `π_alt` = (engagement, rating, popularity)

The reversal is the canonical choice — the antipode, maximally distant from `π`
in the space of priority orderings — so it is the one that best exposes the
sensitivity §4.5 is looking for. The full six-way sweep is available as an
ablation in `configs/ablations.yaml` for anyone who wants the whole picture.

The identifier is **not** part of the permutable priority. It terminates every
key implicitly, and moving it would stop the ordering being total, which would
invalidate the uniqueness of the sorted permutation and with it every
reproducibility claim in the ranking layer. `ranking/sort_keys.py` rejects a
priority that names it.

---

<a id="g16"></a>
## G16 — `m_min^top` at *k* = 1

**Paper.** §2.3.2 defines `m_min^top = min over 1 ≤ j < k of (score(r_j) − score(r_{j+1}))`.

**Ambiguity.** At `k = 1` the minimum is over an empty set.

**Resolution.** `NaN` plus `defined = False`, consistent with G3's treatment of
every other undefined margin. `+inf` — the conventional value for an empty
minimum — would be actively harmful here: it would read as "no constraint on
stability", which is the opposite of "the quantity does not apply", and it would
propagate into percentile summaries as a finite-looking extreme.

---

<a id="g17"></a>
## G17 — Ranking an empty corpus

**Paper.** G3 says `N = 0` is "an error on ranking" but names no exception.

**Resolution.** `EmptyCorpusError(TfidfStabilityError)`, raised by `rank()` and
`rank_top_k()` before any other validation. Distinct from
`EmptyVocabularyError`, which is a *configuration* fault (`min_df` too high);
an empty corpus is a *data* fault.

---

<a id="g18"></a>
## G18 — Where low-norm cosine instability actually begins

**Paper.** §6 states that "cosine similarity becomes unstable for low-norm
vectors" and leaves it qualitative.

**Measured.** The instability has a sharp, predictable onset, and it is not
where the vector's magnitude becomes small — it is where the *square* of a
coordinate does. `l2_norm` computes `sqrt(sum of squares)`, so a coordinate
below `sqrt(DBL_MIN)` ≈ 1.49e-154 squares into the subnormal range:

| coordinate | `cos(u, unit)` | state |
|---|---|---|
| `1e-150` | `1.0` | exact |
| `1e-154` | `1.0` | exact |
| `1e-155` | `1.0000000000000016` | norm degraded |
| `1e-170` | `0.0` | square flushed to zero; a non-zero vector reports zero norm |

The onset agrees with `sqrt(DBL_MIN)` to the digit.

**Resolution: none — this is correct behaviour.** A hypot-style rescaled norm
would avoid it entirely, and §6 explicitly forbids exactly that class of
stabilising transformation. So the implementation is right, and what needed
fixing was a *test* that asserted scale invariance without excluding the regime
in which the specification's own stated limitation applies.

Worth stating in §6 as the concrete form of its own claim, and worth noting that
real TF-IDF norms are bounded below by `1/sqrt(nnz)` (see [G4](#g4)), which is
far above the threshold — so this bites only on synthetic inputs, never on the
corpora this project studies.

### Refinement: scaling moves a vector *across* the threshold

The test fix described above was applied to only half of what needed it, and a
100k-example search later found the other half.

A scale-invariance test compares `cos(k·u, v)` against `cos(u, v)`. The guard
excluded the underflow regime for the **scaled** vector alone — but multiplying
by `k > 1` moves a vector *out* of the regime, so the original `u` can sit below
the threshold while `k·u` sits above it. Both cosines appear in the assertion, so
guarding one endpoint is not enough. The falsifying example:

    u = 8.389842684852489e-160     (below 1.49e-154)
    k = 177795.0
    k·u = 1.49e-154                (above it)

    cos(k·u, v) = 1.0
    cos(u,   v) = 0.9999994865255848

They differ by 5e-7 against a 1e-12 tolerance, and the implementation is right in
both cases — it is exhibiting exactly the instability tabulated above. The
correct guard excludes the regime for **every vector the assertion touches**, not
merely the one the test happens to construct.

**Why it stayed hidden.** The three `assume()` calls reject about 41% of
generated examples (measured: 697 invalid of 1697 at `max_examples=1000`), which
sits close enough to Hypothesis's `filter_too_much` threshold that the health
check fired on the CI runner and not locally. The health check was therefore
masking a real defect: suppressing it — justified, since the filtering is
deliberate — is what exposed the counterexample. A health check that fires
intermittently is worth treating as a symptom rather than noise.

---

## Errata recorded during implementation

**To G3 — "τ > score range" is off by one.** At `τ` *equal* to the range every
tie ball is already the whole corpus, so the degeneracy has begun. Adopt
`τ >= s_max − s_min`, excluding the single case `range == 0 and τ == 0`, which is
the legitimate exact-tie baseline rather than a degenerate configuration. The
carried ratio `τ / (s_max − s_min)` is `0/0` when the range is zero; return
`+inf` explicitly so a NaN never reaches a plot axis.

**To G3 — "all scores equal ⇒ ρ(τ) fires" is incorrect.** With every score
equal, chain and clique coincide, so `ρ = 1` — its *minimum*, not an extreme.
What fires there is `TauExceedsScoreRangeWarning`, because the range is zero.

**To G8 — the justification, not the resolution.** G8 argues that a mean stored
as `sum/count` in binary64 "can make two genuinely equal means compare unequal
in a platform-dependent way". That does not survive contact with the data: with
0.5-quantised ratings and counts below 2^53 the sum is computed *exactly*, and
IEEE-754 mandates correctly-rounded division, so equal means produce
bit-identical doubles on every conforming platform.

The hazard that *is* real is the opposite one — two genuinely **different** means
colliding onto the same double. `1/3` and `(10^17+1)/(3·10^17)` differ as reals
and round to the same binary64; cross-multiplication separates them, and the
products involved stay comfortably inside `int64`. That is information the
tie-break is entitled to and the float path destroys. The resolution stands; the
reason needs restating.

*Implementation note.* The overflow guard must bound `num_i * den_j` over
**distinct** documents. Bounding by `max(num) * max(den)` across the column is a
false positive whenever the largest numerator and the largest denominator belong
to the same document — the normal case, since a large numerator usually
accompanies a large denominator — because that product is never formed by any
comparison.

**Note under G9 — the only faithful implementation.** The obvious way to find a
tie ball is to binary-search for `S[j] ± τ`. Those bounds are themselves rounded,
so the predicate actually evaluated is `S[i] <= fl(S[j] + τ)`, which differs from
G9's pinned `|sᵢ − s_{r_j}| <= τ` precisely at the boundary — the only place tie
groups are interesting. Search on the difference instead: `S[i] − S[j]` is
non-increasing in `i` and `S[j] − S[i]` is non-decreasing, so both bounds remain
binary-searchable while evaluating exactly the subtraction G9 specifies. The
monotonicity holds in binary64, not merely in the reals, because IEEE subtraction
is monotone.

**Note under §2.3.2 — margins are tie-break independent.** `m_k` depends only on
the score *multiset*: the non-increasing rearrangement of a multiset is unique,
and all three operators use score-descending as their primary key. So `m_k` is
identical under `π`, `π_score` and `π_alt`. The paper never states this, and it
is what makes research questions A1 and A2 *independent* rather than confounded
— which is the whole reason they can be answered separately.

**Note under §4.4 — the two conditions constrain disjoint sets of gaps.** §4.4
gives two guarantees and it is easy to read one as stronger than the other. They
are not comparable. `m_min^top` minimises over the gaps *strictly inside* the
top-*k* (ranks 1→2 through (k−1)→k); `m_k` is the gap *at the boundary*
(k→k+1). Those index sets are disjoint, so neither radius bounds the other:

| ranking | tighter radius |
|---|---|
| tight cluster at the top, wide boundary | order (`m_min^top/2`) |
| well-spread top, near-tied boundary | set (`m_k/2`) |

Guaranteeing the top-*k* set **and** its ordering therefore requires
`ε < min(m_k, m_min^top)/2`, which is what
`perturbation/score_bounds.py::StabilityCertificate.joint_radius` reports. A
certificate quoted without saying which invariant it certifies is ambiguous.

**Note under §4.4 — the bound is tight, not merely sufficient.** §4.4 gives
`ε < m_k/2` as a sufficient condition for top-*k* invariance and does not address
necessity. It is in fact exact: perturbing `r_k` by `−ε` and `r_{k+1}` by `+ε`
with `ε = m_k/2 + δ` flips the pair for any `δ > 0`, and at `ε = m_k/2` exactly
the two scores become bit-identical and membership passes entirely to the
tie-break. `tests/test_margins_and_flip_radii.py` constructs the witness in
dyadic rationals so that every step is exact in binary64.

---

<a id="g19"></a>
## G19 — The candidate set varies per leave-one-out fold

**Consequence of [G10](#g10)(3), which the paper does not draw out.**

Excluding a user's remaining profile items from the candidate set — necessary, or
those items retrieve themselves — means the number of rankable documents is
`N − |profile| + 1`, not `N`. Since profile sizes differ between users, and by one
within a user across folds, **`N` differs from query to query**.

Three things follow, all of which affect §7.2 and §7.3 as written:

1. **Margin distributions are pooled over queries with different `N`.** A margin
   at `k = 50` drawn from a 9 000-candidate fold and one from a 9 700-candidate
   fold are not identically distributed, because the density of scores near the
   boundary depends on how many documents are competing for it.
2. **`k` can exceed the candidate count** for a heavy user on a small corpus,
   which is the `k > N` case of [G3](#g3) arriving through the *protocol* rather
   than through a configuration error. It is handled leniently, with
   `k_effective` recorded.
3. **A disagreement *rate* has a varying denominator.** §7.3's "fraction of
   queries for which the top-k set differs" is well defined, but the per-query
   populations it averages over are not the same size.

**Resolution.** `Query.n_candidates` is carried on every query and recorded in
the run manifest, so the variation is visible rather than assumed away. Reported
statistics quote the query count alongside the rate (already required by
[G14](#g14)), and the manifest additionally records the minimum, median and
maximum candidate count so a reader can see the spread.

Proposed for §7.1: state that the candidate set excludes the profile and that its
size therefore varies, and report the spread with the query count.

---

<a id="g20"></a>
## G20 — Profile aggregation is order-sensitive

**Paper.** §7.1 builds a profile "by aggregating text from a user's interacted
items" and says nothing about the order of aggregation.

**Ambiguity.** Concatenation is not commutative in its effect. The concatenated
token stream differs under a different item order, and — because n-grams are
generated over the *concatenated* stream — so does the set of features produced
at the seams between items. Different orders therefore give different
vocabularies, different `df`, and different scores.

Interaction files are not order-stable: a ratings CSV may be grouped by user, by
timestamp, or by neither, and re-exporting it can reorder rows without changing
its content.

**Resolution.** Canonicalise. `group_interactions` returns each user's items in
**ascending identifier byte order**, matching the rule
[`vocabulary.py`](../src/tfidf_stability/vectorisation/vocabulary.py) uses for
tokens, so a profile is a pure function of the *set* of interacted items rather
than of the order they happened to be read in.

Note the interaction with [G7](#g7)'s gap sentinel: inserting a boundary marker
between concatenated items would make aggregation order-*insensitive* by
preventing n-grams from spanning the seam. That is arguably the better
construction, but it is not what §7.1 describes, so it is left as an available
ablation rather than adopted silently.

---

<a id="g21"></a>
## G21 — `vector_sum` and `vector_mean` are indistinguishable by similarity

Measured while implementing [G11](#g11)'s ablations. Summing a user's item
vectors and averaging them differ by the positive scalar `1/n`, and cosine
similarity is invariant under positive per-vector scaling — so the two
aggregations produce **identical similarity scores and identical rankings**.

This is the same invariance that makes the scikit-learn cross-check possible
(see [G8](#g8)'s neighbourhood and the note in
[`tf.py`](../src/tfidf_stability/vectorisation/tf.py)), and it is worth stating
explicitly for the same reason: it is easy to spend effort choosing between two
options that cannot be distinguished by the metric being reported.

They are *not* interchangeable everywhere, however. The **norms** differ by `n`,
and §§4.2–4.3 state their perturbation bounds in terms of norms — so the two
aggregations give the same scores but different stability *certificates*. Both
are therefore retained, and the choice is recorded in the manifest.

---

## G22 — A fine near-tie cannot be manufactured; §7.4 must *identify* one

**Where:** §7.4 ("Case Study: Near-Tie Sensitivity"), §7.3 (stratification by margin).

§7.4 says: *"Two documents A and B are **identified** such that |s_A − s_B| ≤ τ."*
The word *identified* turns out to be load-bearing, and this addendum records why —
because the natural reading, that one **constructs** such a pair by editing document
text, is not achievable.

### Why editing text cannot produce a fine near-tie

§2.2 defines `tf(t, d) = count(t, d) / L_d`. Adding or removing a single token
therefore changes **every** term frequency in that document from `c/L` to `c/(L+1)`
— a *relative* perturbation of `1/(L+1)` applied uniformly, not a small additive
nudge to one coordinate. The induced score separation scales the same way, so:

| target separation | document length required |
| --- | --- |
| 1e−2 | ~100 tokens |
| 1e−4 | ~10,000 tokens |
| 1e−9 | ~1,000,000,000 tokens |

The twin-pair mechanism in `datasets/synthetic.py` (a duplicate plus one extra
token, with the extra token's `df` chosen to tune the gap) works, but spans only
about 1e−3 to 1e−1 — 36× across the whole `df` grid. This is a structural
consequence of the normalisation in §2.2, not a limitation of the generator, and no
choice of extra token can escape it. **The single-token edit is the finest
text-level perturbation that exists**, so `1/L` is a floor.

This is a real distinction from §5's perturbation model, which injects noise
directly into scores or IDF weights and *can* reach any magnitude. Text-level and
score-level perturbations are not interchangeable at fine scales.

### What occurs naturally instead

Measured on a synthetic corpus of 3000 documents (`vocab_size=5000`,
`n_exact_duplicates=30`, `n_twin_pairs=60`, seed as specified, queried with the
first six tokens of document 0), over all 2999 adjacent pairs of the sorted
score vector:

| adjacent gap `m_k` | count | share |
| --- | --- | --- |
| exactly 0 | 553 | 18.4% |
| in (0, 1e−9) | **0** | **0.0%** |
| in [1e−9, 1e−6) | 104 | 3.5% |
| in [1e−6, 1e−3) | 2323 | 77.5% |
| ≥ 1e−3 | 19 | 0.6% |

Smallest strictly-positive gap: **2.0e−08**. 2524 of 3000 documents score above
zero, so the exact-tie mass is not merely the zero block.

**The shares are configuration-dependent, and so — it turns out — is the empty
interval.** Repeating this with a different corpus size, vocabulary or query
moves every row of that table by a few percent.

An earlier version of this addendum went further and claimed the *middle* row was
invariant: that no run had ever placed an adjacent pair in (0, 1e−9), and that
the claim was safe enough to assert as a regression test. **That was an
over-generalisation from synthetic data, and MovieLens falsifies it.**

### What MovieLens actually shows

Measured on `ml-latest-small` (9742 films, digest `696d65a3…`), 12 leave-one-out
folds, 114,504 adjacent pairs, under the **normative naive reduction**:

| adjacent gap | count | share |
| --- | --- | --- |
| exactly 0 | 3129 | 2.73% |
| **in (0, τ_floor = 4.44e−16)** | **197** | **0.172%** |
| smallest strictly-positive gap | **8.67e−19** | — |

The interval is not empty. It contains 197 pairs, and the smallest positive gap
is nine orders of magnitude below the synthetic corpus's, and three orders
*below the arithmetic noise floor itself*.

### Why, and why it matters more than the number

Those gaps are not separations. Recomputed under `Reduction.EXACT` the same
folds give a smallest positive gap of **1.4e−11** — the sub-femto gaps are
manufactured by naive summation, not present in the data. So the normative
backend reports pairs of films as *distinctly scored* when the separation is
smaller than its own error.

That is precisely the situation τ exists to detect, and it means the two
addenda interact on real data in a way they did not on synthetic:

* G23's band is computed from `g_min` under **exact** arithmetic, and on
  MovieLens that still gives a valid 4.5-decade band.
* But the gaps a *consumer* sees come from the **naive** backend, and 0.172% of
  those fall below `tau_floor`. Judged on those, `g_min < tau_floor` and the
  band is **empty** — G23's own "this is a finding, not a bug" case.

Which convention is correct is a question for the paper, not for the code, and
it is not answered here. What is settled is that §7.4's regime cannot be
described as "the interval is empty, so the near-tie regime is the exact-tie
regime" on real data. On MovieLens the exact-tie share is 2.7%, not 17–18%, and
there is a genuine population of sub-noise separations besides.

The regression test
`tests/test_datasets.py::test_the_near_tie_interval_below_tau_is_empty` remains
valid — it is scoped to the synthetic generator, where the property does hold —
but it must not be read as establishing anything about real corpora.

Two consequences the paper should state:

1. **The near-tie regime is, empirically, the exact-tie regime.** At τ ≈ 1e−9 the
   interval (0, τ) is *empty*, so essentially every within-τ pair has a gap of
   exactly zero. §7.3's "m_k ≤ τ" stratum is therefore not measuring
   near-ties-under-noise; it is measuring the exact-tie block, where the outcome
   is decided entirely by the tie-break and the numerical error is irrelevant.
   That is A2's regime, not A1's, and conflating the two would misattribute the
   cause.
2. **The exact-tie mass is structural, not an artefact.** It comes from the
   zero-score block (queries share few terms with most documents) and from
   genuine duplicates — both of which occur in real catalogues.

### Resolution

`datasets.synthetic.find_near_ties` searches a sorted score vector for the closest
adjacent pairs, with `strictly_positive` controlling whether the exact-tie block is
included. §7.4's case study uses it to *identify* a pair at the finest magnitude the
corpus actually contains, and reports that magnitude rather than assuming τ.

Where a specific τ *is* required, it must come from §5's score-level perturbation
model — where any magnitude is reachable — and not from a text edit.

---

## G23 — τ is derived as a *band*, not chosen as a value

**Where:** §7.1 ("τ is chosen to exceed floating-point noise while remaining
small relative to typical score separations"), and every §7.3 result that is
"explicitly conditional on this choice of τ".

That sentence is the paper's complete guidance on τ. It is a two-sided
**qualitative** constraint: it names a lower reference (arithmetic noise) and an
upper one (typical separations), gives no value, no procedure, and no guarantee
that the two leave a non-empty gap. Deriving a single number from it would
manufacture precision the specification does not contain. This addendum records
what is done instead.

### The lower endpoint is exactly twice the arithmetic error

τ's operational job is to decide whether `|s_i − s_j| ≤ τ` is evidence of a real
difference or an artefact. The quantity to bound is therefore the error in a
**margin**, not in a score. If each score carries error at most `η`, then
`s_i − s_j` carries at most `2η`, and both signs are attainable, so

    τ_floor = 2η

The factor 2 is exact, and it is the same 2 as in §4.4's `ε_k^flip = m_k / 2`.

`η` is *measured*, not bounded a priori: the corpus is scored under
`Reduction.{NAIVE, NEUMAIER, PAIRWISE}` against `Reduction.EXACT`
(Shewchuk/`math.fsum`) as correctly-rounded ground truth.

**A trap worth naming.** `TfidfModel.norms` is precomputed under the model's own
reduction, so varying only the policy passed to the scorer holds the norms fixed
and measures the dot product alone. That understates `η` by about **threefold**:
a query dot product runs over a handful of shared terms (1–5 in the measured
corpus), whereas a norm sums the whole document vector (39 on average, up to 75),
and the longer summation is where error accumulates. Measured both ways on the
same 1500-document corpus × 25 queries:

| what varies | share of scores differing | η |
| --- | --- | --- |
| dot product only (**wrong**) | 9.25% | 5.551e−17 |
| dot product and norms (**correct**) | 42.54% | 1.665e−16 |

`Reduction.NEUMAIER` was **exactly correctly-rounded** on every one of the 37,500
comparisons, and `PAIRWISE` was bit-identical to `NAIVE` — the pairwise block is
128 and no summation here is that long, so a reduction-policy sweep over short
queries measures nothing. Worth stating, because it makes such a sweep look
informative when it is vacuous.

### The upper endpoint is the score lattice

The smallest strictly-positive adjacent gap actually observed, `g_min`. Below it
no pair of *distinctly* scored documents is within τ of each other.

### The band can be *proved* invariant, not merely sampled

Every τ-dependent object in the implementation is **piecewise constant in τ**,
with breakpoints only at observed gap values: `tie_chains` cuts where an adjacent
gap exceeds τ; `tie_cliques` admits an interval whose diameter is ≤ τ, and every
diameter is a sum of gaps; `tie_ball(j, τ)` is delimited by `|s_i − s_j| ≤ τ`,
again a gap sum.

So if the half-open band `[τ_floor, g_min)` contains **no** observed gap, every τ
in it yields *bit-identical* tie structure — by argument, not by a sweep.
`TauBand.is_invariant` reports exactly that condition, and
`verify_band_invariance` recomputes the structure at eight logarithmically spaced
probes as an independent check on the code.

Measured (1500 documents, 25 queries):

| quantity | value |
| --- | --- |
| η | 1.665e−16 |
| τ_floor = 2η | 3.331e−16 |
| g_min | 6.958e−10 |
| band width | **6.32 decades** |
| observed gaps inside the band | **0** |
| exact ties / positive gaps | 7781 / 29694 |

### This is not circular

The objection to deriving τ is that τ might be derived from a quantity τ then
determines. It is not: `τ_floor` comes from arithmetic and `g_min` from the score
lattice, and neither is a function of τ.

### What may and may not be claimed

- **May**: that the reported tie structure is invariant across 6+ decades of τ,
  and that the specific value is therefore immaterial.
- **May not**: that this shows the *mechanism* is robust. The band is empty of
  margins (G22), so invariance across it reflects **emptiness of the score
  lattice**, not insensitivity of the tie-break. The plateau statement must
  always be paired with its cause.
- `TauBand.display_tau()` (the geometric midpoint) is a **presentation choice**
  so a caption can name a number. It is not a derived constant and must never be
  cited as one.

### If the band is ever empty

`is_valid` is false when `τ_floor ≥ g_min` — arithmetic noise reaching the
decision boundary. That is a **finding**, not an error: it would mean no τ
separates numerical error from tie structure on that corpus, that every §7.3
result there is contaminated by A1's regime, and that the A1/A2 separation
collapses. The code reports it and refuses to invent a value.

### The scale separation actually observed

```
numerical noise floor      ~1e-16
                              |  ~6-7 orders
smallest observed gap      ~7e-10
                              |  ~5-6 orders
typical adjacent gap       ~1e-4 .. 1e-3
corpus-edit perturbation   1e-4 .. 1e-1
```

Twelve to thirteen orders of magnitude separate the numerical-error regime from
the corpus-perturbation regime. **Numerical error never came within six orders of
magnitude of closing a real score gap.** A1's "bounded perturbations" are
therefore entirely about semantic/corpus perturbation; floating-point error is
not a threat to ranking at any scale this system operates at. That is a positive
result for §6's design, and it is what makes A1 and A2 cleanly separable
questions rather than two names for one effect.

---

## G24 — `cos ∈ [0, 1]` is true in exact arithmetic and false in binary64

**Where:** §2.3 — *"Since all coordinates are non-negative, it follows that
cos(u, v) ∈ [0, 1]."*

The inference is valid over the reals and does not survive rounding. Measured
over 40,000 random non-negative sparse vectors (2–40 non-zeros, values in
[1e−3, 10]):

| quantity | value |
| --- | --- |
| trials where the result exceeded 1.0 | **10,947 (27.37%)** |
| max `cos(v, v)` | 1.0000000000000002 |
| max `cos(v, s·v)` | 1.0000000000000007 |
| worst excess | 6.661e−16 (3 ulp) |

The cause is straightforward: `cos = dot / (‖u‖ ‖v‖)` performs three independent
roundings — the dot product, each norm, and the division — and nothing forces the
numerator and denominator to round in the same direction. Self-similarity is the
worst case precisely because the true value sits exactly on the boundary.

### The implementation does not clamp, and should not

`similarity/cosine.py` returns `dot(u, v, policy) / (nu * nv)` unmodified. That
is correct under §6, which forbids stabilising transformations: clamping to
`min(1.0, x)` would change published digits and would hide the very effect this
study measures. It also would not be free — the clamp would have to be applied
consistently in the C++ mirror or the two backends would stop being bit-identical.

### What must therefore not be assumed downstream

Any consumer treating the result as a cosine **in the mathematical sense** can
fail. In particular `math.acos(1.0000000000000002)` raises `ValueError`, so
converting a similarity to an angle without clamping at the call site is a
latent crash. Nothing in this repository calls `acos` — verified — so there is no
live defect, but the guarantee a reader would reasonably infer from §2.3 does not
hold and anything built on top of this code must clamp at its own boundary.

### Suggested wording for §2.3

> Since all coordinates are non-negative, it follows that `cos(u, v) ∈ [0, 1]` in
> exact arithmetic. In binary64 the computed value may exceed 1 by a few ulp,
> because the dot product, the two norms and the division round independently;
> this was observed in 27% of random trials, with a worst case of 3 ulp. No
> clamping is applied, in keeping with §6.

Related: [G18](#g18) records the other place the idealised cosine identities
break down — scale invariance, which fails below `|x| ≈ √DBL_MIN`.

---

## G25 — the query protocol is not interchangeable, and it changes every number

**Where:** §7.1 ("Query set and evaluation setup"), and every §7.2/7.3 result.

§7.1 specifies that experiments use **user-profile** queries and **leave-one-out**
folds, with item-as-query implemented but explicitly "not evaluated in the present
experiments". An earlier version of the experiment runners used **truncated
document prefixes** instead — a different protocol, and a much easier one.

This addendum records that the substitution is not benign.

### Why a document prefix is an easier query

A prefix of document *d* retrieves *d* itself at very high similarity, because it
is literally a subset of *d*'s own features. The top of the ranking is therefore
dominated by one strongly-separated document, which inflates `m_1` and suppresses
the exact-tie rate. A leave-one-out fold has to retrieve a *held-out* item from
the concatenated text of the user's *other* items, so the top of the ranking is a
contest among genuinely similar candidates.

### The measured difference

Same corpus (`synthetic_tiny`, 120 documents), same code, same `k`:

| quantity | document prefixes | §7.1 leave-one-out |
| --- | --- | --- |
| exact-tie share at k=1 | 23.3% | **50.0%** |
| median `m_1` | 8.769e−02 | **0.0** |
| π vs π_score disagreement at k=1 | 16.7% | **50.0%** |

Under the specified protocol the **median** boundary margin at rank 1 is exactly
zero. The shortcut protocol was materially *understating* the phenomenon the
study exists to measure.

### The A2 result the correct protocol produces

Stratified by margin band (§7.3), π vs π_score at k = 1:

| band | disagreement | n |
| --- | --- | --- |
| `exact_tie` | **100.0%** | 20 |
| `(100·τ, ∞)` | **0.0%** | 20 |

A complete separation: every query with an exact tie at rank 1 disagreed between
operators, and no query without one did. Combined with the bit-identity check
(all three operators provably consume the same scores), this is A2 in its
strongest form — the tie-break is not *a* factor in the disagreement, it is the
*only* factor.

### The candidate set moves, and margins must follow it

Each query excludes its own profile items (G10 decision 3), so **N differs per
query** (G19). Three consequences the implementation must honour, all in
`analysis/query_grid.py`:

1. Margins are computed over the **candidate** scores only. Computing them over
   the full corpus would include documents the query was never allowed to
   retrieve, inflating the apparent separation.
2. The attribute table is restricted to the same subset, or the tie-break would
   rank over non-candidates. `transition_curve` and `certificate_audit` therefore
   accept per-query tables.
3. `k` can exceed a query's candidate count. Those queries are excluded and
   counted, never clamped — a clamped `k` measures a different quantity.

### What this means for reproduction

Any result quoted from this study must state its query mode. Two runs that differ
only in protocol are not comparable, and the protocol is recorded in every run
manifest via `QueryGrid.provenance()`.

---

## G26 — the interaction term of §4.2 never dominates

**Where:** §4.2 — *"In sparse high-dimensional embeddings, the interaction
between local changes and globally shifting IDF weights is a natural mechanism
for perturbation amplification."*

§4.2 decomposes the vector shift into a **local** term (this document's own
content changed), a **global** term (the IDF weights moved beneath it) and an
**interaction** term, and singles out the third as a mechanism for
amplification. `vector_perturb.py` exposes `ThreeTermBound.dominant_term`
specifically so the claim can be checked rather than assumed.

Measured over 25 single-document edits on a 60-document corpus, 1500 per-document
shifts in total:

| dominant term | count |
| --- | --- |
| global | 1479 |
| local | 21 |
| **interaction** | **0** |

The interaction term is never the largest of the three. The **global** term
dominates almost everywhere, which is the sensible outcome given §4.1: one edit
moves `df` by at most one per term, but `idf = ln(N/df)` is steeply non-linear at
small `df`, so a rare term's weight moves far more than the edited document's own
content does.

### What may and may not be claimed

- **May**: that the three-term decomposition is a valid upper bound. It was never
  violated in any measurement here.
- **May not**: that the interaction term is the amplification mechanism. On this
  data it is not a mechanism at all. §4.2's sentence should be softened to say the
  interaction term is *bounded* by the product, and that the observed
  amplification is driven by the global IDF shift.

This does not weaken §4.2's inequality; it corrects the informal reading attached
to it. The bound holds — the attribution does not.

### Verified alongside it

The same adversarial pass confirmed, by execution rather than by reading:

| claim | result |
| --- | --- |
| §4.1 `idf ≥ 1` whenever `df ≤ N` | holds; `idf(df=N=9742) = 1.0` exactly |
| §4.1 one edit moves `df` by at most 1 | max observed `|Δdf|` = 1 |
| §4.2 three-term bound | 0 violations in 25 edits |
| §4.3 score shift within the reported maximum | 0 violations |
| §4.4 certified radius | **0 violations in 18,736 adversarial perturbations** |
| §4.4 tightness | 889 of 1668 flipped *exactly at* the radius |
| §2.3.1 output independent of input order | 0 failures over 400 tie-heavy corpora |
| §2.3.1 distinct scores ⇒ all operators agree | 0 disagreements in 400 |
| §2.3.2 `m_k` identical across operators | 0 differences; no negative `m_k` |
| §2.3.3 tie ball vs brute force | 4724 checked, 0 mismatches |
| §2.3.3 chains partition; cliques within diameter | 0 failures |
| §2.3.3 ball non-transitivity | 216 triples found in random data |

The §4.4 attack is the one that matters: it drove the rank-`k` and rank-`(k+1)`
scores together by the largest amount strictly inside the certified radius --
the worst case the proof must survive, which random sampling would essentially
never find -- and the bound held every time. The closest approach reached
`0.999999999` of the radius.

**A methodological trap, recorded because the first version of this attack fell
into it.** The obvious way to get "as close to the radius as possible" is
`math.nextafter(radius, 0)`. On dyadic scores that value rounds straight back up
when added, so the *realised* movement lands exactly **on** the boundary -- which
the theorem explicitly excludes -- and every trial is silently discarded. The
first run reported thousands of "certified perturbations" of which **none** were
actually inside the radius, and its zero-violation result was therefore vacuous.
The guard is to count only perturbations whose realised delta is verified
strictly less than the radius, and to assert that count is large;
`tests/test_margins_and_flip_radii.py` does both.
