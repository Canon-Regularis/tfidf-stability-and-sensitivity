# Numerical Stability and Perturbation Behaviour in TF-IDF-Based Similarity Systems

## Abstract

This repository holds a self-contained implementation and analysis of a
TF-IDF-based document similarity and ranking system, written to make the
algebraic structure and the perturbation behaviour of the pipeline explicit.

Documents are embedded by explicit tokenisation, n-gram construction, smoothed
inverse document frequency (IDF) and sparse TF-IDF vectors. Similarity is
cosine similarity; rankings are induced by a deterministic sorting procedure.
Intermediate quantities stay accessible: document frequencies, IDF weights,
vector norms, similarity scores. Corpus structure and similarity geometry can
therefore be related directly to ranking behaviour under perturbation.

Alongside the numerical sensitivity of TF-IDF embeddings and cosine similarity,
the repository treats ranking stability as a first-class object. Stability is
characterised empirically through score-separation margins: boundary margins
govern top-k membership, adjacent margins govern ordering within the top-k.
These margins yield explicit sufficient conditions under which a ranking is
invariant under bounded perturbations of the similarity scores.

Deterministic tie-breaking rules are isolated as a separate source of
decision-level discontinuity. Ranking outcomes may change while cosine
similarity scores remain equal to within numerical tolerance. Separating the
numerical stability of the scores from the stability of the induced ranking
exposes structural fragility in similarity-based retrieval and content-based
recommendation systems.

---

## 1. Introduction

This project examines the numerical stability of TF-IDF-based similarity and
ranking systems under small perturbations of the data and of the preprocessing.

TF-IDF (term frequency-inverse document frequency) is a foundational technique in
information retrieval and text-based modelling. It represents documents as
vectors in a high-dimensional feature space, where similarity is usually measured
by cosine similarity. Despite its ubiquity, the behaviour of TF-IDF pipelines
under perturbation (changes in corpus composition, token distributions or
preprocessing rules) is rarely analysed in a systematic and explicit manner.

The setting here is controlled: documents arise from text of interest, and
similarity scores induce content-based k-nearest-neighbour (k-NN) rankings. The
system is implemented twice. A normative pure-Python reference defines
correctness and requires no compiler; an optimised C++20 core is required to be
bit-identical to it, with agreement enforced by comparing raw bit patterns rather
than tolerances. Both give explicit control over preprocessing, vocabulary
construction, IDF computation, vector formation and ranking, so corpus structure
and similarity geometry can be related directly to similarity scores,
neighbourhood structure and ranking behaviour under perturbation.

The central question is one of stability:  
**how sensitive are TF-IDF weights, cosine similarities, and induced rankings to small 
changes in the underlying data?** 
Two aspects of it are addressed:
(i) how score-separation margins govern the stability of similarity-based rankings
under bounded perturbations, and
(ii) how deterministic tie-breaking rules introduce discontinuities in ranking
outcomes that are independent of numerical error in the similarity computation.

---

## 1.1 Purpose and Research Intent

The purpose of this repository is investigative rather than applicative. TF-IDF
is a *classical method*; the aim is to examine the mathematical structures the
pipeline induces, and how they behave under controlled variation of assumptions
and parameters, in the context of content-based similarity and k-nearest-neighbour
ranking.

The implementation exposes preprocessing operations, n-gram structure,
document-frequency thresholds, IDF scaling and sparse vector geometry rather than
hiding the pipeline behind a black box. That access supports detailed analysis of
sensitivity and stability in TF-IDF embeddings, cosine similarity and induced
k-NN neighbourhoods, and in particular the study of cases where small
perturbations of documents, corpus composition or user-derived profiles produce
disproportionate changes in similarity scores, neighbourhood structure or ranking
outcomes.

Ranking stability is a primary object of analysis in its own right. It is
operationalised through empirical score-separation margin distributions, which
quantify the tolerance of a ranking to bounded perturbations. Instability arising
solely from deterministic tie-breaking is isolated separately: ranking outcomes
may change even when similarity scores are equal to within numerical tolerance.

The empirical work concentrates on score-level and decision-level perturbations;
corpus- and embedding-level effects are analysed analytically, to characterise
their influence on downstream stability.

---

## 1.2 Investigative Scope

Retaining the intermediate quantities that higher-level libraries abstract away
permits systematic investigation of:

- the effect of corpus perturbations on **document frequency** and **smoothed IDF values**,
- the response of **TF-IDF embeddings** to token-level edits or preprocessing changes,
- the sensitivity of **cosine similarity** to perturbations in *sparse, non-negative vectors*,
- conditions under which **similarity-based rankings** and **k-NN neighbourhoods** remain
  invariant under bounded perturbations,
- the influence of **user-profile construction** on personalised similarity and
  neighbourhood structure,
- **margin distributions** governing the stability of top-k membership and ordering
  within top-k results,
- **tie-group analysis**, quantifying ranking changes induced by deterministic
  tie-breaking rules when similarity scores fall within numerical tolerance.

Such access joins formal derivation to empirical observation, and makes the
framework suitable for studying conditioning behaviour, perturbation effects and
ranking stability in TF-IDF-based similarity systems.

---

## 1.3 Position Within a Broader Mathematical Context

TF-IDF-based similarity sits at the intersection of information retrieval,
numerical linear algebra, finite-dimensional functional analysis and
probabilistic models of text. The construction gives a concrete setting in which
abstract numerical phenomena (sparsity, scaling behaviour, angular distortion,
perturbation amplification) can be observed and analysed directly.

From a decision-theoretic perspective, similarity-based ranking is a
piecewise-constant functional of the similarity scores, with discontinuities
induced by secondary ordering and tie-breaking rules. This perspective separates
the numerical stability of the underlying similarity computation from the
stability of the ranking it induces.

The same issues recur in high-dimensional feature representations used in machine
learning, in content-based similarity and recommendation systems, and in
retrieval pipelines, where numerical sensitivity and decision discontinuities
carry practical consequences. The work is accordingly an expository and
exploratory study of established mathematical constructions in a TF-IDF setting,
attending to their behaviour under perturbation and to their implications for
ranking stability and interpretability.

---

## 1.4 Intended Use

This repository is not a production-grade recommendation library. It serves as a
reference implementation, an exploratory mathematical environment and a
foundation for *small-scale research investigations* into TF-IDF representations,
cosine similarity, content-based k-NN ranking and ranking stability.

The emphasis throughout is on mathematical behaviour: conditioning, sensitivity
and perturbation effects in sparse vector spaces, together with the relationship
between numerical similarity computation and the ranking decisions downstream of
it. These bear directly on the reliability, reproducibility and interpretability
of similarity-based and content-based recommendation models.

---

## 2. Mathematical Formulation

Let  
𝒟 = { d₁, …, dₙ }  
be a finite corpus of preprocessed documents, and define N := |𝒟|. Each document dᵢ is a
finite sequence of tokens obtained from raw text via a fixed preprocessing map
(normalisation, tokenisation, stopword removal, lemmatisation, and n-gram generation). Throughout,
**n-grams are treated as atomic tokens**.

The normative lemmatiser is the Snowball English (Porter2) **stemmer**. It is called
lemmatisation throughout for continuity with the surrounding literature, but it performs
suffix stripping rather than dictionary lemmatisation, and the two differ on a substantial
fraction of tokens. See `docs/spec_addenda.md`.

All preprocessing operations are deterministic and fixed across all perturbation
experiments.

---

### 2.1 Vocabulary, Document Frequency, and IDF

From the corpus 𝒟, a vocabulary V is constructed by collecting the tokens
(n-grams included) that satisfy a minimum document-frequency threshold and,
optionally, a maximum-feature constraint. For each token t ∈ V, the document
frequency is

df(t) = |{ i : t appears at least once in dᵢ }|.

Let N = |𝒟|. The smoothed inverse document frequency used throughout the implementation is

idf(t) = log((1 + N) / (1 + df(t))) + 1.

Thus idf(t) decays monotonically as df(t) increases, and the additive constant
keeps it strictly positive in the limiting case df(t) = N.

(Here and throughout, log denotes the natural logarithm.)

---

### 2.2 TF-IDF Embedding

For each document dᵢ, let countᵢ(t) denote the number of occurrences of token t ∈ V in
dᵢ. The term frequency is

tfᵢ(t) = countᵢ(t) / ∑ₛ∈V countᵢ(s).

Term frequencies are therefore normalised with respect to **in-vocabulary tokens**,
out-of-vocabulary tokens being ignored once the vocabulary is fixed. A document
whose in-vocabulary token count is zero maps to the zero vector.

The TF-IDF weight of token t in document dᵢ is

wᵢ(t) = tfᵢ(t) · idf(t),

and the document is represented as a sparse vector

wᵢ ∈ ℝ≥0^|V|,

with coordinates indexed by the vocabulary V.

---

### 2.3 Cosine Similarity, Ranking, and Stability Quantities

Given two non-zero TF-IDF vectors u, v ∈ ℝ≥0^|V|, the cosine similarity is

cos(u, v) = (u · v) / (‖u‖₂ ‖v‖₂),

with the convention that the similarity is set to zero if either vector is the zero
vector. All coordinates are non-negative, so

cos(u, v) ∈ [0, 1]

**in exact arithmetic**. In binary64 the computed value may exceed 1 by a few units in the last
place, because the dot product, the two norms and the division round independently; this was
observed in 27% of 40,000 random trials, the largest excess in that sample being 3 ulp. That
figure is a sample maximum and not a bound — an independent resample reached 4 ulp — so a
consumer must not treat 3 ulp as a limit. No clamping is applied, in keeping
with the numerical commitments of section 6, so a consumer converting a similarity to an angle
must clamp at its own call site. See `docs/spec_addenda.md`, G24.

Given a query vector q ∈ ℝ≥0^|V|, embedded using the same vocabulary V and IDF mapping as
the corpus documents, and a collection of document vectors { wᵢ }, the similarity scores

sᵢ = cos(q, wᵢ)

are computed for each document.

---

#### 2.3.1 Ranking Operator

Similarity scores alone do not define a total order when ties occur. The final
ranking is therefore given by a deterministic sorting operator

π = Sort((sᵢ, aᵢ)ᵢ),

where Sort orders documents by decreasing similarity score sᵢ and resolves ties
lexicographically using the fixed attribute tuple aᵢ (e.g. popularity, rating,
engagement), followed by the document **identifier**. The identifier is not an entry of aᵢ
and is never permutable: it is appended implicitly to every key, and it is what makes the
order total. This yields a total ordering (r₁, r₂, …, rₙ), where rⱼ denotes the document at
rank j.

The guarantee is conditional on identifiers being **unique**, which is validated at
construction rather than assumed. With duplicates the order ceases to be uniquely determined
and results become dependent on the sorting algorithm. Given uniqueness the key is injective,
so no two elements compare equal and the sorted output is independent of the input order.

The mapping from similarity scores to rankings is consequently not continuous
globally; it is locally constant away from tie hyperplanes.

---

#### 2.3.2 Score-Separation Margins (A1)

Let score(rⱼ) denote the similarity score of the document at rank j. The **boundary margin**
at rank k is

mₖ = score(rₖ) − score(rₖ₊₁).

This quantity governs the stability of top-k membership under bounded perturbations of
similarity scores.

The **minimum adjacent margin within the top-k** is

m_min^top = min_{1 ≤ j < k} (score(rⱼ) − score(rⱼ₊₁)).

This quantity controls the stability of the ordering within the top-k set.

The corresponding **flip radius** at rank k is

εₖ^flip = mₖ / 2,

the **supremum** of the uniform perturbation magnitudes (in score space) under
which the relative ordering of ranks k and k + 1 is preserved. The supremum is not attained:
the guarantee holds for ε < εₖ^flip, and at ε = εₖ^flip the two scores can be driven
to equality, at which point membership passes to the tie-break. The bound is tight, since a
perturbation of εₖ^flip + δ flips the pair for any δ > 0, but it is worst-case. Under
*random* perturbation no flip was observed until roughly 1.1 × εₖ^flip.

Both margins have edge cases that are reported rather than coerced: mₖ is
undefined for k ≥ N, and m_min^top is undefined at k = 1, where the minimum is over an empty
set. The two constrain **disjoint** sets of gaps, so neither bounds the other; guaranteeing
set and order membership together requires ε < min(mₖ, m_min^top)/2.

---

#### 2.3.3 Tie Groups and Decision Discontinuities (A2)

Fix the ranking  
(r₁, r₂, …, rₙ)  
induced by the deterministic ranking operator π.

To formalise near-ties, fix a numerical tolerance τ ≥ 0. At rank position j, the
associated **tie group** is

G_τ(j) = { i : |sᵢ − score(rⱼ)| ≤ τ }.

The scores sᵢ here are the same similarity scores used to produce the ranking (rⱼ).

Documents within a tie group are indistinguishable *from the reference document rⱼ* at the
level of similarity scores up to numerical tolerance. They are **not** necessarily mutually
indistinguishable: the relation |sᵢ − sⱼ| ≤ τ is reflexive and symmetric but **not transitive**,
so G_τ(j) is a ball around score(rⱼ) rather than an equivalence class, and the family of balls
does not partition the corpus. Practice therefore requires three distinct objects: the
ball above, its transitive closure (single linkage), and the maximal mutually-indistinguishable
sets (complete linkage), together with the ratio between the last two, which measures how far
transitive chaining has inflated the reported group. See `docs/spec_addenda.md`, G1.

In all three cases the final ordering within the group is determined entirely by the
deterministic tie-breaking rules embedded in the ranking operator π.

This construction separates **numerical stability of similarity scores** from
**stability of the induced ranking**, and provides a formal basis for analysing
decision-level discontinuities arising from secondary ordering criteria.

---

## 3. Solution Procedure and Implementation Structure

The repository implements the TF-IDF pipeline and the associated
perturbation-theoretic investigations explicitly and reproducibly. Each stage
exposes intermediate quantities and algebraic structure rather than optimising
performance or hiding implementation detail.

- **Preprocessing and Corpus Construction**  
  A fixed, deterministic preprocessing map is applied to raw **text of interest**:
  normalisation, tokenisation, stopword removal, lemmatisation and n-gram
  generation. The result is a reproducible corpus whose intermediate quantities
  can be inspected directly, which is what makes **controlled perturbation
  analysis** possible.

- **TF-IDF Vectorisation**  
  A pure-Python TF-IDF vectoriser constructs the vocabulary, computes document
  frequencies and smoothed inverse document frequency (IDF) values, and embeds documents
  as vectors in ℝ≥0^|V|. In typical use these embeddings are sparse, the vocabulary
  being large relative to document length.

- **Similarity, Ranking, and k-NN Structure**  
  Cosine similarity is computed between query vectors and corpus vectors, followed by
  deterministic ranking. Under **content-based k-nearest-neighbour
  recommendation**, the top-k elements of this ranking are the neighbourhood
  associated with a query. Secondary attributes, such as auxiliary metadata or
  identifiers, enter through lexicographic tie-breaking and yield a **total order** on
  candidate items.  
  The implementation also computes the **score-separation margins** governing
  top-k membership and within-top-k ordering, and supports stability profiling of
  rankings under bounded perturbations. Observed margins serve as empirical
  certificates of stability, so explicit noise injection is not required. The
  ranking procedure is further instrumented for **tie-breaking ablation
  experiments**, in which ordering is recomputed under alternate tie-break priorities
  or under score-only sorting with a fixed, attribute-independent identifier as the
  final deterministic tie-break, which separates decision-level effects
  attributable to secondary attributes from the numerical similarity computation.

- **User-Profile Documents**  
  User-specific documents are constructed from interactions such as liked, viewed or
  favourited items. They are embedded using the **same vocabulary and IDF mapping
  as the corpus**, so every similarity computation takes place in one vector space.

- **Perturbation Analysis**  
  The explicit structure supports analytical study and targeted empirical
  inspection of how small perturbations in documents, corpus composition or user
  interactions propagate through document frequencies, IDF values, TF-IDF
  embeddings, cosine similarities, induced k-NN neighbourhoods and ranking
  outcomes. Mechanisms that amplify small upstream perturbations become visible
  where sparse vector geometry, IDF scaling, angular similarity and deterministic
  decision rules interact.

The code preserves **algebraic clarity** and exposes intermediate quantities at
every stage, which suits it to further mathematical analysis of **stability**,
**sensitivity** and **decision-level fragility** in TF-IDF-based similarity systems.

Given a fixed corpus, configuration and software environment (library versions
included), all stages of the pipeline are deterministic, so similarity scores and
rankings reproduce across runs.

---

## 4. Error Analysis and Perturbation Quantities

Let wᵢ denote the TF-IDF vector associated with document dᵢ, and let

sᵢ = cos(q, wᵢ)

denote the similarity score between a query vector q and the i-th document. This section
gives quantitative measures for how perturbations affect intermediate quantities
in the TF-IDF pipeline, and how those effects reach the similarity scores and the
induced rankings.

Throughout, perturbations are treated as **bounded changes** in intermediate numerical
quantities. No probabilistic or adversarial noise model is assumed unless stated
explicitly. In the implementation they are analysed mainly at the level of
similarity scores and induced rankings, with upstream effects treated analytically.

---

### 4.1 Perturbations in Document Frequency and IDF

Consider a perturbation of the corpus induced by adding or removing a document, or by
modifying the token content of an existing document. Let df(t) and df′(t) denote the
document frequencies of token t before and after perturbation, and let N and N′ denote
the corresponding corpus sizes.

Under the smoothed IDF definition employed throughout the implementation,

idf(t) = log((1 + N) / (1 + df(t))) + 1,

the corresponding change in IDF is

Δidf(t) = idf′(t) − idf(t)  
    = log((1 + N′)/(1 + df′(t))) − log((1 + N)/(1 + df(t))).

This expression exhibits the competing effects of changes in corpus size and of
changes in the document-frequency distribution. Tokens with low document
frequency remain sensitive to corpus perturbations even under smoothing.

---

### 4.2 Perturbations in TF-IDF Vectors

Let wᵢ and wᵢ′ denote the TF-IDF vectors of a document before and after perturbation.
Writing

wᵢ  = tfᵢ ⊙ idf  
wᵢ′ = tfᵢ′ ⊙ idf′

where ⊙ denotes pointwise (Hadamard) multiplication, we obtain

wᵢ′ − wᵢ  
= (Δtfᵢ) ⊙ idf + tfᵢ ⊙ (Δidf) + (Δtfᵢ) ⊙ (Δidf),

with Δtfᵢ := tfᵢ′ − tfᵢ and Δidf := idf′ − idf.

Applying the inequality ‖a ⊙ b‖₂ ≤ ‖a‖₂ ‖b‖∞ termwise yields the bound

‖wᵢ′ − wᵢ‖₂ ≤ ‖Δtfᵢ‖₂ · ‖idf‖∞  
       + ‖tfᵢ‖₂ · ‖Δidf‖∞  
       + ‖Δtfᵢ‖₂ · ‖Δidf‖∞.

The decomposition separates perturbations arising from **local document edits**
(Δtfᵢ), **global corpus changes** (Δidf), and their interaction. In sparse
high-dimensional embeddings, the interaction of a local change with globally
scaled IDF weights is a natural mechanism for perturbation amplification.

---

### 4.3 Perturbations in Cosine Similarity

Let u, v be TF-IDF vectors and let u′, v′ denote their perturbed counterparts. Under
mild assumptions on the norms of these vectors, a standard inequality yields

|cos(u′, v′) − cos(u, v)| ≤ C (‖u′ − u‖₂ + ‖v′ − v‖₂),

for a constant C depending on lower and upper bounds on ‖u‖₂, ‖v‖₂, ‖u′‖₂, and ‖v′‖₂.

This is a Lipschitz-type bound on **score stability** under bounded perturbations
of TF-IDF vectors. It controls the magnitude of numerical changes in similarity
scores, but it does not by itself determine the stability of induced rankings once
deterministic tie-breaking rules are present (see §4.5), least of all in regimes
where score-separation margins are small or ties occur.

---

### 4.4 Ranking Stability via Score-Separation Margins (A1)

Let (r₁, r₂, …, rₙ) denote the ranking induced by sorting similarity scores in decreasing
order, and let score(rⱼ) denote the similarity score of the document at rank j.

The **boundary margin** at rank k is

mₖ = score(rₖ) − score(rₖ₊₁).

If similarity scores are subject to a uniform perturbation bounded by ε, i.e.

|Δsᵢ| ≤ ε  for all i,

then the top-k set is invariant under perturbation whenever

ε < mₖ / 2.

Similarly, the **minimum adjacent margin within the top-k** is

m_min^top = min_{1 ≤ j < k} (score(rⱼ) − score(rⱼ₊₁)).

Under the same uniform bound, the ordering within the top-k set is preserved whenever

ε < m_min^top / 2.

These are explicit, sufficient criteria for ranking stability in terms of
score-separation margins. They depend only on similarity scores and take no
account of the secondary ordering rules applied in the presence of ties.

---

### 4.5 Tie-Breaking Discontinuities and Decision Sensitivity (A2)

Let aᵢ denote a vector of secondary attributes associated with document i (e.g.
popularity, rating, engagement, identifier). The final ranking operator is

π = Sort(sᵢ, aᵢ),

where similarity scores form the primary key and secondary attributes are applied
lexicographically to resolve ties.

To isolate the effect of tie-breaking, define a **score-only ranking**

π_score = Sort(sᵢ, idᵢ),

where idᵢ is a fixed, stable identifier whose sole role is to impose a
deterministic but attribute-independent order among equal scores.

An **alternate tie-break ranking** reorders the priority of the secondary
attributes:

π_alt = Sort(sᵢ, aᵢ with reordered priority).

Fix a numerical tolerance τ > 0 and consider the regime in which the boundary
margin satisfies

mₖ ≤ τ.

Documents near the top-k boundary may then form a tie group in which
|sᵢ − sⱼ| ≤ τ. Even when

Δsᵢ ≈ 0,

the top-k set or its ordering may differ between π, π_score, and π_alt, through the
choice of deterministic tie-breaking rule alone.

This motivates a notion of **tie-break sensitivity**, measurable for example by:

- an indicator of whether the top-k set differs between π and π_score, and 
- a distance between orderings restricted to tie groups (e.g. inversion count or
  Kendall τ distance).

These quantities capture **decision-level instability** that is independent of
numerical error in similarity computation and arises purely from secondary ordering
criteria.

---

## 5. Interpretation and Scope

Several structural features of **TF-IDF-based similarity systems** become clear once the pipeline is expressed in 
**operator-level form**:

- **IDF sensitivity is governed by explicit logarithmic dependence** on corpus size and document-frequency counts, as seen in 
  Δidf(t) = log((1 + N′)/(1 + df′(t))) − log((1 + N)/(1 + df(t))). 
  IDF stability is thereby traceable to perturbations in corpus composition.

- **TF-IDF perturbations admit an explicit decomposition** into *local* (TF), *global* (IDF), and *second-order interaction* terms, which shows transparently how 
  small edits propagate through the embedding.

- **Cosine similarity admits a geometric interpretation** as the cosine of the angle between *sparse, non-negative vectors*. This framing clarifies how sparsity 
  patterns and IDF scaling influence angular distortion under perturbation.

- **Ranking robustness can be characterised in terms of score-separation margins**, with explicit sufficient conditions ensuring invariance under bounded perturbations. Under 
  **content-based k-nearest-neighbour recommendation**, such ranking stability is the stability of the induced neighbourhoods.

- **Ranking stability is governed primarily by margin distributions, rather than by aggregate or average score changes alone.** 
  Small score-separation margins near decision boundaries dominate stability behaviour even when aggregate similarity scores are 
  numerically well-conditioned.

- **A long tail of near-zero margins can imply rare but extreme fragility.** 
  Most rankings may be stable under small perturbations, yet documents near top-k boundaries with vanishing margins can induce abrupt changes in neighbourhood 
  structure under otherwise negligible score variation.

- **Deterministic tie-breaking introduces non-perturbative discontinuities.**  
  With no meaningful numerical perturbation at all (Δs ≈ 0), ranking outcomes may change through secondary ordering rules alone: stability of computed similarities 
  does not guarantee stability of downstream decisions.

The emphasis throughout is on **derivational transparency** rather than algorithmic optimisation. No dimensionality reduction, latent-semantic modelling, or neural 
embeddings are introduced. The aim is to expose the **algebraic and geometric structure** of the TF-IDF pipeline in a form suitable for 
**perturbation analysis**, **stability reasoning**, and **controlled experimentation**.

---

## 6. Limitations

Several limitations of the present framework should be noted:

- **High-dimensional sparsity complicates geometric intuition.**  
  TF-IDF vectors inhabit a large, sparse subset of ℝⁿ, where small changes in support can
  produce disproportionately large angular effects, hence instability in
  similarity scores and, by extension, in induced k-NN neighbourhoods.

- **Cosine similarity becomes unstable for low-norm vectors.**  
  When documents are short or contain few in-vocabulary tokens, the norms ‖u‖₂ and ‖v‖₂
  may become small, which raises sensitivity to perturbations in similarity values. In
  such regimes score-separation margins tend to shrink, so ranking outcomes fall
  increasingly to deterministic tie-breaking rules rather than to similarity
  geometry.

- **Smoothed IDF reduces but does not eliminate volatility associated with rare tokens.**  
  Tokens with very low document frequency remain highly sensitive to corpus perturbations,
  even under smoothing, and can dominate similarity computations in sparse settings.

- **Deterministic tie-breaking can dominate outcomes under near-ties.**  
  When score-separation margins fall below numerical tolerance, ranking outcomes may be
  determined primarily by secondary ordering rules. Stability results resting solely
  on similarity scores or margin conditions then fail to capture the
  decision-level instability in the induced rankings.

- **No numerical optimisation techniques are employed.**  
  Stabilising transformations, such as sublinear TF scaling, vector normalisation
  variants or dimensionality reduction, are omitted to preserve **analytical
  clarity** and the direct interpretability of perturbation effects.

The framework is accordingly intended for **analytical**, **educational**, and
**controlled experimental use** rather than production deployment or large-scale
retrieval tasks.

---

## 7. Experimental Protocol and Results

This section gives the evaluation protocol and the empirical results for the two
stability questions analysed throughout:

- (A1) How score-separation margins govern the stability of similarity-based rankings
  under bounded perturbations.
- (A2) How deterministic tie-breaking rules induce decision-level discontinuities in
  the presence of near-ties.

The experiments characterise stability and fragility properties of the implemented
TF-IDF plus cosine ranking pipeline under a fixed, deterministic preprocessing,
embedding and ranking configuration. They do not benchmark retrieval quality
against external baselines.

---

### 7.1 Query Set and Evaluation Setup

**Corpus and representation.**  
All documents are embedded using the fixed preprocessing map, vocabulary construction,
smoothed IDF definition, and TF normalisation specified in §2.1 and §2.2. Similarity scores
are computed exclusively using cosine similarity as defined in §2.3.

**Query construction.**  
Two query construction strategies are used, as implemented in the repository:

- **User-profile queries:** a query document is constructed by aggregating text from a
  user’s interacted items (e.g. liked, viewed, or favourited), and embedded into the
  same TF-IDF space as the corpus.
- **Leave-one-out evaluation:** for users with multiple interactions, one interacted
  item is removed and treated as a held-out target; the remaining interactions form
  the query profile.

Other supported query modes (e.g. item-as-query) are part of the implementation but
are not evaluated in the present experiments.

**k values.**  
All stability metrics are evaluated for k ∈ {5, 10, 20, 50}.

**Near-tie tolerance.**  
A fixed numerical tolerance τ > 0 defines the near-tie regime. Two similarity scores
sᵢ and sⱼ are treated as indistinguishable whenever |sᵢ − sⱼ| ≤ τ, and tie groups are
defined as in §2.3.3. The value of τ is chosen to exceed floating-point noise while
remaining small relative to typical score separations. All tie-break sensitivity
results are explicitly conditional on this choice of τ.

All reported distributions are computed over the full set of evaluated queries;
the exact number of queries and users depends on the dataset configuration provided
in the repository.

Unless stated otherwise, all reported results use a fixed τ across queries.

---

### 7.2 Margin Distributions and Ranking Stability (A1)

For each query, let (r₁, r₂, …) denote the ranking induced by sorting similarity scores
in decreasing order. Boundary and within-top-k margins are defined as in §§2.3.2 and 4.4:

- mₖ = score(rₖ) − score(rₖ₊₁)
- m_min^top = min_{1 ≤ j < k} (score(rⱼ) − score(rⱼ₊₁))

For each k ∈ {5, 10, 20, 50}, empirical distributions of mₖ and m_min^top are computed
across all evaluated queries.

**Reported statistics.**
- Percentile summaries of mₖ and m_min^top (including lower-tail, median, and upper-tail
  behaviour).
- Corresponding flip radii εₖ^flip = mₖ / 2, which provide sufficient bounds for top-k
  stability under uniform score perturbations.

**Visualisation.**
- Empirical cumulative distribution functions (ECDFs) or histograms of mₖ.
- ECDFs or histograms of m_min^top.

These distributions show whether ranking stability is typical (margins well
separated from zero) or fragile (substantial mass concentrated near zero), particularly
at decision boundaries.

---

### 7.3 Tie-Break Ablations and Decision Sensitivity (A2)

To isolate decision-level effects arising from deterministic secondary ordering,
rankings are recomputed under the following sorting operators (§4.5):

- **Full ranking:** π = Sort(sᵢ, aᵢ)
- **Score-only ranking:** π_score = Sort(sᵢ, idᵢ)
- **Alternate tie-break ranking:** π_alt = Sort(sᵢ, aᵢ with reordered priority)

For each query and each k, the following quantities are measured:

- **Top-k disagreement rate:** the fraction of queries for which the top-k set differs
  between π and π_score, and between π and π_alt.
- **Within-top-k reordering:** an ordering distance restricted to tie-affected
  subsets (e.g. inversion count or Kendall τ distance).

Results are stratified by the boundary margin mₖ relative to τ, since tie-break effects
are expected to concentrate in the near-tie regime mₖ ≤ τ.

**Visualisation.**
- Probability of top-k disagreement as a function of mₖ, highlighting the transition
  region around mₖ ≈ τ.

This analysis separates numerical stability of similarity scores from instability
introduced purely by deterministic decision rules.

---

### 7.4 Constructed Near-Tie Case Study

To exhibit decision-level discontinuities explicitly, a near-tie case is reported
for a fixed query:

- Two documents A and B are identified such that |s_A − s_B| ≤ τ near the top-k
  boundary (or as adjacent elements within the top-k).
- The tuple (s_A, s_B, mₖ, τ) and the associated tie-break attributes (a_A, a_B) are
  reported.
- Ranking outcomes under π, π_score, and π_alt are compared.

The case study is **illustrative rather than representative**. It makes concrete
that ranking outcomes can change with Δs ≈ 0, driven solely by deterministic
tie-breaking rules, and it instantiates the decision-level discontinuities
analysed abstractly in §§2.3.3 and 4.5.

---

## 8. References

The following sources provide the theoretical, numerical, and conceptual background that informs the present work.

### Classical Information Retrieval

- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.  
  A foundational treatment of TF-IDF, vector-space models, and classical retrieval pipelines.

- Zobel, J., & Moffat, A. (2006). *Inverted files for text search engines*. ACM Computing Surveys.

- Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*.  
  The classical reference for TF-IDF weighting schemes and early vector-space retrieval.

- Cover, T. M., & Hart, P. E. (1967). *Nearest neighbor pattern classification*. IEEE Transactions on Information Theory.

### Numerical Linear Algebra and Stability

- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM.  
  A standard reference on perturbation analysis, conditioning, and stability in numerical computation.

- Trefethen, L. N., & Bau, D. (1997). *Numerical Linear Algebra*. SIAM.  
  Provides geometric intuition for high-dimensional vector spaces and operator behaviour.

### Sparse Vector Geometry and Similarity

- Aggarwal, C. C. (2015). *Data Mining: The Textbook*. Springer.  
  Discusses sparsity, high-dimensional geometry, and similarity measures in data-analytic contexts.

- Leskovec, J., Rajaraman, A., & Ullman, J. D. (2020). *Mining of Massive Datasets*. Cambridge University Press.  
  Covers vector-space models, similarity search, and large-scale retrieval behaviour.

### Statistical Learning Context

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.  
  Provides broader context for feature representations and similarity-based methods.

- Shalev-Shwartz, S., & Ben-David, S. (2014). *Understanding Machine Learning: From Theory to Algorithms*. Cambridge University Press.  
  Offers a theoretical perspective on learning systems that rely on vector-space representations.

### Online Resources

- *Cosine similarity*. Wikipedia.  
  https://en.wikipedia.org/wiki/Cosine_similarity

- *TF-IDF*. Wikipedia.  
  https://en.wikipedia.org/wiki/Tf%E2%80%93idf

- *Vector space model*. Wikipedia.  
  https://en.wikipedia.org/wiki/Vector_space_model

These online resources provide accessible summaries of standard definitions and terminology used throughout the repository.

---

## 9. Authorship

**Implementation and exposition**  
Matthew Maksymilian Miezaniec  
Email: matthewmiezaniec1@gmail.com  

The implementation covers the full TF-IDF similarity pipeline, the explicit perturbation analysis tooling, **stability 
profiling via score-separation margins**, and a **tie-break ablation framework** for isolating decision-level 
discontinuities in ranking outcomes.

**Mathematical and theoretical foundations**  
This work draws on **classical information retrieval methodology**, including TF-IDF weighting and vector-space models 
(Salton; Manning et al.), and on established treatments of **numerical stability**, **conditioning**, and 
**perturbation behaviour** in high-dimensional vector spaces (Higham; Trefethen & Bau).

Reading similarity scores as **content-based k-nearest-neighbour ranking and neighbourhood structure** 
follows classical nearest-neighbour and similarity-search perspectives (Cover & Hart), with no learning-based methods introduced.

Broader contextual connections to feature representations and similarity-based reasoning draw on standard statistical learning 
references (Hastie, Tibshirani & Friedman; Shalev-Shwartz & Ben-David). Supplementary intuition and terminology come from 
widely used online references on **TF-IDF**, **cosine similarity**, and **vector-space models**.

---

## 10. Acknowledgements

The author thanks colleagues and peers for informal discussions that helped clarify aspects of numerical stability, sparse 
vector geometry, and similarity-based reasoning. The work also benefited from exposure to standard academic treatments of 
information retrieval and numerical linear algebra through coursework, independent study, and open-source documentation.

Any remaining errors or omissions are the responsibility of the author alone. This acknowledgement does not imply endorsement 
or direct contribution by any individual or institution.

---

## 11. License

This repository is provided for analytical, educational, and research-oriented use.  
See the accompanying license file for full terms and conditions.
