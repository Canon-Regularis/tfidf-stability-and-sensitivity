# Perturbation bounds, as implemented

Companion to README §4.1 to §4.3. Each bound in §4 is implemented as a
*checkable* object rather than a formula in prose: the code computes both sides
and the property tests search adversarially for a counterexample.

## The chain

```text
corpus edit → Δdf → Δidf → Δw (per document) → Δs (per score) → ranking change
```

`perturbation/experiments.py::run_perturbation` walks the whole chain for one
edit and returns a `PerturbationReport` carrying every intermediate, so a
violation can be localised to a stage rather than merely observed at the end.

## §4.1. Document frequency and IDF

A corpus edit changes `df` by at most one per term, but the induced `Δidf` is
**not** uniform: `idf = ln(N/df)` is steeply non-linear at small `df`, so a rare
term moves far more than a common one. `idf_perturb.py::analyse_idf_shift`
reports the shift per term and its L2 aggregate.

The models before and after an edit generally have **different vocabularies**:
adding a document can introduce terms, removing one can retire them.
`align_models` handles this explicitly rather than assuming a shared index space;
an implementation that compared vectors positionally would silently compare
different terms.

## §4.2. TF-IDF vectors, and the three-term bound

§4.2 decomposes the vector shift into a **local** term (this document's own
content changed), a **global** term (the IDF weights moved beneath it) and an
**interaction** term, and claims the interaction term is a mechanism for
amplification.

`ThreeTermBound` computes all three and records which dominated, so the claim is
checked against data rather than asserted. `PerturbationReport.dominant_terms`
counts the dominance across an experiment.

## §4.3. Cosine

Both arguments move. §3 requires a query to use the same vocabulary and IDF
mapping as the corpus it is scored against, so under a corpus perturbation the
**query vector itself moves too**. Overlooking this understates the bound;
`run_perturbation` re-embeds the query into each model separately.

## §4.4. The ranking certificate

Covered in [ranking_stability.md](ranking_stability.md). The essential asymmetry:
`certified_stable` returning `True` is a **proof**; returning `False` means only
"not covered by the certificate" and predicts nothing about whether the ranking
changes. Measured, ~60% of uncertified perturbations left the top-k unchanged
anyway.

## Property tests as adversarial search

`tests/test_perturbation_bounds.py` uses Hypothesis to **search for
counterexamples** to §4's inequalities rather than to sample typical inputs. A
bound that holds on random data and fails on a crafted edge case is no bound.

Two subtleties decide whether these tests are theorem checks or flaky annoyances:

- **Assume on the realised delta rather than the drawn one.** The proofs are over
  the reals, but `fl(s + d)` rounds, so realised movement can exceed `|d|` by half
  an ulp. A test that filtered on the drawn `d` would fail intermittently at the
  boundary, and the theorem is about the movement that happened.
- **Draw indices from the actual length.** An early test used
  `assume(j < len(s))`, which tripped Hypothesis's `filter_too_much` health check
  *intermittently*, because the example database cached the passing runs. Before
  changing any code, the search itself was brute-forced over 600,000 cases
  (signed zeros, subnormals, ulp-adjacent values) to confirm it was correct; the
  fix went into the test.

## What perturbation magnitudes actually occur

Measured over 40 corpus edits (add, remove, edit) on a 200-document corpus:

| perturbation | max abs score shift |
| --- | --- |
| corpus edit | 1.694e−04 to 1.405e−01 |
| arithmetic noise floor | ≤ 1.665e−16 |

**Thirteen orders of magnitude.** §5's injected score-level noise can reach any
magnitude by construction, but *text-level* perturbation cannot go below roughly
`1/L`; see [G22](spec_addenda.md#g22). The two perturbation models are not
interchangeable at fine scales, and the paper should say which it means.

## Reproducing

```bash
python -m pytest tests/test_perturbation_bounds.py -q --hypothesis-profile=nightly
```

The `nightly` profile runs far more examples than the PR gate; a bound that only
holds under `dev` is not established.
