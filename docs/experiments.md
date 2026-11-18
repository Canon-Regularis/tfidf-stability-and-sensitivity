# Experiments: running them, and reading the output

Companion to README §7. Every number in the study is produced by one of the
scripts below and written to a JSON file carrying its own digest.

## The reproducibility contract

Each runner writes an `ExperimentResult` envelope:

```json
{
  "experiment": "stability_profile",
  "result_digest": "dbadab32e6f7…",
  "parameters":      { "dataset": "…", "k": 10, "seed": …, "model_digest": "…" },
  "data_provenance": { "kind": "synthetic", "spec_digest": "…", "redistributable": true },
  "payload":         { … },
  "environment":     { … }
}
```

`result_digest` is computed over the payload and parameters **with volatile fields
stripped**, so two runs on the same data produce the same digest although
their timestamps differ. A published number is checked by rerunning the
experiment and comparing one hex value. CI asserts this.

`data_provenance` records `"redistributable": false` for MovieLens. This marks
a result that cannot be reproduced from the repository alone.

## The runners

```bash
# E0 (tau derivation) + E1 (margin distributions) + E2 (A1 transition)
python scripts/run_stability_profile.py --dataset synthetic_tiny -o reports/

# E3 (tie-break ablation) + E4 (section 7.4 case study).  --tau is REQUIRED.
python scripts/run_tie_break_ablations.py --dataset synthetic_tiny --tau 4.8e-13 -o reports/

# plain scoring, with margins beside every ranking
python scripts/run_similarity.py --dataset synthetic_tiny -o reports/

# section 1.2's intermediates for one document, with raw bit patterns
python scripts/export_intermediates.py --dataset synthetic_tiny --doc d000000 -o reports/

# figures, rendered from the JSON above; never from a live computation
python scripts/make_figures.py --reports reports/

# E5 (retrieval on a BEIR test split); fetch the archive first
python scripts/fetch_data.py --dataset beir_scifact
python scripts/run_retrieval_eval.py --dataset beir_scifact     --archive data/raw/beir-scifact.zip -o reports/
```

`run_stability_profile.py` **exits non-zero** if any perturbation inside §4.4's
certified radius flipped the top-k set. The theorem is therefore checked against
data on every push.

Scale up with `--dataset synthetic_small --queries 200`. For MovieLens, fetch it
first (see [`data/README.md`](../data/README.md)) and pass `--archive`.

## E0. Deriving τ

§7.1's guidance is a two-sided *qualitative* constraint with no value and no
procedure. E0 measures both endpoints instead of inventing a number:

- **τ_floor = 2η**, where `η` is the worst per-score disagreement between a
  reduction policy and exact summation. The factor 2 is exact (the error in a
  *margin* is at most `e_i + e_j`) and it is the same 2 as in `ε_k^flip = m_k/2`.
- **g_min**, the smallest strictly-positive adjacent score gap.

Every τ-dependent object is **piecewise constant in τ**, with breakpoints only at
observed gap values. When the band contains no observed gap, every τ inside it
gives *bit-identical* tie structure by argument rather than by sampling.
`verify_band_invariance` recomputes at eight probes as a check on the code.

Measured (1500 documents, 25 queries): `η = 1.665e−16`, `τ_floor = 3.331e−16`,
`g_min = 6.958e−10`, a **6.32-decade** band containing **zero** observed gaps.

The caveat: invariance across the band reflects *emptiness of the score lattice*
rather than robustness of the tie-break mechanism. The plateau must always be
reported with its cause. ([G23](spec_addenda.md#g23))

`TauBand.display_tau()` exists so a caption can name a number. It is a
presentation choice and must never be cited as a derived constant.

**If the band is ever empty** (`is_valid` false), that is a finding rather than a
bug: arithmetic noise would reach the decision boundary, every §7.3 result on
that corpus would be contaminated by A1's regime, and A1 and A2 would not be
separable there. The code says so and refuses to invent a value.

## E1. Margin distributions

`m_k` across queries at each `k`, summarised with **nearest-rank** percentiles,
so every reported value is a margin that some query produced. An interpolated
percentile can be a margin that no query produced, which cannot be looked up in
the raw data or compared with `same_bits`.

Degenerate queries are excluded per G3, and the exclusion count is reported.
The **exact-tie share** is reported separately: it is G3's headline statistic,
and the percentiles hide it once it exceeds 50%.

## E2. The A1 transition

See [ranking_stability.md](ranking_stability.md).

## E3/E4. The A2 ablations and the case study

See [tie_breaking_discontinuities.md](tie_breaking_discontinuities.md).

## E5. Retrieval effectiveness on BEIR

`run_retrieval_eval.py` scores the test split of a BEIR benchmark
(`beir_scifact` or `beir_nfcorpus`) with the benchmark's own text queries and
graded judgements. For each reduction policy and each identifier tie order it
reports:

- nDCG@k (default 10), Recall@depth (default 100) and MRR@k, defined as in
  trec_eval ([Relevance metric discount](engineering_notes.md#relevance-metric-discount)).
- Per query: the boundary margin at k, the size of the exact tie at position k,
  whether the top k differ between `id_asc` and `id_desc` as a set and as an
  order, and the number of documents with a non-zero score.
- Over the seeded shuffles of the identifier order (default 20): the minimum,
  maximum and range of each mean.

The tie orders are `id_asc`, `id_desc` and `shuffle_<seed>`
([Identifier tie orders](engineering_notes.md#identifier-tie-orders)). A BEIR
document has no attribute, so the three operators of section 4.5 coincide and
the identifier order alone decides a score tie. Every variant ranks the same
scores, so a metric that moves between variants moves through the tie-break.

The corpus is fitted once, and each policy reuses the fit with its own norms,
which gives the model a refit would. Queries pass through the corpus's own
preprocessing pipeline. A query whose features are all removed is degenerate:
it ranks nothing and scores 0. A query with no relevant judgement is undefined
and left out of each mean, with a count.

Three differences from published BEIR figures are by design:

- BEIR's lexical baseline ranks with BM25; this study ranks with cosine TF-IDF
  (README section 2), so its nDCG is expected to be lower.
- trec_eval breaks score ties by reverse identifier order, which is `id_desc`
  here, so that is the variant to compare. The metric code never re-sorts a
  ranking, so the tie order under study is the one scored.
- BEIR's BM25 run returns only documents that share a term with the query.
  Here the ranking to depth fills with zero-score documents in tie order, and
  Recall@depth counts any relevant one among them; `n_nonzero` records how many
  documents scored above zero.

## Figures, and how each one could falsify its hypothesis

Figures are rendered from the recorded JSON and never recompute their data. A
figure that recomputed its data could silently disagree with the numbers in the text.
Each figure carries the digest of the result it was built from.

| figure | falsified if |
| --- | --- |
| `fig_transition.png` | the flip-rate curve rises left of `ε = m_k/2` |
| `fig_margins.png` | margins sit far from zero with no exact-tie mass |
| `fig_ablation.png` | every bar is zero (scores are bit-identical, so any bar is the tie-break) |

`matplotlib` is an optional dependency (`pip install "tfidf-stability[viz]"`); the
normative pipeline never imports it.

## Notebooks

`notebooks/` mirrors these experiments interactively. The notebooks are
exploratory. Every published number comes from a runner and its JSON file.

## Datasets

`synthetic_tiny` (120 documents) serves tests and CI, `synthetic_small` (2000)
serves the full run, `movielens_small` serves external validity, and
`beir_scifact` and `beir_nfcorpus` carry relevance judgements for E5. The
synthetic generator is seeded and byte-reproducible. CI generates the data twice
and requires the files to be identical. The check is there to catch a change that
reintroduces a PRNG call whose output is not stable across versions.
