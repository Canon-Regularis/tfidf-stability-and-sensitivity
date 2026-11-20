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
python scripts/run_retrieval_eval.py --dataset beir_scifact \
    --archive data/raw/beir-scifact.zip -o reports/

# the same split under BM25, unigrams and Lucene's one-byte document lengths
python scripts/run_retrieval_eval.py --dataset beir_scifact \
    --archive data/raw/beir-scifact.zip --weighting bm25 --n-max 1 \
    --length-norm lucene_byte -o reports/

# E6 to E10 read one score store per split; build it once (about 2 minutes)
python scripts/build_score_store.py --dataset beir_scifact --archive data/raw/beir-scifact.zip
python scripts/run_e6_census.py --dataset beir_scifact --archive data/raw/beir-scifact.zip
python scripts/run_e7_margin_noise.py --dataset beir_scifact --archive data/raw/beir-scifact.zip
python scripts/run_e8_hidden_rules.py --dataset beir_scifact --archive data/raw/beir-scifact.zip
python scripts/run_e9_ann.py --dataset beir_scifact --archive data/raw/beir-scifact.zip
python scripts/run_e10_floor.py --dataset beir_scifact
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
graded judgements, under TF-IDF cosine or BM25 (`--weighting`), with unigram or
unigram-and-bigram features (`--n-max`). BM25 follows Lucene
([G30](spec_addenda.md#g30)); `--length-norm lucene_byte` stores document
lengths as Lucene does, in one byte. Each run writes
`retrieval_<dataset>_<tfidf|bm25-<norm>>_n<n_max>.json`. For each reduction
policy and each identifier tie order it reports:

- nDCG@k (default 10), Recall@depth (default 100) and MRR@k, defined as in
  trec_eval ([Relevance metric discount](engineering_notes.md#relevance-metric-discount)).
- Per query: the boundary margin at k, the size of the exact tie at position k,
  whether the top k differ between `id_asc` and `id_desc` as a set and as an
  order, and the number of documents with a non-zero score.
- Over the seeded shuffles of the identifier order (default 20): the minimum,
  maximum and range of each mean.
- A near-tie census: per tau from 0 to 1e-4, absolute and relative to `r_k`,
  the queries with a document past position k within tau of `r_k`.
- Each policy's noise against `exact`: scores whose bits differ, the largest
  difference, and the top-k sets and orders that change
  ([Policy noise against exact](engineering_notes.md#policy-noise-against-exact)).

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

- BEIR's lexical baseline is Anserini's BM25. This study's BM25 shares its
  formula and IDF but not its stemmer, stopwords, tokeniser or float32 scores,
  so the published figures are a sanity range rather than a target. TF-IDF
  cosine is expected to score below either.
- trec_eval breaks score ties by reverse identifier order, which is `id_desc`
  here, so that is the variant to compare. The metric code never re-sorts a
  ranking, so the tie order under study is the one scored.
- BEIR's BM25 run returns only documents that share a term with the query.
  Here the ranking to depth fills with zero-score documents in tie order, and
  Recall@depth counts any relevant one among them; `n_nonzero` records how many
  documents scored above zero.

## External runs

Scores from a model outside the package are evaluated by `run_external_eval.py`
with the same metrics, tie orders and census as E5. Three inputs are read:

| Input | Format | Flag |
| --- | --- | --- |
| A run | JSON Lines `{"query_id", "doc_id", "score"}` | `--run` |
| Sparse vectors | JSON Lines `{"id", "vector": {token: weight}}`, for documents and for queries | `--sparse-corpus`, `--sparse-queries` |
| Embeddings | `.npy`, C order, `'<f4'` or `'<f8'`, one row per document | read by `evaluation/external.read_npy` |

A run either scores every document of the corpus, or with `--partial` lists the
top N per query, as an approximate search returns. A partial list is ranked
inside itself, so its margins lie between listed documents. `--reference` adds a
second run of the same queries. For a whole-corpus run the report then gives the
bit-level noise between the two and a noise floor labelled with `--label`. For
either kind it gives top-k agreement, as recall against the reference. Sparse
vectors are scored here under every reduction policy, as BM25 is.

```bash
python scripts/run_external_eval.py --dataset beir_scifact \
    --archive data/raw/beir-scifact.zip --label bge_cpu_batch32 \
    --run runs/bge_cpu_batch32.jsonl --reference runs/bge_cpu_batch1.jsonl -o reports/
```

## The score store

E6 to E10 read stored scores rather than scoring again, so every study sees the
same bits ([Score store](engineering_notes.md#score-store)).
`build_score_store.py` scores every judged query of a split under TF-IDF over
n-grams to 2 (`tfidf_n2`) and BM25 over unigrams with exact and with Lucene byte
lengths (`bm25-exact_n1`, `bm25-lucene_byte_n1`), each under the four reduction
policies, and writes `data/processed/scores/<dataset>/`. Each label is
`<condition>.<policy>`; `naive`, `neumaier` and `pairwise` name `exact` as their
reference. Other systems join under their own labels, and re-running E6 to E10
then reports them with no code change:

| Input | Flag | Label the studies expect |
| --- | --- | --- |
| SPLADE vectors (#18), scored here under every policy | `--sparse splade-float=CORPUS,QUERIES` | `splade-float.<policy>`, and `splade-x100.<policy>` for the scaled weights |
| Dense scores (#16), a whole-corpus run | `--run dense-fp32.run=RUN` | `dense-fp32.run` |
| Dense scores (#16), a matrix with its query and document ids | `--matrix dense-fp32.run=NPY,QIDS,DIDS` | `dense-fp32.run` |

`--reference LABEL=REFERENCE` names a run's or matrix's reference for E7. Every
entry of a store covers the same judged queries, so a later addition with
another `--limit` is refused.

The tables below are SciFact (300 judged queries, 5183 documents) and NFCorpus
(323, 3633), with `--cutoffs 10 100`, depth 100 and 20 shuffles.

## E6. The tie census

`run_e6_census.py` writes `e6_census_<dataset>.json`. Per row and cutoff it
reports the share of queries with an exact tie at the cutoff, the share with a
document past position k within tau of `r_k` (absolute and relative to `|r_k|`),
and each metric's range over `id_asc`, `id_desc` and the 20 shuffles
([Tie census](engineering_notes.md#tie-census)). Every row is under the pinned
`naive` policy.

| Dataset, k | Row | Exact tie | tau 1e-6 abs / rel | tau 1e-4 abs / rel | nDCG@k range | Recall@100 range | RR@k range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SciFact, 10 | TF-IDF | 0.000 | 0.000 / 0.000 | 0.037 / 0.003 | 0 | 0 | 0 |
| SciFact, 10 | BM25 exact | 0.000 | 0.000 / 0.000 | 0.003 / 0.003 | 0 | 0 | 0 |
| SciFact, 10 | BM25 lucene_byte | 0.003 | 0.003 / 0.003 | 0.003 / 0.003 | 0 | 0 | 0 |
| SciFact, 100 | TF-IDF | 0.007 | 0.013 / 0.010 | 0.437 / 0.027 | 0 | 0 | 0 |
| SciFact, 100 | BM25 exact | 0.023 | 0.023 / 0.023 | 0.033 / 0.050 | 0 | 0 | 0 |
| SciFact, 100 | BM25 lucene_byte | 0.073 | 0.073 / 0.073 | 0.083 / 0.100 | 0 | 0 | 0 |
| NFCorpus, 10 | TF-IDF | 0.241 | 0.241 / 0.241 | 0.257 / 0.241 | 0.0024 | 0.0125 | 0.0062 |
| NFCorpus, 10 | BM25 exact | 0.248 | 0.248 / 0.248 | 0.254 / 0.257 | 0.0020 | 0.0125 | 0.0046 |
| NFCorpus, 10 | BM25 lucene_byte | 0.282 | 0.282 / 0.282 | 0.282 / 0.282 | 0.0023 | 0.0124 | 0.0046 |
| NFCorpus, 100 | TF-IDF | 0.396 | 0.399 / 0.396 | 0.638 / 0.399 | 0.0053 | 0.0125 | 0.0064 |
| NFCorpus, 100 | BM25 exact | 0.427 | 0.427 / 0.427 | 0.443 / 0.449 | 0.0052 | 0.0125 | 0.0048 |
| NFCorpus, 100 | BM25 lucene_byte | 0.641 | 0.641 / 0.641 | 0.647 / 0.653 | 0.0053 | 0.0124 | 0.0049 |
| both | SPLADE float, SPLADE x100 | missing: SPLADE scores arrive with #18 | | | | | |
| both | dense fp32 | missing: dense scores arrive with #16 | | | | | |

The shares at tau 1e-12 and 1e-9 equal the exact-tie share in every row, so
the reduction policy's noise creates no near tie; the report holds them. On
SciFact a tie at the cutoff never moves a mean. On NFCorpus a quarter of the
queries tie at the cutoff at k = 10, 15 of them because every document scores
zero, and the tie order alone moves nDCG@10 by up to 0.0024 and Recall@100 by
up to 0.0125. Lucene's byte lengths raise the tie share at k = 100 on NFCorpus
from 0.427 to 0.641. The whole Recall@100 range comes from the zero-score
documents that fill each ranking to depth 100: for BM25 exact, `shuffle_0`
brings 64 relevant zero-score documents into the top 100 and `id_asc` none,
while the relevant documents with a positive score are the same under both.

## E7. Margin against noise

`run_e7_margin_noise.py` writes `e7_margin_noise_<dataset>.json`. For every
policy against `exact` it predicts that a query's top k may change only when
its margin is at most `2η`, and lists every observed change the prediction
misses ([Margin against noise](engineering_notes.md#margin-against-noise)). The
runner exits 1 if one exists.

| Dataset | Condition | η (largest, over policies) | k = 10: predicted / changed | k = 100: predicted / changed |
| --- | --- | --- | --- | --- |
| SciFact | TF-IDF | 1.7e-15 | 0 / 0 | 2 / 0 |
| SciFact | BM25 exact | 1.4e-14 | 0 / 0 | 7 / 0 |
| SciFact | BM25 lucene_byte | 7.1e-15 | 1 / 0 | 22 / 0 |
| NFCorpus | TF-IDF | 1.1e-15 | 78 / 0 | 128 / 0 |
| NFCorpus | BM25 exact | 1.8e-15 | 80 / 0 | 138 / 0 |
| NFCorpus | BM25 lucene_byte | 3.6e-15 | 91 / 0 | 207 / 0 |

All 36 tests, three policies per condition at two cutoffs on two datasets, find
no falsification, in both the global and the per-query scope. No policy changes
a top-k set. Every predicted change is a query with an exact tie at the cutoff:
its margin is 0, below `2η`, and both grids break the tie by the same
identifier order. `neumaier` equals `exact` bit for bit (`η = 0`). The float
certificate of `is_top_k_stable` agrees with the exact test on every query.

## E8. Hidden tie rules

`run_e8_hidden_rules.py` writes `e8_hidden_rules_<dataset>.json`: each stored
`naive` label, and each whole-corpus run, ranked as trec_eval, Anserini and FAISS
rank, against the tie-aware expected metric
([Hidden tie rules](engineering_notes.md#hidden-tie-rules)). FAISS ids follow
`corpus.jsonl` order unless `--insertion-order` gives another. The table gives
each rule's difference from the tie-aware value on NFCorpus at k = 10.

| Condition | Tie-aware nDCG@10 / Recall@100 / RR@10 | trec_eval | Anserini | FAISS insertion order |
| --- | --- | --- | --- | --- |
| TF-IDF | 0.3142 / 0.2525 / 0.5103 | -0.0009 / -0.0070 / -0.0030 | +0.0001 / -0.0070 / +0.0021 | -0.0009 / -0.0037 / -0.0030 |
| BM25 exact | 0.3273 / 0.2548 / 0.5245 | -0.0007 / -0.0070 / -0.0020 | -0.0002 / -0.0070 / +0.0011 | -0.0007 / -0.0037 / -0.0020 |
| BM25 lucene_byte | 0.3268 / 0.2543 / 0.5235 | -0.0011 / -0.0072 / -0.0022 | +0.0002 / -0.0071 / +0.0014 | -0.0010 / -0.0038 / -0.0022 |

On SciFact every rule equals the tie-aware value to the last reported digit.
On NFCorpus each tool's rule moves nDCG@10 by up to 0.0011 and RR@10 by up to
0.0030; RR@10 moves in opposite directions under Anserini and trec_eval, which
break ties by docid ascending and descending. Anserini's matched-only list gives the same
values as its full list here. Lucene's byte lengths, a score change rather than
a tie rule, move the tie-aware nDCG@10 by +0.0011 on SciFact and -0.0006 on
NFCorpus.

## E9. ANN decomposition

`run_e9_ann.py` writes `e9_ann_<dataset>.json`. Each `--ann LABEL=RUN`, a
top-N run in its tool's own order, is set against `--reference`, a stored
whole-corpus label, and each miss from the exact top k is classed as
approximation, numerics or tie-breaking
([Ann decomposition](engineering_notes.md#ann-decomposition)). The HNSW builds
arrive with #16, so both reports now state that no run is present; the
decomposition is tested on constructed runs only.

```bash
python scripts/run_e9_ann.py --dataset beir_scifact --archive data/raw/beir-scifact.zip \
    --reference dense-fp32.run --ann hnsw-m16=runs/hnsw_m16.jsonl \
    --ann hnsw-m32=runs/hnsw_m32.jsonl --group hnsw-m16=hnsw --group hnsw-m32=hnsw
```

## E10. The repeatability floor

`run_e10_floor.py` writes `e10_floor_<dataset>.json`: per metric, the largest
spread E6 to E9 report, with its source named, against the published
differences in `configs/published_differences.yaml`
([Repeatability floor](engineering_notes.md#repeatability-floor)).

| Dataset | Metric | Floor | Source | Published difference (BM25 flat against multifield) |
| --- | --- | --- | --- | --- |
| SciFact | nDCG@10 | 0.0011 | E8, BM25 exact against lucene_byte lengths | 0.679 / 0.665, 0.014: above |
| SciFact | Recall@100 | 0 | no experiment shows a spread | 0.925 / 0.908, 0.017: above |
| SciFact | RR@10 | 0.0013 | E8, BM25 exact against lucene_byte lengths | |
| NFCorpus | nDCG@10 | 0.0024 | E6, TF-IDF, `shuffle_19` against `shuffle_16` | 0.322 / 0.325, 0.003: within the 0.001 rounding of the table |
| NFCorpus | Recall@100 | 0.0125 | E6, BM25 exact, `id_asc` against `shuffle_0` | 0.246 / 0.250, 0.004: below |
| NFCorpus | RR@10 | 0.0062 | E6, TF-IDF, `shuffle_19` against `shuffle_8` | |

On SciFact both published differences lie well above the floor. On NFCorpus the
nDCG@10 difference between Anserini's two BM25 indexes exceeds the spread from
the tie order alone by 0.0006, less than the 0.001 to which the published values
are rounded, so the comparison cannot separate them.
The NFCorpus Recall@100 floor is the order of the zero-score documents that fill
the ranking to depth 100. Anserini returns only matching documents, so that
floor does not bound its two runs. It does bound a tool that fills the ranking,
as an exact dense search does.

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
