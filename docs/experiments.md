# Experiments — running them, and reading the output

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

`result_digest` is taken over the payload and parameters **with volatile fields
stripped**, so two runs on the same data produce the same string even though
their timestamps differ. That is what makes a published number checkable: rerun
and compare one hex value rather than eyeballing a table. CI asserts it.

`data_provenance` records `"redistributable": false` for MovieLens, so a reader
can see immediately that a given result cannot be reproduced from the repository
alone.

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

# figures, rendered from the JSON above -- never from a live computation
python scripts/make_figures.py --reports reports/
```

`run_stability_profile.py` **exits non-zero** if any perturbation inside §4.4's
certified radius flipped the top-k set. The theorem is therefore checked against
data on every push, not merely asserted in a docstring.

Scale up with `--dataset synthetic_small --queries 200`. For MovieLens, fetch it
first (see [`data/README.md`](../data/README.md)) and pass `--archive`.

## E0 — deriving τ

§7.1's guidance is a two-sided *qualitative* constraint with no value and no
procedure. Rather than invent a number, E0 measures both endpoints:

- **τ_floor = 2η**, where `η` is the worst per-score disagreement between a
  reduction policy and exact summation. The factor 2 is exact — the error in a
  *margin* is at most `e_i + e_j` — and it is the same 2 as in `ε_k^flip = m_k/2`.
- **g_min**, the smallest strictly-positive adjacent score gap.

Every τ-dependent object is **piecewise constant in τ**, with breakpoints only at
observed gap values. So when the band contains no observed gap, every τ inside it
gives *bit-identical* tie structure — by argument, not by sampling.
`verify_band_invariance` recomputes at eight probes as a check on the code.

Measured (1500 documents, 25 queries): `η = 1.665e−16`, `τ_floor = 3.331e−16`,
`g_min = 6.958e−10` — a **6.32-decade** band containing **zero** observed gaps.

**Read the caveat.** Invariance across the band reflects *emptiness of the score
lattice*, not robustness of the tie-break mechanism. The plateau must always be
reported with its cause. ([G23](spec_addenda.md#g23))

`TauBand.display_tau()` exists so a caption can name a number. It is a
presentation choice and must never be cited as a derived constant.

**If the band is ever empty** (`is_valid` false), that is a finding, not a bug: it
would mean arithmetic noise reaches the decision boundary, that every §7.3 result
on that corpus is contaminated by A1's regime, and that A1 and A2 are not
separable there. The code says so and refuses to invent a value.

## E1 — margin distributions

`m_k` across queries at each `k`, summarised with **nearest-rank** percentiles —
every reported value is an observation that actually occurred. Interpolating
would invent a margin no query produced, which could not be looked up in the raw
data or compared with `same_bits`.

Degenerate queries are excluded per G3, and the exclusion count is reported
rather than silently applied. The **exact-tie share** is broken out separately
because it is G3's headline statistic and vanishes into the percentiles once it
exceeds 50%.

## E2 — the A1 transition

See [ranking_stability.md](ranking_stability.md).

## E3/E4 — the A2 ablations and the case study

See [tie_breaking_discontinuities.md](tie_breaking_discontinuities.md).

## Figures, and how each one could falsify its hypothesis

Figures are rendered from the recorded JSON, never recomputed. A figure that
recomputed its own data could silently disagree with the numbers in the text. Each
carries the digest of the result it was built from.

| figure | falsified if |
| --- | --- |
| `fig_transition.png` | the flip-rate curve rises left of `ε = m_k/2` |
| `fig_margins.png` | margins sit far from zero with no exact-tie mass |
| `fig_ablation.png` | every bar is zero (scores are bit-identical, so any bar is the tie-break) |

`matplotlib` is an optional dependency (`pip install "tfidf-stability[viz]"`); the
normative pipeline never imports it.

## Notebooks

`notebooks/` mirrors these experiments interactively. They are exploratory
companions, not the source of any published number — that is always a runner
plus its JSON.

## Datasets

`synthetic_tiny` (120 documents) for tests and CI, `synthetic_small` (2000) for
the full run, `movielens_small` for external validity. The synthetic generator is
seeded and byte-reproducible; CI regenerates it twice and requires the files to
be identical, which is what would catch a regression reintroducing a
non-version-stable PRNG call.
