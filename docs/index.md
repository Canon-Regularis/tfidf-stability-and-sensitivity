# Implementation notes

`README.md` (and `docs/main.pdf`, the same document) is the **specification**.
These pages are the **implementation companion**: what was built, where it
departs from the paper and why, and how to reproduce every number.

They do not restate the paper. Where a page overlaps a README section, the
overlap marks something the implementation found and the specification did not
say.

## Where to start

| If you want to… | Read |
| --- | --- |
| run something in two minutes | [`examples/minimal_corpus_demo.py`](../examples/minimal_corpus_demo.py) |
| understand the pipeline | [mathematical_formulation.md](mathematical_formulation.md) |
| understand A1 | [ranking_stability.md](ranking_stability.md) |
| understand A2 | [tie_breaking_discontinuities.md](tie_breaking_discontinuities.md) |
| understand the §4 bounds | [perturbation_notes.md](perturbation_notes.md) |
| reproduce the results | [experiments.md](experiments.md) |
| see every spec resolution | [spec_addenda.md](spec_addenda.md) |
| get the data | [`data/README.md`](../data/README.md) |

## The two backends

There are two implementations, and the relationship between them is the single
most important architectural fact in the repository.

**The pure-Python reference is normative.** It defines correctness. It has no
compiled dependency, runs on any platform with a stdlib Python, and every
published number is defined by what it produces.

**The C++20 core is an optimisation.** It is required to be *bit-identical*, and
the test suite enforces that by comparing **raw bit patterns**
(`struct.pack("<d", x)`), never tolerances. A tolerance would silently permit the
divergence this study is about.

```bash
pip install .                                        # both backends
pip install . -C cmake.define.TFIDF_BUILD_NATIVE=OFF # reference only, no compiler
```

CI tests the reference-only path, so the claim that it stands alone is checkable.

## Three decisions that shape everything

**Floating point is removed rather than handled carefully.** `math.log` is not
correctly rounded: it differs from the correctly-rounded value in about 15% of
IDF entries, and UCRT, glibc and Apple libm each round differently. IDF is
therefore computed once in Python via `decimal.Decimal.ln()` at 60 digits, and
the C++ core never sees a logarithm, only the operations IEEE-754 *requires* to
be correctly rounded (`+ − × ÷ √`). The same move is applied to the tie-break,
where attributes are rank-encoded to `int32` at construction, so the comparator
contains no floating point at all. See [G13](spec_addenda.md#g13).

**Reduction policy is an explicit parameter rather than a hidden default.**
`Naive` (normative, plain left-to-right fold), `Neumaier`, `Pairwise`, `Exact`
(Shewchuk). Explicitness is what allows the arithmetic noise floor to be
*measured* rather than assumed; see [G23](spec_addenda.md#g23).

**τ is never defaulted.** `configs/default.yaml` has no `τ` key, and every
function takes it explicitly. The paper gives a two-sided qualitative constraint
rather than a value, so the implementation derives an admissible **band** and
shows the choice inside it is immaterial.

## What the implementation found

Three results that are properties of the study rather than of the code, each
recorded as an addendum and enforced by a test:

- **Numerical error cannot flip a ranking of distinctly-scored documents.** The
  measured noise floor is ~1e−16 and the smallest observed score gap ~7e−10, six
  orders of magnitude apart. Every ranking instability in the unperturbed
  pipeline comes from *exact ties*, i.e. from the tie-break. ([G23](spec_addenda.md#g23))
- **Fine near-ties cannot be manufactured from text.** `tf = count/L` makes a
  one-token edit a `1/L` *relative* perturbation, so a 1e−9 separation would need
  a billion-token document. §7.4 must *identify* a near-tie, which its own
  wording permits. ([G22](spec_addenda.md#g22))
- **§4.4's bound is tight in the worst case and conservative on average.** A
  dyadic witness flips the pair at exactly `m_k/2 + δ`; random perturbations do
  not flip until ~1.1× that radius and reach ~50% around 4×. Reporting either
  alone would mislead. ([ranking_stability.md](ranking_stability.md))

## Gates

```bash
python -m pytest tests/ -q          # 592 tests, 90% coverage
ctest --preset mingw                # 79 cases, 20,074 assertions
python scripts/benchmark.py         # speedups, each gated on bit-identity
python -m ruff check src tests scripts && python -m ruff format --check src tests scripts
python -m mypy
python scripts/snapshot.py          # the cross-platform reproducibility digest
```

`scripts/snapshot.py` is the acid test: CI computes it on Linux, macOS and
Windows, at three optimisation levels, under both backends, and requires every
digest to be the same string.

## Performance

The C++ core exists to be fast *while staying bit-identical*, and every
benchmark row asserts the second before reporting the first; a speedup on a wrong
answer is worthless. Measured on 2000 documents, |V| = 3428:

| operation | reference | native | speedup |
| --- | --- | --- | --- |
| score 20 queries x 2000 docs (TAAT) | 415.9 ms | 1.37 ms | **×304** |
| score 20 queries x 2000 docs (DAAT) | 455.8 ms | 13.2 ms | ×34.5 |
| rank 2000 documents | 12.6 ms | 1.28 ms | ×9.9 |
| exact reduction | 1.55 ms | 1.63 ms | ×1.0 |

Fitting has no native counterpart: IDF is computed once in exact decimal
arithmetic so the core never evaluates a logarithm
([G13](spec_addenda.md#g13)). The exact reduction shows no speedup, as expected,
since the same Shewchuk expansion dominates in both languages.
