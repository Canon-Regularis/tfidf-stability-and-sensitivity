"""Timing the native backend against the normative reference.

The C++20 core exists as an optimisation, so it has to be fast while staying
bit-identical. The differential suite enforces bit-identity; nothing enforced
speed, which left "fast" as an assertion. This module measures it.

Four rules shape the design.

A benchmark is never a correctness escape hatch. Every comparison proves the two
backends produced bit-identical results before it may report a ratio, and raises
:class:`BitIdentityError` otherwise. Nobody reads timing output looking for wrong
answers.

The minimum rather than the mean. See :func:`measure`.

Reference-only is a first-class mode: a contributor with no compiler gets
reference-only timings rather than an error.

Like-for-like, or say so. Two comparisons pair operations of different scope (the
native index constructor also builds an inverted index the reference has no
counterpart for) and carry a note giving the direction of the bias.

The workload is the seeded synthetic corpus, which contains exact duplicates and
twin pairs. The ranking rows need those: on distinctly scored documents the
tie-break is never consulted, so random scores would time the comparator's first
component and nothing else.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final

from tfidf_stability._native import native_available, unavailable_reason
from tfidf_stability.datasets.synthetic import SyntheticSpec, generate
from tfidf_stability.ranking.attributes import AttributeTable
from tfidf_stability.ranking.ranker import rank, rank_top_k
from tfidf_stability.ranking.sort_keys import PI
from tfidf_stability.similarity.cosine import cosine_against_corpus
from tfidf_stability.utils.numerics import Reduction, bits_of, reduce_sum, same_bits
from tfidf_stability.utils.validation import TfidfStabilityError
from tfidf_stability.vectorisation.sparse import SparseVector
from tfidf_stability.vectorisation.tfidf import TfidfModel, TfidfVectoriser

__all__ = [
    "DEFAULT_REPEATS",
    "BenchmarkReport",
    "BitIdentityError",
    "Comparison",
    "Timing",
    "Workload",
    "format_report",
    "measure",
    "run_benchmarks",
]

#: Timed batches per measurement. Seven gives the minimum a fair chance of a
#: quiet interval on a loaded machine without making a full run tedious.
DEFAULT_REPEATS: Final[int] = 7

#: Batches shorter than this are not trusted. ``perf_counter`` resolves to about
#: 100 ns on Windows and the Python call overhead is tens of nanoseconds, so a
#: batch must run orders of magnitude longer than either before the ratio of two
#: of them means anything.
_MIN_BATCH_SECONDS: Final[float] = 0.02

#: Ceiling on the auto-scaled inner repetition count, so a mistakenly trivial
#: callable cannot spin for minutes trying to reach the batch floor.
_MAX_INNER: Final[int] = 1 << 22

_REDUCTIONS: Final[tuple[Reduction, ...]] = tuple(Reduction)


class BitIdentityError(TfidfStabilityError):
    """The native backend disagreed with the reference inside a benchmark.

    A package error rather than :class:`AssertionError`: a finding about the
    build must survive ``-O``.
    """


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Timing:
    """One measured cost, with everything needed to judge it.

    Attributes:
        label: What was timed.
        backend: ``"reference"`` or ``"native"``.
        seconds: Best observed cost of a single call.
        repeats: How many timed batches the best was taken over.
        inner: Calls per batch, chosen automatically so each batch clears
            :data:`_MIN_BATCH_SECONDS`.
    """

    label: str
    backend: str
    seconds: float
    repeats: int
    inner: int

    @property
    def calls(self) -> int:
        """Total calls made, warm-up and calibration excluded."""
        return self.repeats * self.inner

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "backend": self.backend,
            "seconds": self.seconds,
            "repeats": self.repeats,
            "inner": self.inner,
        }


def _time_batch(fn: Callable[[], object], inner: int) -> float:
    """Wall-clock seconds for ``inner`` consecutive calls, with the collector off.

    A collection is charged to whichever call happens to trigger it, unrelated to
    that call's cost.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        start = time.perf_counter()
        for _ in range(inner):
            fn()
        return time.perf_counter() - start
    finally:
        if was_enabled:
            gc.enable()


def measure(
    fn: Callable[[], object],
    *,
    label: str,
    backend: str,
    repeats: int = DEFAULT_REPEATS,
) -> Timing:
    """Time ``fn`` and report the minimum per-call cost over ``repeats`` batches.

    Timing noise is one-sided: a run interrupted by another process, a page fault
    or a cache eviction can only be slower. The mean therefore estimates the true
    cost plus a machine-dependent noise term, while the minimum is the sample
    closest to interference-free execution. It is biased low against what a user
    sees on a busy machine, which is the right trade when comparing two
    implementations, where a shared additive term only obscures the difference.

    A warm-up call precedes everything, so lazy imports, a cold branch predictor
    and freshly allocated arenas land outside the measurement rather than on the
    first batch.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")

    fn()

    inner = 1
    while inner < _MAX_INNER and _time_batch(fn, inner) < _MIN_BATCH_SECONDS:
        inner *= 2

    best = min(_time_batch(fn, inner) for _ in range(repeats))
    return Timing(label=label, backend=backend, seconds=best / inner, repeats=repeats, inner=inner)


# ---------------------------------------------------------------------------
# Bit-identity guards
# ---------------------------------------------------------------------------
def check_same_bits(reference: Sequence[float], native: Sequence[float], what: str) -> str:
    """Verify two float sequences are bit-identical, or raise.

    Bit patterns, the standard the differential suite holds the native backend
    to: ``==`` calls ``-0.0`` and ``0.0`` equal, and a tolerance permits the
    divergence this project exists to detect.

    Returns a short description of what was checked, for the report.
    """
    if len(reference) != len(native):
        raise BitIdentityError(
            f"{what}: reference produced {len(reference)} values but native produced "
            f"{len(native)}. Refusing to report a speedup."
        )
    for i, (a, b) in enumerate(zip(reference, native, strict=True)):
        if not same_bits(a, b):
            raise BitIdentityError(
                f"{what}: value {i} differs -- reference {a!r} ({bits_of(a).hex()}) "
                f"!= native {b!r} ({bits_of(b).hex()}). The native backend is required "
                f"to be bit-identical; a speedup measured on a divergent result is "
                f"meaningless, so no timing is reported."
            )
    return f"{len(reference)} value{'' if len(reference) == 1 else 's'} bit-identical"


def check_same_order(reference: Sequence[int], native: Sequence[int], what: str) -> str:
    """Verify two permutations are identical, or raise.

    A ranking is a sequence of ``int32``, so bit-identity degenerates to
    element-by-element equality. The float half of the ranking layer, the sorted
    score array, goes through :func:`check_same_bits` instead.
    """
    if len(reference) != len(native):
        raise BitIdentityError(
            f"{what}: reference selected {len(reference)} documents but native "
            f"selected {len(native)}. Refusing to report a speedup."
        )
    for i, (a, b) in enumerate(zip(reference, native, strict=True)):
        if a != b:
            raise BitIdentityError(
                f"{what}: position {i} differs -- reference document {a} != native {b}. "
                f"The permutation must be identical; no timing is reported."
            )
    return f"permutation of {len(reference)} documents identical"


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Comparison:
    """One operation timed on both backends, with the identity check that
    licensed its ratio.

    Attributes:
        name: The operation.
        reference: Reference timing; always present.
        native: Native timing, or ``None`` when the backend is unavailable or
            the operation has no native counterpart.
        verified: What was proven identical before timing.
        note: Where the two sides differ in scope, with the direction of the
            resulting bias.
    """

    name: str
    reference: Timing
    native: Timing | None
    verified: str
    note: str = ""

    @property
    def speedup(self) -> float | None:
        """Reference cost divided by native cost, or ``None`` if not comparable."""
        if self.native is None or self.native.seconds <= 0.0:
            return None
        return self.reference.seconds / self.native.seconds

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "reference": self.reference.as_dict(),
            "native": None if self.native is None else self.native.as_dict(),
            "speedup": self.speedup,
            "verified": self.verified,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class Workload:
    """The corpus and query set the timings are measured on.

    Recorded in full beside the numbers: a speedup without its problem size says
    nothing.
    """

    n_docs: int = 2000
    vocab_size: int = 4000
    n_queries: int = 20
    query_length: int = 8
    k: int = 10
    seed: int = 20260811

    def spec(self) -> SyntheticSpec:
        """The generator spec, with duplicates and twins scaled to ``n_docs``.

        Scaled rather than fixed so a tiny smoke-test workload still holds the
        exact ties the ranking rows exercise, instead of tripping the generator's
        "too small for that many duplicates" guard.
        """
        return SyntheticSpec(
            seed=self.seed,
            n_docs=self.n_docs,
            vocab_size=self.vocab_size,
            n_exact_duplicates=max(1, self.n_docs // 20),
            n_twin_pairs=max(1, self.n_docs // 20),
            # Interactions belong to the profile experiments; here they would add
            # generation time and nothing else.
            n_users=0,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "n_docs": self.n_docs,
            "vocab_size": self.vocab_size,
            "n_queries": self.n_queries,
            "query_length": self.query_length,
            "k": self.k,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Every comparison from one run, plus the environment that produced it."""

    workload: Workload
    repeats: int
    comparisons: tuple[Comparison, ...]
    native: bool
    native_reason: str | None
    build: dict[str, object] | None
    n_features: int
    nnz: int

    def as_dict(self) -> dict[str, object]:
        return {
            "workload": self.workload.as_dict(),
            "repeats": self.repeats,
            "native": self.native,
            "native_reason": self.native_reason,
            "build": self.build,
            "n_features": self.n_features,
            "nnz": self.nnz,
            "comparisons": [c.as_dict() for c in self.comparisons],
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class _Fixture:
    """Everything both backends operate on. Built once, never timed."""

    workload: Workload
    features: list[list[str]]
    doc_ids: list[str]
    model: TfidfModel
    documents: list[SparseVector]
    table: AttributeTable
    queries: list[SparseVector]
    #: Scores of the first query, the input to the ranking rows.
    scores: list[float]
    #: A real vector for the reduction rows: every stored weight in the corpus.
    values: list[float]


def _build_fixture(workload: Workload) -> _Fixture:
    corpus = generate(workload.spec())
    features = [list(d) for d in corpus.documents]
    doc_ids = list(corpus.doc_ids)
    model = TfidfVectoriser().fit(features, doc_ids)
    documents = [model.document(i) for i in range(model.n_documents)]

    # Queries are drawn from the corpus at a fixed stride so their terms exist: a
    # query of unseen tokens embeds to the zero vector and would time the
    # degenerate early return instead of the scoring kernel.
    stride = max(1, len(features) // max(1, workload.n_queries))
    query_features = [
        features[i * stride][: workload.query_length]
        for i in range(min(workload.n_queries, len(features)))
    ]
    queries = [TfidfVectoriser.transform_query(q, model) for q in query_features]

    return _Fixture(
        workload=workload,
        features=features,
        doc_ids=doc_ids,
        model=model,
        documents=documents,
        table=AttributeTable.from_records(corpus.records()),
        queries=queries,
        scores=cosine_against_corpus(queries[0], documents, model.norms, Reduction.NAIVE),
        values=list(model.matrix.values),
    )


@dataclass(slots=True)
class _NativeFixture:
    """The same data, in the array form the compiled extension accepts."""

    module: Any
    index: Any
    ranker: Any
    csr: tuple[Any, Any, Any]
    query_arrays: list[tuple[Any, Any]]
    scores: Any
    values: Any


def _build_native_fixture(fixture: _Fixture) -> _NativeFixture:
    """Marshal the fixture for the native backend.

    numpy is imported here rather than at module scope: the reference backend is
    stdlib-only, and this file has to stay importable on an install with no
    compiler and no numpy.
    """
    import numpy as np

    from tfidf_stability._native import _tfidf_native as module

    model = fixture.model
    table = fixture.table
    indptr = np.array(model.matrix.indptr, dtype=np.int64)
    indices = np.array(model.matrix.indices, dtype=np.int32)
    values = np.array(model.matrix.values, dtype=np.float64)

    index = module.NativeIndex(
        indptr,
        indices,
        values,
        model.n_documents,
        model.n_features,
        int(module.REDUCTION[Reduction.NAIVE.value]),
    )

    # Attribute ranks cross the boundary as data and are never recomputed on the
    # native side, which keeps permutation identity an integer question rather
    # than a semantic one.
    names = table.names()
    flat: list[int] = []
    for row in table.rank_matrix(names):
        flat.extend(row)
    ranker = module.NativeRanker(
        np.array(flat, dtype=np.int32),
        np.array(table.id_ranks, dtype=np.int32),
        np.array([names.index(p) for p in PI.priority], dtype=np.int32),
        len(names),
    )

    return _NativeFixture(
        module=module,
        index=index,
        ranker=ranker,
        csr=(indptr, indices, values),
        query_arrays=[
            (np.array(q.indices, dtype=np.int32), np.array(q.values, dtype=np.float64))
            for q in fixture.queries
        ],
        scores=np.array(fixture.scores, dtype=np.float64),
        values=np.array(fixture.values, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# The comparisons
# ---------------------------------------------------------------------------
def _compare(
    name: str,
    *,
    reference_call: Callable[[], object],
    native_call: Callable[[], object] | None,
    verify: Callable[[], str] | None,
    repeats: int,
    note: str = "",
    reference_only_reason: str = "",
) -> Comparison:
    """Time one operation on both backends, refusing to time a divergent one.

    Verification runs first, so no path computes a ratio and checks afterwards.
    """
    if native_call is None or verify is None:
        verified = reference_only_reason or "reference only"
        return Comparison(
            name=name,
            reference=measure(reference_call, label=name, backend="reference", repeats=repeats),
            native=None,
            verified=verified,
            note=note,
        )

    verified = verify()
    reference = measure(reference_call, label=name, backend="reference", repeats=repeats)
    native = measure(native_call, label=name, backend="native", repeats=repeats)
    return Comparison(name=name, reference=reference, native=native, verified=verified, note=note)


def _fit_comparison(fixture: _Fixture, repeats: int) -> Comparison:
    """Fitting: reference only.

    The logarithm is evaluated once in exact decimal arithmetic in Python so the
    native core never sees a transcendental (G13), leaving nothing to compare
    against.
    """
    features, doc_ids = fixture.features, fixture.doc_ids
    return _compare(
        "fit (vocabulary + idf + matrix)",
        reference_call=lambda: TfidfVectoriser().fit(features, doc_ids),
        native_call=None,
        verify=None,
        repeats=repeats,
        reference_only_reason="no native counterpart exists",
        note="fitting is deliberately Python-only: idf is computed once in exact "
        "decimal arithmetic so the native core never evaluates a logarithm (G13)",
    )


def _index_comparison(fixture: _Fixture, native: _NativeFixture | None, repeats: int) -> Comparison:
    """Index build: reference row norms against the native index constructor."""
    model = fixture.model

    def reference_call() -> object:
        return model.matrix.row_norms(Reduction.NAIVE)

    if native is None:
        return _compare(
            "build index (row norms)",
            reference_call=reference_call,
            native_call=None,
            verify=None,
            repeats=repeats,
        )

    module, csr = native.module, native.csr
    policy = int(module.REDUCTION[Reduction.NAIVE.value])
    n_docs, n_features = model.n_documents, model.n_features

    def native_call() -> object:
        return module.NativeIndex(csr[0], csr[1], csr[2], n_docs, n_features, policy)

    def verify() -> str:
        # Checks what `native_call` builds, not the index `_build_native_fixture`
        # prepared earlier. The two take the same arguments, so they agree -- but
        # only the timed expression has to be right for the ratio to mean
        # anything, and an argument wrong here alone would have been timed and
        # reported as a speedup without ever being compared against the
        # reference. Every other row verifies the object it times; this one did
        # not, and the difference is invisible until the two disagree.
        built = native_call()
        return check_same_bits(
            model.matrix.row_norms(Reduction.NAIVE),
            [float(x) for x in built.norms],  # type: ignore[attr-defined]
            "row norms",
        )

    return _compare(
        "build index (row norms)",
        reference_call=reference_call,
        native_call=native_call,
        verify=verify,
        repeats=repeats,
        note="not like-for-like: the native constructor also builds the inverted "
        "index TAAT needs, so the ratio understates the norm kernel alone",
    )


def _scoring_comparison(
    fixture: _Fixture, native: _NativeFixture | None, repeats: int, algorithm: str
) -> Comparison:
    """Scoring the whole query set against the whole corpus: the hot path."""
    model, documents, queries = fixture.model, fixture.documents, fixture.queries
    norms = model.norms
    name = f"score {len(queries)} queries x {model.n_documents} docs ({algorithm.upper()})"

    def reference_call() -> object:
        return [cosine_against_corpus(q, documents, norms, Reduction.NAIVE) for q in queries]

    if native is None:
        return _compare(
            name,
            reference_call=reference_call,
            native_call=None,
            verify=None,
            repeats=repeats,
        )

    module, index = native.module, native.index
    algo = int(module.ALGORITHM[algorithm])
    query_arrays = native.query_arrays

    def native_call() -> object:
        return [index.score(qi, qv, algo) for qi, qv in query_arrays]

    def verify() -> str:
        expected: list[float] = []
        for q in queries:
            expected.extend(cosine_against_corpus(q, documents, norms, Reduction.NAIVE))
        got: list[float] = []
        for qi, qv in query_arrays:
            got.extend(float(x) for x in index.score(qi, qv, algo))
        return check_same_bits(expected, got, f"scores ({algorithm})")

    return _compare(
        name,
        reference_call=reference_call,
        native_call=native_call,
        verify=verify,
        repeats=repeats,
    )


def _reduction_comparison(
    fixture: _Fixture, native: _NativeFixture | None, repeats: int, policy: Reduction
) -> Comparison:
    """One reduction policy over every stored weight in the corpus.

    Real weights rather than random floats: the policies differ only in how they
    handle cancellation and magnitude spread, so a synthetic vector would time a
    distribution this project never sums.
    """
    values = fixture.values
    name = f"reduce {len(values)} values ({policy})"

    def reference_call() -> object:
        return reduce_sum(values, policy)

    if native is None:
        return _compare(
            name, reference_call=reference_call, native_call=None, verify=None, repeats=repeats
        )

    module, array = native.module, native.values
    code = int(module.REDUCTION[policy.value])

    def native_call() -> object:
        return module.reduce_sum(array, code)

    def verify() -> str:
        return check_same_bits(
            [reduce_sum(values, policy)], [float(module.reduce_sum(array, code))], f"sum ({policy})"
        )

    return _compare(
        name,
        reference_call=reference_call,
        native_call=native_call,
        verify=verify,
        repeats=repeats,
        note="the native side receives a prepared float64 array; marshalling is "
        "excluded from both sides",
    )


def _rank_comparison(fixture: _Fixture, native: _NativeFixture | None, repeats: int) -> Comparison:
    """The full ranking operator, pi, over a tie-heavy score vector."""
    scores, table = fixture.scores, fixture.table
    name = f"rank {len(scores)} documents (pi, full order)"

    def reference_call() -> object:
        return rank(scores, table, PI)

    if native is None:
        return _compare(
            name, reference_call=reference_call, native_call=None, verify=None, repeats=repeats
        )

    module, ranker, array = native.module, native.ranker, native.scores
    selection = int(module.SELECTION["full_sort"])

    # `rank` returns the permutation and the sorted score array; the native
    # permutation kernel returns only the former. Pairing it with the native sort
    # keeps both sides doing the same work and adds a bit-identity check on the
    # float half of the result.
    def native_call() -> object:
        return ranker.rank(array, selection), module.sorted_scores_desc(array)

    def verify() -> str:
        expected = rank(scores, table, PI)
        order = check_same_order(
            expected.order,
            [int(x) for x in ranker.rank(array, selection)],
            "ranking (pi)",
        )
        sorted_scores = check_same_bits(
            expected.sorted_scores,
            [float(x) for x in module.sorted_scores_desc(array)],
            "sorted scores",
        )
        return f"{order}; {sorted_scores}"

    return _compare(
        name,
        reference_call=reference_call,
        native_call=native_call,
        verify=verify,
        repeats=repeats,
    )


def _top_k_comparison(fixture: _Fixture, native: _NativeFixture | None, repeats: int) -> Comparison:
    """Top-k selection, which returns ``k + 1`` documents so ``m_k`` stays defined."""
    scores, table = fixture.scores, fixture.table
    k = fixture.workload.k
    m = min(k + 1, len(scores))
    name = f"rank top-{k} of {len(scores)} (pi)"

    def reference_call() -> object:
        return rank_top_k(scores, table, PI, k=k)

    if native is None:
        return _compare(
            name, reference_call=reference_call, native_call=None, verify=None, repeats=repeats
        )

    module, ranker, array = native.module, native.ranker, native.scores
    selection = int(module.SELECTION["bounded_heap"])

    def native_call() -> object:
        return ranker.top_k(array, m, selection), module.sorted_scores_desc(array)

    def verify() -> str:
        expected = rank_top_k(scores, table, PI, k=k)
        return check_same_order(
            expected.order,
            [int(x) for x in ranker.top_k(array, m, selection)],
            f"top-{k} selection (pi)",
        )

    return _compare(
        name,
        reference_call=reference_call,
        native_call=native_call,
        verify=verify,
        repeats=repeats,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_benchmarks(
    workload: Workload | None = None,
    *,
    repeats: int = DEFAULT_REPEATS,
    use_native: bool = True,
) -> BenchmarkReport:
    """Run every comparison and return the report.

    Args:
        workload: Corpus and query sizes. Defaults to :class:`Workload`.
        repeats: Timed batches per measurement; the minimum is reported.
        use_native: Set ``False`` to force the reference-only path even where the
            compiled backend exists, which is how the no-compiler install's
            benchmark is exercised on a machine that has a compiler.

    Raises:
        BitIdentityError: If any native result differs from the reference. No
            timing is reported in that case.
    """
    workload = workload or Workload()
    fixture = _build_fixture(workload)

    available = use_native and native_available()
    native = _build_native_fixture(fixture) if available else None

    comparisons = [
        _fit_comparison(fixture, repeats),
        _index_comparison(fixture, native, repeats),
        _scoring_comparison(fixture, native, repeats, "taat"),
        _scoring_comparison(fixture, native, repeats, "daat"),
        *(_reduction_comparison(fixture, native, repeats, p) for p in _REDUCTIONS),
        _rank_comparison(fixture, native, repeats),
        _top_k_comparison(fixture, native, repeats),
    ]

    build: dict[str, object] | None = None
    if native is not None:
        build = dict(native.module.build_info())

    reason = None if available else (unavailable_reason() or "the native backend was not requested")
    return BenchmarkReport(
        workload=workload,
        repeats=repeats,
        comparisons=tuple(comparisons),
        native=available,
        native_reason=reason,
        build=build,
        n_features=fixture.model.n_features,
        nnz=fixture.model.matrix.nnz,
    )


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------
def _format_seconds(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:7.1f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:7.2f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:7.2f} ms"
    return f"{seconds:7.3f} s "


def format_report(report: BenchmarkReport) -> str:
    """Render a report as a plain-text table.

    The check sits beside the ratio: a row whose check is empty has a speedup
    that means nothing.
    """
    w = report.workload
    lines = [
        f"workload   {w.n_docs} documents, |V| = {report.n_features}, nnz = {report.nnz}, "
        f"{w.n_queries} queries x {w.query_length} terms",
        f"timing     minimum of {report.repeats} batches, auto-sized, gc disabled, "
        f"warm-up excluded",
    ]
    if report.native and report.build is not None:
        b = report.build
        lines.append(
            f"native     {b.get('compiler_id')} {b.get('compiler_ver')} "
            f"({b.get('build_type')}), reproducible = {b.get('reproducible')}"
        )
    else:
        lines.append(f"native     unavailable -- {report.native_reason}")
    lines.append("")

    width = max(len(c.name) for c in report.comparisons)
    header = f"{'operation':<{width}}  {'reference':>11}  {'native':>11}  {'speedup':>8}  checked"
    lines.append(header)
    lines.append("-" * len(header))

    for c in report.comparisons:
        native = "--" if c.native is None else _format_seconds(c.native.seconds)
        speedup = "--" if c.speedup is None else f"x{c.speedup:.1f}"
        lines.append(
            f"{c.name:<{width}}  {_format_seconds(c.reference.seconds):>11}  "
            f"{native:>11}  {speedup:>8}  {c.verified}"
        )

    # Deduplicated: the four reduction rows share one caveat, and repeating it
    # would bury the two notes specific to a single row. `as_dict` still carries
    # every note per row.
    seen: dict[str, str] = {}
    for c in report.comparisons:
        if c.note:
            seen.setdefault(c.note, c.name)
    if seen:
        lines.append("")
        lines.extend(f"note  {name}: {note}" for note, name in seen.items())
    return "\n".join(lines)
