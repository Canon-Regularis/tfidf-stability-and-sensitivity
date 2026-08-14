"""Reference vs native: bit-exact equivalence.

This is the load-bearing test of the whole architecture. The project's claim is
that the optimised C++20 evaluator performs *the same floating-point operations
in the same order* as the pure-Python reference, so that section 6's "no
numerical optimisation techniques are employed" stays literally true while the
experiment grid still runs in minutes rather than days.

The comparison is therefore on **bit patterns**, never on tolerances. A
tolerance-based test would pass while quietly permitting exactly the divergence
this project exists to detect. ``pytest.approx`` does not appear in this file.

Three independent implementations are compared:

* the Python reference (normative);
* the native **TAAT** kernel, which walks postings lists out of an inverted
  index into a dense accumulator;
* the native **DAAT** kernel, which merges each document's row against the query
  and never builds an inverted index at all.

TAAT and DAAT share no data structure and no loop nesting, so their agreeing to
the last bit is a much stronger statement than either matching a recorded
expectation.
"""

from __future__ import annotations

import math
import random
import struct

import numpy as np
import pytest

from tfidf_stability._native import build_info, native_available, unavailable_reason
from tfidf_stability.similarity.cosine import cosine_against_corpus
from tfidf_stability.utils.numerics import Reduction, reduce_sum, same_bits
from tfidf_stability.vectorisation.sparse import SparseVector, dot, l2_norm
from tfidf_stability.vectorisation.tfidf import TfidfModel, TfidfVectoriser

pytestmark = [
    pytest.mark.native,
    pytest.mark.differential,
    pytest.mark.skipif(not native_available(), reason=unavailable_reason() or "no native backend"),
]

if native_available():
    from tfidf_stability._native import _tfidf_native as nat  # type: ignore[attr-defined]


def _policy(p: Reduction) -> int:
    return int(nat.REDUCTION[p.value])


@pytest.fixture(scope="module")
def corpus() -> list[list[str]]:
    rng = random.Random(20260811)
    alpha = [f"t{i}" for i in range(400)]
    # Deliberately includes empty documents, which embed to the zero vector.
    return [[rng.choice(alpha) for _ in range(rng.randint(0, 40))] for _ in range(300)]


@pytest.fixture(scope="module")
def model(corpus: list[list[str]]) -> TfidfModel:
    return TfidfVectoriser().fit(corpus)


@pytest.fixture(scope="module")
def index(model: TfidfModel):  # type: ignore[no-untyped-def]
    return nat.NativeIndex(
        np.array(model.matrix.indptr, dtype=np.int64),
        np.array(model.matrix.indices, dtype=np.int32),
        np.array(model.matrix.values, dtype=np.float64),
        model.n_documents,
        model.n_features,
        _policy(Reduction.NAIVE),
    )


# ---------------------------------------------------------------------------
# Provenance and environment
# ---------------------------------------------------------------------------
def test_build_is_reproducible() -> None:
    """A build with fast-math or arch tuning must never produce published numbers.

    The contraction check is per compiler, because the two spell it differently
    and one of the spellings used to be wrong. MSVC has no negative form of
    ``/fp:contract``; the flag list carried ``/fp:contract-`` for a long time and
    MSVC answered ``warning D9002: ignoring unknown option`` and went on
    contracting, so this assertion passed on a substring of a flag that did
    nothing. ``/fp:strict`` is the documented way to forbid it there.
    """
    info = build_info()
    assert info["reproducible"] is True, f"non-reproducible build: {info}"
    assert info["fast_math"] is False
    assert info["arch_tune"] is False

    flags = str(info["numeric_flags"])
    compiler = str(info["compiler_id"])
    if compiler == "MSVC":
        # Not "contract-": that is the string that never worked.
        assert "/fp:strict" in flags, f"MSVC may contract to FMA: {flags}"
    else:
        assert "-ffp-contract=off" in flags, f"{compiler} may contract to FMA: {flags}"


def test_float_environment_is_trustworthy() -> None:
    """Catches a BLAS having set MXCSR flush-to-zero when numpy imported."""
    flags = int(nat.fp_selftest())
    assert flags == 0, nat.fp_describe(flags)


# ---------------------------------------------------------------------------
# Reduction policies
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("policy", list(Reduction))
def test_reduction_policies_are_bit_exact(policy: Reduction) -> None:
    """Every policy, over lengths spanning the pairwise tree-shape boundaries."""
    rng = random.Random(1234)
    for _ in range(60):
        n = rng.randint(0, 600)
        v = [rng.uniform(-1, 1) * 10 ** rng.randint(-14, 4) for _ in range(n)]
        got = nat.reduce_sum(np.array(v, dtype=np.float64), _policy(policy))
        assert same_bits(reduce_sum(v, policy), got), f"n={n} policy={policy}"


@pytest.mark.parametrize("n", [0, 1, 127, 128, 129, 255, 256, 257, 384, 385, 1000])
def test_pairwise_agrees_at_tree_shape_boundaries(n: int) -> None:
    """Pairwise is the policy most sensitive to how the summation tree is built.

    The two implementations originally used different formulations -- recursive
    split at n//2 versus a streaming binary-counter merge -- and first diverged
    at n = 129. Both are legitimate; the streaming one is the pinned
    specification because the dot-product kernel cannot see n in advance.
    """
    v = ([1.0] + [1e-17] * (n - 1)) if n else []
    got = nat.reduce_sum(np.array(v, dtype=np.float64), _policy(Reduction.PAIRWISE))
    assert same_bits(reduce_sum(v, Reduction.PAIRWISE), got)


def test_exact_reduction_matches_math_fsum() -> None:
    """`Exact` is the ground truth for the noise-floor study, so both languages
    must share it. CPython's half-even correction is reproduced verbatim."""
    cases = [
        [1e-16, 1.0, 1e16],  # CPython's own half-even regression case
        [1e100, 1.0, -1e100, 1.0],  # catastrophic cancellation
        [1.0] + [1e-17] * 100,  # addends individually below half an ulp
    ]
    for v in cases:
        got = nat.reduce_sum(np.array(v, dtype=np.float64), _policy(Reduction.EXACT))
        assert same_bits(reduce_sum(v, Reduction.EXACT), got), v


# ---------------------------------------------------------------------------
# Sparse primitives
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("policy", list(Reduction))
def test_dot_and_norm_are_bit_exact(policy: Reduction) -> None:
    rng = random.Random(99)
    dim = 200
    for _ in range(50):
        a = SparseVector.from_mapping(
            {rng.randrange(dim): rng.uniform(0.01, 5.0) for _ in range(rng.randint(0, 40))}, dim
        )
        b = SparseVector.from_mapping(
            {rng.randrange(dim): rng.uniform(0.01, 5.0) for _ in range(rng.randint(0, 40))}, dim
        )
        ai, av = np.array(a.indices, dtype=np.int32), np.array(a.values, dtype=np.float64)
        bi, bv = np.array(b.indices, dtype=np.int32), np.array(b.values, dtype=np.float64)
        assert same_bits(dot(a, b, policy), nat.dot(ai, av, bi, bv, dim, _policy(policy)))
        assert same_bits(l2_norm(a, policy), nat.l2_norm(ai, av, dim, _policy(policy)))


# ---------------------------------------------------------------------------
# The whole pipeline
# ---------------------------------------------------------------------------
def test_row_norms_are_bit_exact(model: TfidfModel, index) -> None:  # type: ignore[no-untyped-def]
    assert all(same_bits(a, b) for a, b in zip(model.norms, index.norms, strict=True))


def test_document_frequencies_agree(model: TfidfModel, index) -> None:  # type: ignore[no-untyped-def]
    """The inverted index's postings-list lengths must equal df exactly."""
    for t in range(model.n_features):
        assert index.df(t) == model.vocabulary.df[t]


@pytest.mark.parametrize("algorithm", ["taat", "daat"])
def test_scores_are_bit_exact(
    model: TfidfModel, index, corpus: list[list[str]], algorithm: str
) -> None:  # type: ignore[no-untyped-def]
    """The headline equivalence, over tens of thousands of individual scores."""
    rng = random.Random(555)
    alpha = sorted({t for d in corpus for t in d})
    docs = [model.document(i) for i in range(model.n_documents)]
    algo = int(nat.ALGORITHM[algorithm])

    compared = 0
    for _ in range(40):
        qf = [rng.choice(alpha) for _ in range(rng.randint(0, 25))]
        q = TfidfVectoriser.transform_query(qf, model)
        expected = cosine_against_corpus(q, docs, model.norms, Reduction.NAIVE)
        got = index.score(
            np.array(q.indices, dtype=np.int32),
            np.array(q.values, dtype=np.float64),
            algo,
        )
        for i, (a, b) in enumerate(zip(expected, got, strict=True)):
            assert same_bits(a, b), f"doc {i}: reference {a!r} != native/{algorithm} {b!r}"
            compared += 1
    assert compared > 10_000


def test_taat_and_daat_agree(model: TfidfModel, index, corpus: list[list[str]]) -> None:  # type: ignore[no-untyped-def]
    """Two structurally unrelated traversals of the same data.

    TAAT accumulates over postings lists into a dense array; DAAT merges each
    row independently and never builds an inverted index. Identical bits from
    both leaves very little room for an indexing or accumulation bug to hide.
    """
    rng = random.Random(31337)
    alpha = sorted({t for d in corpus for t in d})
    for _ in range(30):
        qf = [rng.choice(alpha) for _ in range(rng.randint(0, 30))]
        q = TfidfVectoriser.transform_query(qf, model)
        qi = np.array(q.indices, dtype=np.int32)
        qv = np.array(q.values, dtype=np.float64)
        a = index.score(qi, qv, int(nat.ALGORITHM["taat"]))
        b = index.score(qi, qv, int(nat.ALGORITHM["daat"]))
        assert all(same_bits(x, y) for x, y in zip(a, b, strict=True))


def test_scratch_reuse_does_not_leak_between_queries(model: TfidfModel, index) -> None:  # type: ignore[no-untyped-def]
    """The dense accumulator is reused across calls; a stale entry would
    silently contaminate the next query's scores."""
    q1 = TfidfVectoriser.transform_query(["t1", "t2", "t3"], model)
    q2 = TfidfVectoriser.transform_query(["t100", "t200"], model)

    def run(q):  # type: ignore[no-untyped-def]
        return index.score(
            np.array(q.indices, dtype=np.int32), np.array(q.values, dtype=np.float64), 0
        )

    first = run(q2)
    run(q1)
    after = run(q2)
    assert all(same_bits(a, b) for a, b in zip(first, after, strict=True))


def test_zero_query_scores_zero_everywhere(model: TfidfModel, index) -> None:  # type: ignore[no-untyped-def]
    """Degenerate but legitimate (spec_addenda G3): no NaN may escape."""
    empty = index.score(np.array([], dtype=np.int32), np.array([], dtype=np.float64), 0)
    assert len(empty) == model.n_documents
    assert all(s == 0.0 for s in empty)
    assert not np.isnan(empty).any()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_non_canonical_matrix_is_rejected() -> None:
    """Ascending indices are an invariant the kernels rely on, not a hint."""
    with pytest.raises(ValueError, match="not canonical"):
        nat.NativeIndex(
            np.array([0, 3], dtype=np.int64),
            np.array([2, 0, 1], dtype=np.int32),  # descending
            np.array([1.0, 1.0, 1.0], dtype=np.float64),
            1,
            4,
            _policy(Reduction.NAIVE),
        )


def test_bad_indptr_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="indptr"):
        nat.NativeIndex(
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int32),
            np.array([1.0], dtype=np.float64),
            5,
            4,
            _policy(Reduction.NAIVE),
        )


def test_non_canonical_query_is_rejected(index) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="not canonical"):
        index.score(
            np.array([5, 2], dtype=np.int32),  # descending
            np.array([1.0, 1.0], dtype=np.float64),
            0,
        )


def test_out_of_range_term_is_rejected(index) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(IndexError):
        index.df(10**9)


# ---------------------------------------------------------------------------
# The boundary is the last line of defence (G3)
# ---------------------------------------------------------------------------
def test_paired_arrays_must_agree_in_length() -> None:
    """``dot`` built its two spans independently and read past the end.

    With 64 indices and 1 value it consumed 63 doubles beyond the values buffer:
    undefined behaviour reachable from pure Python with no unsafe API, and it
    showed -- identical calls returned ``nan``, ``1.0``, ``nan``, ``nan``. The
    reference rejects the same input, and ``NativeIndex`` already checked it;
    only the free functions did not.
    """
    many = np.arange(64, dtype=np.int32)
    one = np.array([1.0], dtype=np.float64)
    full = np.ones(64, dtype=np.float64)

    with pytest.raises(ValueError, match="same length"):
        nat.dot(many, one, many, full, 64, 0)
    with pytest.raises(ValueError, match="same length"):
        nat.dot(many, full, many, one, 64, 0)
    with pytest.raises(ValueError, match="same length"):
        nat.l2_norm(many, one, 64, 0)

    # The reference refuses to build the vector at all, which is why the
    # native path is the only way to reach the read.
    with pytest.raises(ValueError, match="differ in length"):
        SparseVector(indices=tuple(range(64)), values=(1.0,), dim=64)


@pytest.mark.parametrize("policy", [-1, 4, 999])
def test_a_reduction_policy_outside_the_enumeration_is_rejected(policy: int) -> None:
    """``static_cast<Reduction>(999)`` silently fell back to a policy.

    This project's central claim is that the summation policy is never implicit
    and is recorded in every run manifest. Quietly substituting one is the worst
    available outcome: the manifest would name a policy the arithmetic did not
    use. ``Reduction(999)`` raises in Python; it does here now too.
    """
    values = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError, match="policy"):
        nat.reduce_sum(values, policy)


@pytest.mark.parametrize(
    "name", ["sorted_scores_desc", "boundary_margin", "min_adjacent_margin_top", "tie_chains"]
)
def test_every_score_taking_entry_point_rejects_nan(name: str) -> None:
    """A NaN makes ``<`` false both ways, destroying the strict weak ordering.

    ``NativeRanker.rank`` re-checked, per G3; the free functions did not, and it
    was observable: ``min_adjacent_margin_top`` returned ``inf`` where the
    normative reference returns ``nan`` -- a bit-level divergence in a core whose
    whole contract is bit-identity. Sorting 65,536 scores containing NaN did not
    crash, but it is formally undefined behaviour and must not be reachable.
    """
    scores = np.array([0.9, float("nan"), 0.1], dtype=np.float64)
    extra = {"boundary_margin": (1,), "min_adjacent_margin_top": (2,), "tie_chains": (0.1,)}
    with pytest.raises(ValueError, match="finite"):
        getattr(nat, name)(scores, *extra.get(name, ()))

    # The same call on finite scores must still work, or this proves nothing.
    getattr(nat, name)(np.array([0.9, 0.5, 0.1]), *extra.get(name, ()))


@pytest.mark.parametrize(("n_docs", "n_terms"), [(-1, 4), (0, -1), (1, -1), (-1, -1), (-5, -5)])
def test_negative_dimensions_are_rejected_before_any_size_arithmetic(
    n_docs: int, n_terms: int
) -> None:
    """These segfaulted the interpreter (SIGSEGV, rc 139) from pure Python.

    ``n_docs = -1`` satisfies ``indptr.size() == n_docs + 1`` when indptr is
    empty, so the length check waved it through; ``is_canonical()`` then called
    ``front()``/``back()`` on an empty span and ``transpose()`` sized a colptr
    from a negative count. Every later check casts to ``std::size_t``, where a
    negative wraps to something enormous, so the guard has to come first.
    """
    empty_i64 = np.array([], dtype=np.int64)
    empty_i32 = np.array([], dtype=np.int32)
    empty_f64 = np.array([], dtype=np.float64)
    indptr = empty_i64 if n_docs < 0 else np.zeros(n_docs + 1, dtype=np.int64)

    with pytest.raises(ValueError, match="non-negative"):
        nat.NativeIndex(indptr, empty_i32, empty_f64, n_docs, n_terms, 0)


@pytest.mark.parametrize("values", [[-0.0], [-0.0, -0.0], [-0.0, -0.0, -0.0], [-0.0, 0.0], [0.0]])
def test_exact_agrees_with_fsum_on_signed_zero(values: list[float]) -> None:
    """``Exact`` is the declared cross-language ground truth, so it of all
    policies must agree bit-for-bit.

    CPython's ``math_fsum`` appends the running total only ``if (x != 0.0)``;
    the C++ pushed unconditionally. For an input of nothing but negative zeros
    that made ``value()`` return -0.0 against the reference's +0.0. Because
    ``-0.0 == 0.0``, neither a tolerance nor an equality check could see it --
    only a bit comparison, which is what this project uses everywhere else.
    """
    native = nat.reduce_sum(np.array(values, dtype=np.float64), _policy(Reduction.EXACT))
    reference = reduce_sum(values, Reduction.EXACT)
    assert same_bits(native, reference), (
        f"native {struct.pack('<d', native).hex()} != reference "
        f"{struct.pack('<d', reference).hex()}"
    )
    assert same_bits(reference, math.fsum(values)), "the reference is math.fsum"
