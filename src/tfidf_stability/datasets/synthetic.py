"""A seeded synthetic corpus generator for near-ties, scaling and offline CI.

Draws use only :meth:`random.Random.random` and :meth:`random.Random.getrandbits`,
stable across CPython versions; a non-integer Zipf exponent goes through libm
(``docs/spec_addenda.md#g13``). Rationale: docs/engineering_notes.md#synthetic-prng
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from tfidf_stability.utils.hashing import hash_text
from tfidf_stability.utils.io import atomic_write_text, write_json, write_jsonl

__all__ = [
    "NearTie",
    "SyntheticCorpus",
    "SyntheticSpec",
    "find_near_ties",
    "generate",
    "write_corpus",
]

#: Scale for the integer Zipf weights. Large enough that the tail is not
#: quantised flat, small enough to stay well inside int64.
_ZIPF_SCALE = 1 << 40


@dataclass(frozen=True, slots=True)
class SyntheticSpec:
    """Everything that determines a corpus. Serialised beside the output."""

    seed: int = 20260811
    n_docs: int = 2000
    vocab_size: int = 4000
    #: Zipf exponent. ``1`` is computed exactly in integers; anything else goes
    #: through ``pow`` and is platform-dependent, see the module docstring.
    zipf_exponent: float = 1.0
    len_min: int = 3
    len_max: int = 40
    n_users: int = 200
    max_interactions_per_user: int = 12
    #: Duplicated documents, which tie exactly: the tau = 0 baseline, and the
    #: case where only the tie-break can separate two items.
    n_exact_duplicates: int = 20
    #: Twin pairs: a copy plus one extra token. The extra token's document
    #: frequency controls how far the two scores separate, giving a grid of
    #: near-tie magnitudes rather than a single value.
    n_twin_pairs: int = 40
    twin_extra_token_df: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)

    def digest(self) -> str:
        """Identity of this spec, for the manifest."""
        return hash_text(repr(sorted(asdict(self).items())))


@dataclass(frozen=True, slots=True)
class SyntheticCorpus:
    """A generated corpus, its attributes and its interactions."""

    spec: SyntheticSpec
    doc_ids: tuple[str, ...]
    documents: tuple[tuple[str, ...], ...]
    #: Per document: popularity, the exact rating pair, engagement.
    attributes: tuple[dict[str, Any], ...]
    #: ``(user_id, doc_id, weight)`` triples.
    interactions: tuple[tuple[str, str, float], ...]
    #: ``(a, b, extra_token_df)`` for each twin pair, so an experiment can find
    #: the constructed near-ties without searching for them.
    twins: tuple[tuple[str, str, int], ...]
    #: Document ids that are exact duplicates of one another.
    exact_duplicate_pairs: tuple[tuple[str, str], ...]

    @property
    def n_documents(self) -> int:
        return len(self.doc_ids)

    def features_by_doc(self) -> dict[str, tuple[str, ...]]:
        return dict(zip(self.doc_ids, self.documents, strict=True))

    def records(self) -> list[dict[str, Any]]:
        """Corpus rows in the shape the loaders and the CLI expect."""
        return [
            {"doc_id": doc_id, "text": " ".join(tokens), **attributes}
            for doc_id, tokens, attributes in zip(
                self.doc_ids, self.documents, self.attributes, strict=True
            )
        ]


# ---------------------------------------------------------------------------
# Deterministic primitives
# ---------------------------------------------------------------------------
def _zipf_weights(vocab_size: int, exponent: float) -> list[int]:
    """Integer Zipf weights, exact when ``exponent`` is 1.

    Integers so the cumulative distribution and every comparison against it are
    exact, making the sampling reproducible without depending on floating-point
    rounding.
    """
    if exponent == 1.0:
        return [max(1, _ZIPF_SCALE // (rank + 1)) for rank in range(vocab_size)]
    # Non-integer exponents go through pow, i.e. the platform libm. Permitted,
    # but the generated files rather than the spec become the artefact.
    return [max(1, int(_ZIPF_SCALE / (rank + 1) ** exponent)) for rank in range(vocab_size)]


def _cumulative(weights: list[int]) -> list[int]:
    total = 0
    out = []
    for weight in weights:
        total += weight
        out.append(total)
    return out


def _pick(rng: random.Random, cumulative: list[int]) -> int:
    """Sample an index from an integer cumulative distribution.

    ``getrandbits`` and integer comparison only. ``random.choices`` would be
    shorter and would bring in both a float comparison and a function whose
    implementation is not promised stable across CPython versions.
    """
    total = cumulative[-1]
    # A distribution with no mass cannot be sampled, and the rejection loop
    # below cannot say so: `(0).bit_length()` is 0, `getrandbits(0)` returns 0
    # and `0 < 0` is false, so it spins forever. No current caller reaches it:
    # every caller floors its weights at 1.
    if total <= 0:
        raise ValueError(
            f"the cumulative distribution carries no mass (total {total}); "
            f"every weight is zero, so no index can be drawn"
        )
    # Rejection-sample to a uniform integer in [0, total), so the result does
    # not depend on how a float division rounds.
    bits = total.bit_length()
    while True:
        draw = rng.getrandbits(bits)
        if draw < total:
            break

    lo, hi = 0, len(cumulative) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cumulative[mid] <= draw:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _uniform_int(rng: random.Random, low: int, high: int) -> int:
    """Uniform integer in ``[low, high]``, inclusive, without ``randrange``.

    Raises:
        ValueError: If the range is empty or inverted.
    """
    span = high - low + 1
    # An empty or inverted range would hang the loop below: at `span == 0`,
    # `getrandbits(0)` is 0 and `0 < 0` is false; at `span == -1` the draw is 0
    # or 1, never below -1. `generate` validates len_min against len_max and the
    # other call sites derive their range from a validated count.
    if span <= 0:
        raise ValueError(
            f"the range [{low}, {high}] contains no integer, so nothing can be "
            f"drawn from it uniformly"
        )
    bits = span.bit_length()
    while True:
        draw = rng.getrandbits(bits)
        if draw < span:
            return low + draw


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate(spec: SyntheticSpec | None = None) -> SyntheticCorpus:
    """Generate a corpus from a spec.

    Exact duplicates give ``m_k = 0`` and the ``tau = 0`` baseline; twins are a copy plus
    one token whose df sets the gap, grading the magnitudes section 7.3's transition plot uses.
    Both are recorded on the result so an experiment can locate them directly.
    """
    spec = spec or SyntheticSpec()
    # Checked before the generator draws anything. `_uniform_int` rejection-samples
    # until the draw lands below `high - low + 1`, so an inverted range makes that
    # bound non-positive and the loop never terminates: a hang rather than an
    # error, and one that cannot be caught by a test that has to finish.
    if spec.len_min < 1:
        raise ValueError(f"len_min must be at least 1, got {spec.len_min}")
    if spec.len_max < spec.len_min:
        raise ValueError(
            f"len_max={spec.len_max} is below len_min={spec.len_min}; "
            f"document lengths are drawn from the inclusive range between them"
        )
    # Checked here so the failure names the field. An empty vocabulary otherwise
    # reaches `_cumulative([])`, which indexes `[-1]` of an empty list and
    # surfaces two calls down as a bare `IndexError: list index out of range`.
    if spec.vocab_size < 1:
        raise ValueError(f"vocab_size must be at least 1, got {spec.vocab_size}")
    rng = random.Random(spec.seed)

    vocabulary = [f"w{i:05d}" for i in range(spec.vocab_size)]
    cumulative = _cumulative(_zipf_weights(spec.vocab_size, spec.zipf_exponent))

    n_base = spec.n_docs - spec.n_exact_duplicates - 2 * spec.n_twin_pairs
    if n_base < 1:
        raise ValueError(
            f"n_docs={spec.n_docs} is too small for {spec.n_exact_duplicates} duplicates "
            f"and {spec.n_twin_pairs} twin pairs"
        )

    doc_ids: list[str] = []
    documents: list[tuple[str, ...]] = []

    for i in range(n_base):
        length = _uniform_int(rng, spec.len_min, spec.len_max)
        documents.append(tuple(vocabulary[_pick(rng, cumulative)] for _ in range(length)))
        doc_ids.append(f"d{i:06d}")

    # Exact duplicates: identical text, so identical scores against every query.
    exact_pairs: list[tuple[str, str]] = []
    for j in range(spec.n_exact_duplicates):
        source = _uniform_int(rng, 0, n_base - 1)
        new_id = f"dup{j:04d}"
        documents.append(documents[source])
        doc_ids.append(new_id)
        exact_pairs.append((doc_ids[source], new_id))

    # Twins: a copy plus one extra token of controlled document frequency, so
    # the pair separates by a tunable amount rather than not at all.
    twins: list[tuple[str, str, int]] = []
    for j in range(spec.n_twin_pairs):
        source = _uniform_int(rng, 0, n_base - 1)
        target_df = spec.twin_extra_token_df[j % len(spec.twin_extra_token_df)]
        # A token drawn from the Zipf head has high df, from the tail low df;
        # picking by rank makes the intended df explicit rather than incidental.
        extra = vocabulary[min(spec.vocab_size - 1, max(0, spec.vocab_size // target_df - 1))]

        a_id, b_id = f"twin{j:04d}a", f"twin{j:04d}b"
        documents.append(documents[source])
        doc_ids.append(a_id)
        documents.append((*documents[source], extra))
        doc_ids.append(b_id)
        twins.append((a_id, b_id, target_df))

    attributes = [
        {
            "popularity": _uniform_int(rng, 0, 500),
            # G8: the exact integer pair; no float mean is ever formed. Ratings
            # are 0.5-quantised, so 2 * sum is an integer.
            "rating_sum2": _uniform_int(rng, 2, 10) * _uniform_int(rng, 1, 20),
            "rating_count": _uniform_int(rng, 1, 20),
            "engagement": _uniform_int(rng, 0, 50),
        }
        for _ in doc_ids
    ]

    interactions: list[tuple[str, str, float]] = []
    for u in range(spec.n_users):
        user = f"u{u:05d}"
        # Heavy-tailed: most users interact little, a few a great deal. Profile
        # size drives both the candidate-set size (G19) and how length-weighted
        # the profile is (G11).
        n_interactions = 1 + _pick(
            rng, _cumulative([max(1, 64 // (i + 1)) for i in range(spec.max_interactions_per_user)])
        )
        chosen: list[int] = []
        for _ in range(n_interactions):
            candidate = _uniform_int(rng, 0, len(doc_ids) - 1)
            if candidate not in chosen:
                chosen.append(candidate)
        for index in chosen:
            weight = 3.0 + _uniform_int(rng, 0, 4) * 0.5  # 3.0 .. 5.0 in 0.5 steps
            interactions.append((user, doc_ids[index], weight))

    return SyntheticCorpus(
        spec=spec,
        doc_ids=tuple(doc_ids),
        documents=tuple(documents),
        attributes=tuple(attributes),
        interactions=tuple(interactions),
        twins=tuple(twins),
        exact_duplicate_pairs=tuple(exact_pairs),
    )


def write_corpus(corpus: SyntheticCorpus, directory: Path | str) -> dict[str, str]:
    """Write the corpus, its interactions and its spec, with a digest manifest.

    Everything downstream reads these files. Nothing regenerates from the spec at
    experiment time, which keeps PRNG portability out of the reproducibility
    surface: the committed bytes are the artefact.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)

    corpus_path = write_jsonl(out / "corpus.jsonl", corpus.records())
    interactions_path = write_jsonl(
        out / "interactions.jsonl",
        [{"user_id": u, "doc_id": d, "weight": w} for u, d, w in corpus.interactions],
    )
    spec_path = write_json(
        out / "spec.json",
        {
            "spec": asdict(corpus.spec),
            "spec_digest": corpus.spec.digest(),
            "n_documents": corpus.n_documents,
            "n_interactions": len(corpus.interactions),
            # Recorded so an experiment can find the constructed cases directly
            # rather than searching for them.
            "twins": [{"a": a, "b": b, "extra_token_df": df} for a, b, df in corpus.twins],
            "exact_duplicate_pairs": [list(p) for p in corpus.exact_duplicate_pairs],
        },
    )

    digests = {
        p.name: hash_text(p.read_text(encoding="utf-8"))
        for p in (corpus_path, interactions_path, spec_path)
    }
    atomic_write_text(
        out / "MANIFEST.sha256",
        "".join(f"{d}  {n}\n" for n, d in sorted(digests.items())),
    )
    return digests


# ---------------------------------------------------------------------------
# Locating near-ties (section 7.4)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class NearTie:
    """A pair of adjacent ranks whose scores are close (section 7.4)."""

    rank: int
    score_a: float
    score_b: float
    gap: float

    @property
    def is_exact(self) -> bool:
        return self.gap == 0.0


def find_near_ties(
    sorted_scores: Sequence[float], *, limit: int = 10, strictly_positive: bool = True
) -> list[NearTie]:
    """Return the closest adjacent pairs of a ranking, tightest first (section 7.4).

    ``sorted_scores`` is non-increasing; ``limit`` caps the count; ``strictly_positive`` skips
    exact ties. ``rank`` is 1-indexed; ``gap`` is ``m_rank``. :func:`generate` twins reach ``1e-3``
    at finest (``docs/spec_addenda.md#g22``). Rationale: docs/engineering_notes.md#near-tie-floor
    """
    if limit < 0:
        # `pairs[:-1]` is a legal slice: a negative limit drops the widest gap
        # rather than returning nothing, changing which pairs section 7.4
        # reports as its case study. `build_query_grid` guards its own `limit`
        # for the same reason.
        raise ValueError(f"limit must be non-negative, got {limit}")

    pairs = [
        NearTie(i + 1, above, below, above - below)
        for i, (above, below) in enumerate(pairwise(sorted_scores))
    ]
    if strictly_positive:
        pairs = [p for p in pairs if p.gap > 0.0]
    pairs.sort(key=lambda p: (p.gap, p.rank))
    return pairs[:limit]
