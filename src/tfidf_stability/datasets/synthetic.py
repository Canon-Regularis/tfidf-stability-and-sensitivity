"""A seeded synthetic corpus generator.

Why this exists alongside a real dataset
----------------------------------------
Real short text gives external validity but no control. This generator gives
control, and three things MovieLens cannot:

1. **Near-ties at chosen magnitudes.** Section 7.4 asks for a constructed
   near-tie case; on real data, near-ties in the interesting range are rare
   enough that finding one is a search problem. Here they are built.
2. **Scaling.** Sweeping ``N``, ``|V|`` and document length shows how margins
   scale, which the paper does not currently examine.
3. **A redistributable corpus.** MovieLens may not be redistributed, so CI
   cannot use it. This can be committed and runs offline.

Determinism, and a trap
-----------------------
``random.choice``, ``random.sample`` and ``random.shuffle`` are not promised to
be stable across CPython versions and their implementations have changed, so a
corpus generated with them would not regenerate identically on another
interpreter. Only the Mersenne Twister core is stable, so everything here comes
from :meth:`random.Random.random` and :meth:`random.Random.getrandbits` alone,
with the selection logic written out.

Transcendentals are avoided for the same reason at one remove: ``pow`` with a
non-integer exponent goes to the platform libm, which
``docs/spec_addenda.md#g13`` shows disagrees across systems. The default Zipf
exponent of 1 is computed in exact integer arithmetic; a non-integer exponent is
permitted but flagged, since it makes the spec rather than the generated files
the reproducible artefact.

Hence the design rule: the generator writes files and downstream consumes the
files. Nothing re-derives a corpus from a spec at experiment time, which keeps
PRNG portability out of the reproducibility surface.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from tfidf_stability.utils.hashing import hash_text
from tfidf_stability.utils.io import write_json, write_jsonl

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
    # A distribution with no mass cannot be sampled, and the loop below cannot
    # say so: `(0).bit_length()` is 0, `getrandbits(0)` returns 0, and `0 < 0` is
    # false, so it spins forever. Reached whenever every weight rounds to zero.
    # Raising here turns the one failure a test cannot catch into one it can.
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
    # An empty or inverted range has no integer to return, and the loop below
    # cannot report that -- it hangs. At `span == 0`, `(0).bit_length()` is 0, so
    # `getrandbits(0)` returns 0 and `0 < 0` is false; at `span == -1`,
    # `(-1).bit_length()` is 1, so the draw is 0 or 1 and never below -1. Neither
    # ever exits, and a hang is the single failure mode a test cannot assert on,
    # because the assertion is never reached.
    #
    # `generate` already checks the one spec field that reaches here (len_min
    # against len_max, below), and every other call site passes a range derived
    # from a count it has just validated. So this guard is not for today's
    # callers; it is so that the next one fails loudly instead of silently never
    # returning.
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

    What each near-tie mechanism buys:

    * **exact duplicates** give ``m_k = 0``: membership decided by the tie-break,
      and the ``tau = 0`` baseline;
    * **twins** give graded separation. A copy plus one extra token, whose
      document frequency sets how far apart the two scores land; sweeping that
      frequency produces the range of near-tie magnitudes section 7.3's
      transition plot needs.

    Both are recorded on the result, so an experiment can locate the constructed
    cases without searching the corpus.
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
    # Checked here so the failure names the field. An empty vocabulary reached
    # `_cumulative([])`, which indexed `[-1]` of an empty list and surfaced as a
    # bare `IndexError: list index out of range` from two calls down -- true, but
    # it does not say which spec field was wrong, and it is the only guard in
    # this block that was missing.
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
    (out / "MANIFEST.sha256").write_text(
        "".join(f"{d}  {n}\n" for n, d in sorted(digests.items())),
        encoding="utf-8",
        newline="\n",
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
    """Find the closest adjacent pairs in a scored ranking.

    Section 7.4 asks for two documents "identified such that ``|s_A - s_B| <=
    tau``". Identified, because a fine near-tie cannot be manufactured: section
    2.2 normalises ``tf = count / L``, so adding or removing one token moves
    every term frequency from ``c/L`` to ``c/(L+1)``, a relative perturbation of
    ``1/(L+1)``. A separation of ``1e-9`` would need a document of roughly a
    billion tokens. The twin mechanism in :func:`generate` reaches the ``1e-3``
    to ``1e-1`` range and no finer, which follows from the specification rather
    than from the generator.

    What is observed instead, measured on a 3000-document synthetic corpus:

    ==========================  =======
    adjacent gap                 share
    ==========================  =======
    exactly 0                    17.2%
    in (0, 1e-9)                  0.0%
    in [1e-9, 1e-6)               1.7%
    in [1e-6, 1e-3)              79.5%
    ==========================  =======

    So the near-tie regime is empirically the exact-tie regime: at
    ``tau = 1e-9`` essentially every within-tau pair has a gap of zero. See
    ``docs/spec_addenda.md#g22``.

    Args:
        sorted_scores: Scores in non-increasing order.
        limit: How many pairs to return.
        strictly_positive: Skip exact ties. ``False`` finds the exact-tie block,
            which is the case that dominates.

    Returns:
        The closest pairs, tightest first. ``rank`` is 1-indexed, so the pair is
        ``(r_rank, r_{rank+1})`` and ``gap`` is ``m_rank``.
    """
    pairs = [
        NearTie(i + 1, above, below, above - below)
        for i, (above, below) in enumerate(pairwise(sorted_scores))
    ]
    if strictly_positive:
        pairs = [p for p in pairs if p.gap > 0.0]
    pairs.sort(key=lambda p: (p.gap, p.rank))
    return pairs[:limit]
