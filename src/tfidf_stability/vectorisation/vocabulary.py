"""Vocabulary construction (README section 2.1).

This module carries the project's determinism guarantee; a bug here silently
invalidates every published number:

    The vocabulary, the identifier assignment, and therefore every document
    frequency, IDF value, weight, norm, score and ranking are pure functions of
    the *multiset* of preprocessed documents. They are invariant to document
    presentation order, to hash seeds, to dictionary iteration order, to
    ``PYTHONHASHSEED``, and to thread count.

Two phases give that. During accumulation any hash map will do, since nothing
about its iteration order escapes. At freeze time the surviving tokens are sorted
into UTF-8 byte order and identifiers assigned by position, so token to
identifier depends only on the token set.

Byte order in preference to locale collation or Unicode collation, whose
tailorings vary by ICU version and platform. Tokens are NFKC-normalised upstream,
so byte order is well defined on a canonical form and reproducible in C++ via
``memcmp``.

The paper leaves the ``max_features`` truncation rule unspecified; see
``docs/spec_addenda.md#g6`` for the total order adopted here (``TFIDF-SPEC-01``).
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from itertools import pairwise
from typing import Literal

from tfidf_stability.utils.validation import EmptyVocabularyError, TfidfStabilityError

__all__ = [
    "MaxFeaturesPolicy",
    "Vocabulary",
    "VocabularyConfig",
    "build_vocabulary",
]


class MaxFeaturesPolicy(str, Enum):
    """How to choose which tokens survive a ``max_features`` cut."""

    #: TFIDF-SPEC-01: (df desc, cf desc, token bytes asc). The default.
    DF_DESC = "df_desc"
    #: (cf desc, df desc, token bytes asc). Ranks by total occurrences instead.
    CF_DESC = "cf_desc"
    #: scikit-learn's rule, so the differential test against sklearn can run
    #: with a matching criterion.
    SKLEARN_COMPAT = "sklearn_compat"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True, slots=True)
class VocabularyConfig:
    """Vocabulary filtering options.

    Attributes:
        min_df: Minimum document frequency. Integer counts are absolute; a float
            in (0, 1] is a proportion of the corpus. Section 2.1's "minimum
            document-frequency threshold".
        max_df: Maximum document frequency, same convention. ``None`` disables.
            Absent from the paper; standard, and inert when unset.
        max_features: Retain at most this many tokens. ``None`` disables.
        max_features_policy: Which total order decides the cut (see G6).
    """

    min_df: int | float = 1
    max_df: int | float | None = None
    max_features: int | None = None
    max_features_policy: MaxFeaturesPolicy = MaxFeaturesPolicy.DF_DESC


def _resolve_threshold(
    value: int | float, n_docs: int, *, name: str, bound: Literal["lower", "upper"] = "lower"
) -> int:
    """Turn an absolute-or-proportional threshold into an absolute document count.

    Proportions are resolved exactly: this threshold decides vocabulary
    membership, and membership decides every number downstream.

    ``limit_denominator`` does that work. ``Fraction(0.1)`` is the exact binary64
    value ``3602879701896397/36028797018963968``, a shade above one tenth, so
    ``Fraction(0.1) * 30`` exceeds 3 and ceils to 4, a stricter filter than the
    ``0.1`` the caller wrote. Snapping to ``1/10`` first gives 3. Drop the
    ``limit_denominator`` and ``min_df=0.1`` silently changes the vocabulary.

    (An earlier version of this note justified the Fraction by claiming
    ``0.3 * 10`` is ``2.9999999999999996`` and ``0.1 * 30`` is
    ``3.0000000000000004``. Both are ``3.0`` in binary64, so that hazard does not
    exist and the real one went unrecorded.)

    Rounding direction is a parameter because the two ends differ. ``min_df``
    keeps ``df >= p*n`` and rounds up; ``max_df`` keeps ``df <= p*n`` and must
    round down. Rounding an upper bound up admits what the caller asked to
    exclude: at ``p=0.5, n=3`` it keeps a token present in 2 of 3 documents
    (66.7%), and at ``p=0.95, n=7`` it resolves to 7 and filters nothing.
    """
    if isinstance(value, float):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} as a proportion must be in (0, 1], got {value}")
        exact = Fraction(value).limit_denominator(1_000_000) * n_docs
        if bound == "upper":
            return exact.numerator // exact.denominator  # floor
        return max(1, -(-exact.numerator // exact.denominator))  # ceiling
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return int(value)


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """A frozen, lexicographically ordered vocabulary.

    ``tokens[i]`` is the token with identifier ``i``, and ``tokens`` is sorted in
    ascending UTF-8 byte order, so identifiers are a deterministic function of
    the token set alone.
    """

    tokens: tuple[str, ...]
    #: Document frequency of each token, indexed by identifier.
    df: tuple[int, ...]
    #: Collection frequency (total occurrences), indexed by identifier.
    cf: tuple[int, ...]
    #: Number of documents the vocabulary was built from: N in section 2.1.
    n_documents: int
    #: Tokens seen during accumulation but discarded by filtering.
    n_discarded: int
    _index: Mapping[str, int]

    def __len__(self) -> int:
        return len(self.tokens)

    def __contains__(self, token: str) -> bool:
        return token in self._index

    def id_of(self, token: str) -> int | None:
        """Identifier of ``token``, or ``None`` if out of vocabulary."""
        return self._index.get(token)

    def token_of(self, term_id: int) -> str:
        """The token with the given identifier."""
        return self.tokens[term_id]

    def df_of(self, token: str) -> int:
        """Document frequency of ``token``; 0 if out of vocabulary."""
        i = self._index.get(token)
        return 0 if i is None else self.df[i]

    def digest(self) -> str:
        """SHA-256 over the canonical ``token\\tdf\\tcf`` listing.

        Recorded in every run manifest. The listing is in identifier order and
        identifiers are byte-sorted, so the digest is a complete,
        order-independent fingerprint of the vocabulary.
        """
        h = hashlib.sha256()
        h.update(f"N={self.n_documents}\n".encode())
        for t, d, c in zip(self.tokens, self.df, self.cf, strict=True):
            h.update(f"{t}\t{d}\t{c}\n".encode())
        return h.hexdigest()

    def is_sorted(self) -> bool:
        """Whether identifiers really are in ascending UTF-8 byte order.

        The determinism guarantee rests on this, so it is asserted at
        construction and re-checked by the reproducibility tests.
        """
        encoded = [t.encode("utf-8") for t in self.tokens]
        return all(a < b for a, b in pairwise(encoded))


def _byte_key(token: str) -> bytes:
    """Sort key giving UTF-8 byte order.

    Python's default string ordering is by code point, which for UTF-8 coincides
    with byte order; encoding explicitly states the intent and guarantees the C++
    ``memcmp`` implementation agrees.
    """
    return token.encode("utf-8")


def build_vocabulary(
    documents: Iterable[Sequence[str]],
    config: VocabularyConfig | None = None,
) -> Vocabulary:
    """Build a frozen vocabulary from preprocessed feature streams.

    Args:
        documents: One feature (n-gram) sequence per document, as produced by
            :class:`~tfidf_stability.preprocessing.pipeline.PreprocessingPipeline`.
        config: Filtering options; permissive defaults if omitted.

    Returns:
        The frozen :class:`Vocabulary`.

    Raises:
        EmptyVocabularyError: If filtering removes every token. Treated as a
            configuration error: it almost always means ``min_df`` is too high
            for the corpus size.
    """
    cfg = config or VocabularyConfig()

    # --- accumulate ---------------------------------------------------------
    # Iteration order of these counters never escapes: everything that leaves
    # this function is byte-sorted below.
    df_counter: Counter[str] = Counter()
    cf_counter: Counter[str] = Counter()
    n_docs = 0
    for features in documents:
        n_docs += 1
        cf_counter.update(features)
        df_counter.update(set(features))

    if n_docs == 0:
        raise EmptyVocabularyError("cannot build a vocabulary from an empty corpus")

    n_seen = len(df_counter)

    # ``survivors[:-1]`` is a legal slice, so ``max_features=-1`` quietly dropped
    # the lowest-ranked token instead of raising. ``-1`` is the plausible typo
    # for "unlimited", which this codebase spells ``null``. ``max_features=0``
    # already failed via ``EmptyVocabularyError``.
    if cfg.max_features is not None and cfg.max_features < 0:
        raise ValueError(f"max_features must be non-negative or None, got {cfg.max_features}")

    # --- filter by document frequency ---------------------------------------
    min_df = _resolve_threshold(cfg.min_df, n_docs, name="min_df")
    max_df = (
        n_docs
        if cfg.max_df is None
        else _resolve_threshold(cfg.max_df, n_docs, name="max_df", bound="upper")
    )
    survivors = [t for t, d in df_counter.items() if min_df <= d <= max_df]

    # --- apply max_features (TFIDF-SPEC-01, spec_addenda G6) ----------------
    if cfg.max_features is not None and len(survivors) > cfg.max_features:
        policy = cfg.max_features_policy

        # All three keys share one shape so the sort is a single total order.
        # sklearn ranks by collection frequency alone and breaks ties by term;
        # the constant middle component leaves that ordering unchanged.
        def rank(t: str) -> tuple[int, int, bytes]:
            if policy is MaxFeaturesPolicy.CF_DESC:
                return (-cf_counter[t], -df_counter[t], _byte_key(t))
            if policy is MaxFeaturesPolicy.SKLEARN_COMPAT:
                return (-cf_counter[t], 0, _byte_key(t))
            return (-df_counter[t], -cf_counter[t], _byte_key(t))

        survivors.sort(key=rank)
        survivors = survivors[: cfg.max_features]

    if not survivors:
        raise EmptyVocabularyError(
            f"no token survived filtering (min_df={min_df}, max_df={max_df}, "
            f"max_features={cfg.max_features}) over {n_docs} documents "
            f"with {n_seen} distinct features"
        )

    # --- freeze: byte order fixes the identifiers ---------------------------
    survivors.sort(key=_byte_key)
    tokens = tuple(survivors)
    index = {t: i for i, t in enumerate(tokens)}

    vocab = Vocabulary(
        tokens=tokens,
        df=tuple(df_counter[t] for t in tokens),
        cf=tuple(cf_counter[t] for t in tokens),
        n_documents=n_docs,
        n_discarded=n_seen - len(tokens),
        _index=index,
    )
    # O(|V|), and it is the invariant everything downstream assumes: the binary
    # searches in tf.py and the merge in align_models are only correct on a sorted
    # vocabulary. Raised rather than asserted because `python -O` deletes an
    # assert, and this is the one check standing between a mis-sorted vocabulary
    # and silently wrong weights. Same reasoning as BitIdentityError in
    # benchmarks/tfidf_perf.py: a finding about correctness must survive -O.
    if not vocab.is_sorted():
        raise TfidfStabilityError("vocabulary identifiers are not in UTF-8 byte order")
    return vocab
