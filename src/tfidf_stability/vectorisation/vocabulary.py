"""Vocabulary construction (README section 2.1).

Identifiers follow the UTF-8 byte order of the surviving tokens, so every number downstream
depends only on the document multiset. ``max_features`` follows ``docs/spec_addenda.md#g6``
(``TFIDF-SPEC-01``). Rationale: docs/engineering_notes.md#vocabulary-byte-order
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

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class VocabularyConfig:
    """Vocabulary filtering options.

    ``min_df`` (section 2.1) and ``max_df`` are absolute counts as integers and corpus
    proportions in (0, 1] as floats. ``max_features`` caps the vocabulary size. ``None``
    disables ``max_df`` or ``max_features``. ``max_features_policy`` orders the cut (G6).
    """

    min_df: int | float = 1
    max_df: int | float | None = None
    max_features: int | None = None
    max_features_policy: MaxFeaturesPolicy = MaxFeaturesPolicy.DF_DESC

    def __post_init__(self) -> None:
        """Normalise the policy to the enum member, refusing an unknown value.

        ``build_vocabulary`` selects the ranking order with ``is``, and ``"cf_desc"``
        equals the ``str`` enum member without being it, so an uncoerced string ranks by
        ``DF_DESC`` and changes the survivors while the config records the policy requested.
        """
        object.__setattr__(self, "max_features_policy", MaxFeaturesPolicy(self.max_features_policy))


def _resolve_threshold(
    value: int | float, n_docs: int, *, name: str, bound: Literal["lower", "upper"] = "lower"
) -> int:
    """Turn an absolute-or-proportional threshold into an absolute document count.

    Proportions snap to a ``limit_denominator`` fraction, so ``min_df=0.1`` over 30
    documents resolves to 3, not 4. ``bound="lower"`` ceils (``df >= p*n``) and
    ``"upper"`` floors (``df <= p*n``). Rationale: docs/engineering_notes.md#threshold-rounding
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
        """Whether identifiers are in ascending UTF-8 byte order.

        The determinism guarantee rests on this. ``build_vocabulary`` and
        ``load_model`` raise when it is false, and the tests re-check it.
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
    """Build a frozen :class:`Vocabulary` from preprocessed feature streams.

    ``documents`` holds one n-gram sequence per document, as produced by
    :class:`~tfidf_stability.preprocessing.pipeline.PreprocessingPipeline`. Raises
    ``EmptyVocabularyError`` for an empty corpus or when filtering removes every token.
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

    # ``survivors[:-1]`` is a legal slice, so a negative ``max_features`` would drop the
    # lowest-ranked tokens silently. ``-1`` is a plausible typo for "unlimited", which this
    # codebase spells ``null``; ``max_features=0`` fails via ``EmptyVocabularyError``.
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
    # O(|V|) check of the strict UTF-8 byte order that the identifiers, the digest and
    # `load_model` assume. Raised rather than asserted because `python -O` deletes an
    # assert, as with BitIdentityError in tfidf_stability.benchmarks.tfidf_perf.
    if not vocab.is_sorted():
        raise TfidfStabilityError("vocabulary identifiers are not in UTF-8 byte order")
    return vocab
