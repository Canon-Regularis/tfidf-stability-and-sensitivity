"""Lemmatisation (section 2), implemented as a set of interchangeable backends.

README section 2 requires lemmatisation inside a "fixed, deterministic
preprocessing map". NLTK's WordNet lemmatiser needs a downloaded, versioned
corpus plus a POS tagger; spaCy needs a statistical model whose output moves
between releases. Neither is reproducible, and neither ports to C++.

The default backend is Porter2 (the Snowball English algorithm): a complete
published algorithmic specification with no data files, one upstream source
generating both the Python and the C implementation, and a canonical 42 649-word
test-vector pair from snowballstem.org, so this step is machine-verified.

The implementation is vendored; see
:mod:`tfidf_stability.preprocessing._snowball` for reasoning and provenance.

Porter2 is a stemmer, so it produces "happi" from "happy", which is not a word.
That limitation is recorded in ``docs/spec_addenda.md#g7`` with the proposed
wording change to the paper. :class:`LookupLemmatiser` serves callers who need
true lemmas.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Protocol, runtime_checkable

from tfidf_stability.preprocessing._snowball import EnglishStemmer
from tfidf_stability.preprocessing.tokenise import GAP

__all__ = [
    "IdentityLemmatiser",
    "Lemmatiser",
    "LemmatiserKind",
    "LookupLemmatiser",
    "Porter2Stemmer",
    "make_lemmatiser",
    "porter2_stem",
]


class LemmatiserKind(str, Enum):
    """Which lemmatisation backend to use."""

    NONE = "none"
    PORTER2 = "porter2"
    LOOKUP = "lookup"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@runtime_checkable
class Lemmatiser(Protocol):
    """The contract every backend satisfies."""

    #: Stable identity of this backend, recorded in the run manifest.
    name: str

    def __call__(self, token: str) -> str:
        """Map a single normalised token to its lemma or stem."""
        ...

    def apply(self, tokens: Sequence[str]) -> list[str]:
        """Map a token stream, preserving gap sentinels."""
        ...


# ---------------------------------------------------------------------------
# Porter2 / Snowball English
# ---------------------------------------------------------------------------
_STEMMER = EnglishStemmer()


def porter2_stem(word: str) -> str:
    """Reduce an English word to its Porter2 (Snowball English) stem.

    Expects an already-normalised, lowercase token. Verified against the official
    ``voc.txt``/``output.txt`` vector pair (42 649 words) by
    ``tests/test_preprocessing_determinism.py``.
    """
    return str(_STEMMER.stemWord(word))


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class IdentityLemmatiser:
    """Pass tokens through unchanged. Useful as an ablation baseline."""

    name = "none"

    def __call__(self, token: str) -> str:
        return token

    def apply(self, tokens: Sequence[str]) -> list[str]:
        return list(tokens)


class Porter2Stemmer:
    """The default backend: Snowball English, with a per-token memo."""

    name = "porter2"

    def __init__(self) -> None:
        # Stemming is pure, so the memo cannot change a result. Corpora are
        # Zipfian, so the hit rate is high.
        self._memo: dict[str, str] = {}

    def __call__(self, token: str) -> str:
        cached = self._memo.get(token)
        if cached is None:
            cached = porter2_stem(token)
            self._memo[token] = cached
        return cached

    def apply(self, tokens: Sequence[str]) -> list[str]:
        return [t if t == GAP else self(t) for t in tokens]


class LookupLemmatiser:
    """Table-driven true lemmatisation, falling back to a secondary backend.

    The table comes from a frozen, hash-verified asset, so the mapping is part of
    the run's recorded provenance instead of an ambient property of the machine.
    """

    name = "lookup"

    def __init__(self, table: dict[str, str], fallback: Lemmatiser | None = None) -> None:
        self._table = table
        self._fallback: Lemmatiser = fallback or IdentityLemmatiser()

    def __call__(self, token: str) -> str:
        hit = self._table.get(token)
        return hit if hit is not None else self._fallback(token)

    def apply(self, tokens: Sequence[str]) -> list[str]:
        return [t if t == GAP else self(t) for t in tokens]


def make_lemmatiser(kind: LemmatiserKind | str = LemmatiserKind.PORTER2) -> Lemmatiser:
    """Construct a backend by name."""
    k = LemmatiserKind(kind)
    if k is LemmatiserKind.NONE:
        return IdentityLemmatiser()
    if k is LemmatiserKind.PORTER2:
        return Porter2Stemmer()
    raise ValueError(
        f"the {k.value!r} backend needs an explicit table; construct LookupLemmatiser directly"
    )
