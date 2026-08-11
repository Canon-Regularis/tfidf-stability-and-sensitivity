"""Lemmatisation (section 2), implemented as a set of interchangeable backends.

README section 2 requires lemmatisation as part of a "fixed, deterministic
preprocessing map". Meeting that literally is harder than it looks: NLTK's
WordNet lemmatiser needs a downloaded, versioned corpus plus a POS tagger, and
spaCy needs a statistical model whose output is not stable across releases.
Neither is acceptable in an artefact whose central claim is reproducibility, and
neither ports to C++.

The default backend is therefore **Porter2** (the Snowball English algorithm),
chosen for three reasons: it is a complete published algorithmic specification
with no data files; the same upstream source generates both the Python and the C
implementation, so the reference and native backends agree by construction; and
snowballstem.org publishes a canonical 42 649-word test-vector pair, so the
preprocessing step is machine-verified rather than merely asserted.

The implementation is vendored rather than hand-written -- see
:mod:`tfidf_stability.preprocessing._snowball` for the reasoning and provenance.

Porter2 is a **stemmer**, not a lemmatiser -- it produces "happi" from "happy",
which is not a word. That is an honest limitation, recorded in
``docs/spec_addenda.md#g7`` along with the proposed wording change to the paper.
:class:`LookupLemmatiser` is provided for callers who need true lemmas.
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

    Expects an already-normalised, lowercase token.

    Verified against the official ``voc.txt``/``output.txt`` vector pair
    (42 649 words) by ``tests/test_preprocessing_determinism.py``.
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
        # Stemming is a pure function, so memoising cannot change any result --
        # it only avoids re-deriving the stem of every repeated token. Corpora
        # are Zipfian, so the hit rate is very high.
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

    The table is loaded from a frozen, hash-verified asset so the mapping is part
    of the run's recorded provenance rather than an ambient property of the
    machine.
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
