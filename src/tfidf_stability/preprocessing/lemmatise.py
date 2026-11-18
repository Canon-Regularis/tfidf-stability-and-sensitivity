"""Lemmatisation (section 2), implemented as interchangeable backends.

The default is Porter2 (Snowball English), in :mod:`tfidf_stability.preprocessing._snowball`.
Porter2 is a stemmer (``docs/spec_addenda.md#g7``); :class:`LookupLemmatiser` gives true lemmas.
Rationale: docs/engineering_notes.md#porter2-default-backend
"""

from __future__ import annotations

import hashlib
import json
import threading
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
    "lemmatiser_identity",
    "make_lemmatiser",
    "porter2_stem",
]


class LemmatiserKind(str, Enum):
    """Which lemmatisation backend to use."""

    NONE = "none"
    PORTER2 = "porter2"
    LOOKUP = "lookup"

    def __str__(self) -> str:
        return self.value


@runtime_checkable
class Lemmatiser(Protocol):
    """The contract every backend satisfies.

    ``name`` is the whole identity for :class:`IdentityLemmatiser` and :class:`Porter2Stemmer`,
    not for :class:`LookupLemmatiser`, which also carries a ``digest`` outside this Protocol.
    Callers read :func:`lemmatiser_identity`, which combines the two.
    """

    #: Which backend this is, recorded in the run manifest. Not necessarily the
    #: whole identity: see the class docstring.
    name: str

    def __call__(self, token: str) -> str:
        """Map a single normalised token to its lemma or stem."""
        ...

    def apply(self, tokens: Sequence[str]) -> list[str]:
        """Map a token stream, preserving gap sentinels."""
        ...


# Porter2 / Snowball English
#: One stemmer per thread. ``BaseStemmer.stemWord`` writes ``current``, ``cursor``, ``limit``,
#: ``limit_backward``, ``bra`` and ``ket``; threads sharing an instance stem spliced words.
#: ``PreprocessingPipeline.preprocess_corpus`` documents this stage as parallelisable.
_LOCAL = threading.local()


def porter2_stem(word: str) -> str:
    """Reduce an English word to its Porter2 (Snowball English) stem.

    Expects an already-normalised, lowercase token. Verified against the official
    ``voc.txt``/``output.txt`` vector pair (42 649 words) by
    ``tests/test_preprocessing_determinism.py``.
    """
    stemmer = getattr(_LOCAL, "stemmer", None)
    if stemmer is None:
        stemmer = EnglishStemmer()
        _LOCAL.stemmer = stemmer
    return str(stemmer.stemWord(word))


# Backends
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

    Output depends on ``table`` and ``fallback`` while ``name`` is always ``"lookup"``, so each
    instance also carries a ``digest``: ``name`` says which backend, ``digest`` which instance.
    See :class:`Lemmatiser` for who reads it.
    """

    name = "lookup"

    def __init__(self, table: dict[str, str], fallback: Lemmatiser | None = None) -> None:
        self._table = table
        self._fallback: Lemmatiser = fallback or IdentityLemmatiser()
        #: SHA-256 over everything that can change this instance's output: the
        #: table in canonical sorted form, and the fallback's own identity.
        #: The fallback matters as much as the table -- it handles every token
        #: the table misses, which on a real corpus is most of them.
        payload = json.dumps(
            {
                "table": dict(sorted(table.items())),
                "fallback": lemmatiser_identity(self._fallback),
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

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


def lemmatiser_identity(lemmatiser: Lemmatiser) -> str:
    """Everything about ``lemmatiser`` that can change a feature stream.

    The bare ``name`` (``"porter2"``) without a ``digest``, ``"lookup:<sha256>"`` with one.
    Provenance records, including :meth:`PreprocessingPipeline.digest`, use this, not ``name``.
    Rationale: docs/engineering_notes.md#lemmatiser-identity
    """
    content = getattr(lemmatiser, "digest", None)
    return lemmatiser.name if content is None else f"{lemmatiser.name}:{content}"
