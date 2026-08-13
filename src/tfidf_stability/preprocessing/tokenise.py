"""Tokenisation: normalised text to a token stream (section 2).

The tokeniser pattern is data, not code: it is stored in the config, hashed into
every run manifest, and can be swapped without touching this module. That matters
because the pattern determines the vocabulary, and hence every number downstream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

__all__ = ["GAP", "Token", "TokenisationConfig", "tokenise", "tokenise_with_offsets"]

#: Sentinel marking a position where a token was removed (a stopword) or a hard
#: boundary occurred (end of a field or sentence). N-grams must never span one.
#:
#: Without this, removing the stopword from "king of pop" would yield the bigram
#: "king pop" -- a feature that appears in no document, manufactured purely by
#: the preprocessing order. See ``docs/spec_addenda.md#g7``.
GAP: Final[str] = "\x00"

#: Unicode word pattern: runs of letters or digits. Apostrophes and hyphens are
#: deliberately *not* included, so "don't" tokenises as ("don", "t"). That is a
#: choice, not an oversight; it is pinned here and hashed into the manifest.
DEFAULT_PATTERN: Final[str] = r"[^\W_]+"

#: ASCII-only alternative, for the restricted profile used in fuzzing where
#: Unicode property tables would make C++/Python agreement depend on ICU.
ASCII_PATTERN: Final[str] = r"[a-z0-9]+"


@dataclass(frozen=True, slots=True)
class TokenisationConfig:
    """Pinned tokenisation options."""

    pattern: str = DEFAULT_PATTERN
    min_token_length: int = 1
    max_token_length: int = 64  # guards against pathological inputs from fuzzing


@dataclass(frozen=True, slots=True)
class Token:
    """A token together with its span in the normalised source text.

    Offsets are retained because README section 1.2 requires intermediate
    quantities to remain inspectable: without them there is no way to trace a
    surprising vocabulary entry back to the text that produced it.
    """

    text: str
    start: int
    end: int


_DEFAULT = TokenisationConfig()
_CACHE: dict[str, re.Pattern[str]] = {}


def _compiled(pattern: str) -> re.Pattern[str]:
    """Compile and memoise. Compilation is pure, so caching cannot affect results."""
    p = _CACHE.get(pattern)
    if p is None:
        p = re.compile(pattern, re.UNICODE)
        _CACHE[pattern] = p
    return p


def tokenise(text: str, config: TokenisationConfig | None = None) -> list[str]:
    """Split normalised text into tokens.

    Args:
        text: Text that has already passed through
            :func:`~tfidf_stability.preprocessing.normalise.normalise`.
        config: Pinned options; the normative defaults if omitted.

    Returns:
        Tokens in order of appearance. Length filters are applied here rather
        than downstream so that the length bounds are part of the tokenisation
        contract and get hashed with it.

    A consequence worth stating, because it surprises: a length-filtered run
    leaves **no gap sentinel**, so an n-gram closes over it. ``"king <66 z's>
    pop"`` and ``"king pop"`` produce identical features, including the bigram
    ``king|pop``. That is deliberate and consistent -- an underscore, a comma
    and any other pattern-excluded run behave the same way, because the pattern
    and the bounds together define what a token *is*. Only stopword removal
    inserts a gap (``spec_addenda.md#g7``), and only because it deletes
    something that was already a token. The collision is reachable in
    configuration rather than merely in theory: at ``min_token_length: 2``,
    ``"vitamin c deficiency"`` and ``"vitamin deficiency"`` become the same
    document.
    """
    cfg = config or _DEFAULT
    lo, hi = cfg.min_token_length, cfg.max_token_length
    return [
        m.group(0) for m in _compiled(cfg.pattern).finditer(text) if lo <= len(m.group(0)) <= hi
    ]


def tokenise_with_offsets(text: str, config: TokenisationConfig | None = None) -> list[Token]:
    """As :func:`tokenise`, but retaining source spans for provenance."""
    cfg = config or _DEFAULT
    lo, hi = cfg.min_token_length, cfg.max_token_length
    return [
        Token(m.group(0), m.start(), m.end())
        for m in _compiled(cfg.pattern).finditer(text)
        if lo <= len(m.group(0)) <= hi
    ]
