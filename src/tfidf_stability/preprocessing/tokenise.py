"""Tokenisation: normalised text to a token stream (section 2).

The tokeniser pattern is data: stored in the config, hashed into every run
manifest, and swappable without touching this module. It determines the
vocabulary, and hence every number downstream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

__all__ = ["GAP", "Token", "TokenisationConfig", "tokenise", "tokenise_with_offsets"]

#: Sentinel marking a position where a token was removed (a stopword) or a hard
#: boundary occurred (end of a field or sentence). N-grams must never span one.
#:
#: Without it, removing the stopword from "king of pop" yields the bigram "king
#: pop", a feature that appears in no document and is manufactured by the
#: preprocessing order. See ``docs/spec_addenda.md#g7``.
GAP: Final[str] = "\x00"

#: Unicode word pattern: runs of letters or digits. Apostrophes and hyphens are
#: excluded, so "don't" tokenises as ("don", "t"). Pinned here and hashed into
#: the manifest.
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

    Offsets are retained for README section 1.2: without them a surprising
    vocabulary entry cannot be traced back to the text that produced it.
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
        than downstream, so the bounds are part of the tokenisation contract and
        are hashed with it.

    A length-filtered run leaves no gap sentinel, so an n-gram closes over it:
    ``"king <66 z's> pop"`` and ``"king pop"`` produce identical features,
    including the bigram ``king|pop``. Underscores, commas and any other
    pattern-excluded run behave the same way, since the pattern and the bounds
    together define what a token is. Only stopword removal inserts a gap
    (``spec_addenda.md#g7``), because only it deletes something that was already
    a token. Reachable in configuration: at ``min_token_length: 2``,
    ``"vitamin c deficiency"`` and ``"vitamin deficiency"`` become the same
    document.
    """
    cfg = config or _DEFAULT
    lo, hi = cfg.min_token_length, cfg.max_token_length
    return [
        m.group(0) for m in _compiled(cfg.pattern).finditer(text) if lo <= len(m.group(0)) <= hi
    ]


def tokenise_with_offsets(text: str, config: TokenisationConfig | None = None) -> list[Token]:
    """As :func:`tokenise`, retaining source spans for provenance."""
    cfg = config or _DEFAULT
    lo, hi = cfg.min_token_length, cfg.max_token_length
    return [
        Token(m.group(0), m.start(), m.end())
        for m in _compiled(cfg.pattern).finditer(text)
        if lo <= len(m.group(0)) <= hi
    ]
