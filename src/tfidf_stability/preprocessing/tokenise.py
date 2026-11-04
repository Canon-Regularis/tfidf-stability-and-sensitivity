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

#: Sentinel marking a removed token (a stopword) or a hard boundary (end of a
#: field or sentence). Normative n-grams never span one, so removing "of" from
#: "king of pop" does not create the bigram "king pop", found in no document.
#: See ``docs/spec_addenda.md#g7``.
GAP: Final[str] = "\x00"

#: Unicode word pattern: runs of letters or digits, so "don't" tokenises as
#: ("don", "t"). A combining mark is neither, so one that NFKC leaves uncomposed
#: separates tokens: "Istanbul" spelt with U+0130 lowercases to "i" plus U+0307
#: and gives ("i", "stanbul"). See spec_addenda.md#g7. Hashed into the manifest.
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
    """Compile and memoise a pattern.

    Compilation is pure, so caching cannot affect results.
    """
    p = _CACHE.get(pattern)
    if p is None:
        p = re.compile(pattern, re.UNICODE)
        _CACHE[pattern] = p
    return p


def tokenise(text: str, config: TokenisationConfig | None = None) -> list[str]:
    """Split normalised text into tokens in order, applying the length bounds.

    Takes :func:`~tfidf_stability.preprocessing.normalise.normalise` output; ``config`` defaults
    to the normative options. Stopwords leave a gap (``spec_addenda.md#g7``); filtered runs do not.
    Rationale: docs/engineering_notes.md#length-filtered-runs-leave-no-gap
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
