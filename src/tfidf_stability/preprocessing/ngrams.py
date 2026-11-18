"""N-gram construction: n-grams are treated as atomic tokens (section 2).

The joiner, ASCII Unit Separator, cannot occur inside a token, so the encoding is injective;
n-grams never span a gap sentinel. Both change the vocabulary; see ``docs/spec_addenda.md#g7``.
Rationale: docs/engineering_notes.md#ngram-joiner-and-gap-boundaries
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Final

from tfidf_stability.preprocessing.tokenise import GAP

__all__ = ["JOINER", "generate_ngrams", "split_ngram"]

#: ASCII Unit Separator (U+001F). Cannot appear inside a token, so the encoding
#: is injective and reversible.
JOINER: Final[str] = "\x1f"


def generate_ngrams(
    tokens: Sequence[str],
    n_min: int = 1,
    n_max: int = 2,
    *,
    joiner: str = JOINER,
    cross_gaps: bool = False,
) -> list[str]:
    """Expand a token stream into n-grams of every order in ``[n_min, n_max]``.

    ``tokens`` may hold :data:`GAP` sentinels; ``cross_gaps=True`` lets n-grams bridge them
    (an ablation, off in the normative configuration). ``joiner`` must not occur in a token.
    Raises ValueError if ``n_min < 1`` or ``n_max < n_min``. Output order is deterministic.
    """
    if n_min < 1:
        raise ValueError(f"n_min must be at least 1, got {n_min}")
    if n_max < n_min:
        raise ValueError(f"n_max ({n_max}) must be at least n_min ({n_min})")

    # Segment on gap sentinels; each segment is n-grammed independently. When
    # bridging is permitted the sentinels are dropped rather than treated as
    # tokens: they are boundary markers, never features in their own right.
    segments: list[Sequence[str]] = (
        [[t for t in tokens if t != GAP]] if cross_gaps else _split_on_gaps(tokens)
    )

    out: list[str] = []
    for segment in segments:
        seg_len = len(segment)
        for n in range(n_min, n_max + 1):
            if n > seg_len:
                break
            if n == 1:
                out.extend(segment)
            else:
                for i in range(seg_len - n + 1):
                    out.append(joiner.join(segment[i : i + n]))
    return out


def _split_on_gaps(tokens: Sequence[str]) -> list[Sequence[str]]:
    """Split a token stream into maximal gap-free runs."""
    segments: list[Sequence[str]] = []
    current: list[str] = []
    for t in tokens:
        if t == GAP:
            if current:
                segments.append(current)
                current = []
        else:
            current.append(t)
    if current:
        segments.append(current)
    return segments


def split_ngram(ngram: str, joiner: str = JOINER) -> list[str]:
    """Recover the constituent tokens of an n-gram.

    Well-defined because the joiner cannot occur inside a token, which is what
    makes the encoding injective. Used by the intermediate-inspection tooling of
    README section 1.2.
    """
    return ngram.split(joiner)


def ngram_order(ngram: str, joiner: str = JOINER) -> int:
    """The order (n) of an n-gram: how many tokens it comprises."""
    return ngram.count(joiner) + 1


def iter_gap_free(tokens: Iterable[str]) -> Iterable[str]:
    """Yield tokens with gap sentinels removed."""
    return (t for t in tokens if t != GAP)
