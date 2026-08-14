"""N-gram construction (section 2: "n-grams are treated as atomic tokens").

Two decisions are pinned here, both of which change the vocabulary and therefore
every number downstream. See ``docs/spec_addenda.md#g7``.

The joiner is ASCII Unit Separator. After normalisation and
tokenisation no token can contain a control character, so ``\\x1f`` cannot occur
inside a token. The token-sequence to n-gram encoding is therefore injective:
the bigram ("new", "york") and a hypothetical single token "new york" land on
different vocabulary entries, and an n-gram splits back into its constituents.
A space joiner conflates them.

N-grams never span a gap sentinel. A gap marks a removed stopword or a hard
boundary; bridging one manufactures features that appear in no document ("king
of pop" becoming "king pop"), an artefact of preprocessing order.
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

    Args:
        tokens: Token stream, possibly containing :data:`GAP` sentinels.
        n_min: Smallest n-gram order (1 keeps the unigrams).
        n_max: Largest n-gram order, inclusive.
        joiner: Separator placed between constituents. Do not change without
            re-reading the injectivity argument in this module's docstring.
        cross_gaps: If ``True``, n-grams may bridge a gap sentinel. Off by
            default and in the normative configuration; exposed so the effect can
            be measured as an ablation.

    Returns:
        N-grams in order. Unigrams come from the same pass, so the relative order
        of a document's features is deterministic.

    Raises:
        ValueError: If the order range is invalid.
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
