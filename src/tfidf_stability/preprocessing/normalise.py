"""Text normalisation: the first step of the fixed preprocessing map (section 2).

Every decision here is pinned rather than left to a library default, because the
preprocessing map must be reproducible across Python versions, platforms, and the
C++ mirror of this code. See ``docs/spec_addenda.md#g7``.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Final, Literal

__all__ = ["NormalisationConfig", "UnicodeForm", "normalise"]

#: The four Unicode normalisation forms. Typed as a Literal so that a typo in a
#: config file is a static error rather than a silent runtime fallback.
UnicodeForm = Literal["NFC", "NFD", "NFKC", "NFKD"]

#: Unicode normalisation form. NFKC folds compatibility variants (ligatures,
#: full-width forms, superscripts) onto canonical ones, so visually identical
#: text produces identical tokens. Chosen over NFC because corpora scraped from
#: the web routinely mix the two.
_FORM: Final[UnicodeForm] = "NFKC"

#: Characters we delete outright rather than map to a separator. Control
#: characters would otherwise collide with the n-gram joiner (see ngrams.py).
_CONTROL_CATEGORIES: Final = frozenset({"Cc", "Cf", "Co", "Cs"})


@dataclass(frozen=True, slots=True)
class NormalisationConfig:
    """Pinned normalisation options.

    Attributes:
        unicode_form: Always ``"NFKC"`` in the normative configuration.
        lowercase: Apply :meth:`str.lower`. Note this is *not*
            :meth:`str.casefold`: full Unicode case folding cannot be reproduced
            in C++ without ICU, whose tailorings vary by version, so the stricter
            and portable ``lower()`` is used. Documented in spec_addenda G7.
        strip_control: Remove Unicode control and format characters.
        collapse_whitespace: Collapse runs of whitespace to a single space and
            strip the ends, so token boundaries do not depend on incidental
            formatting.
    """

    unicode_form: UnicodeForm = _FORM
    lowercase: bool = True
    strip_control: bool = True
    collapse_whitespace: bool = True


_DEFAULT = NormalisationConfig()


def _canonical_case(text: str, cfg: NormalisationConfig) -> str:
    """Normalisation form and case, applied in the order that makes them commute.

    Normalise first so that case mapping sees canonical forms: some
    compatibility characters case-map differently before and after NFKC. Then
    normalise again, because ``lower()`` can denormalise a handful of characters
    -- that second pass is what makes the idempotence property hold at all.
    """
    out = unicodedata.normalize(cfg.unicode_form, text)
    if cfg.lowercase:
        out = unicodedata.normalize(cfg.unicode_form, out.lower())
    return out


def normalise(text: str, config: NormalisationConfig | None = None) -> str:
    """Apply the normalisation stage of the preprocessing map.

    The operation is **idempotent**: ``normalise(normalise(t)) == normalise(t)``
    for all ``t``. This is asserted as a property test, because a non-idempotent
    normaliser would make the preprocessing map depend on how many times it had
    been applied -- a subtle and very hard-to-find source of irreproducibility.

    Args:
        text: Raw input text.
        config: Pinned options; the normative defaults if omitted.

    Returns:
        The normalised string.
    """
    cfg = config or _DEFAULT

    out = _canonical_case(text, cfg)

    if cfg.strip_control:
        stripped = "".join(
            ch for ch in out if ch.isspace() or unicodedata.category(ch) not in _CONTROL_CATEGORIES
        )
        if stripped != out:
            # Deleting a format character can bring a base character and a
            # combining mark into contact, and the pair then has a composed form
            # that the first pass never saw: "e", U+200B, U+0301 leaves
            # "e" U+0301, which is not NFKC, and which tokenises to "e" where
            # the composed "e-acute" tokenises to itself. Recomposing here is
            # enough and is needed only here: canonical composition merges a
            # starter with a following mark and so can never produce a character
            # this stage would want to delete, and collapsing whitespace below
            # always leaves a space between the characters it separates.
            out = _canonical_case(stripped, cfg)

    if cfg.collapse_whitespace:
        out = " ".join(out.split())

    return out
