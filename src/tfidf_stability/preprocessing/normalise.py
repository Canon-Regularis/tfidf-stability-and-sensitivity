"""Text normalisation: the first step of the fixed preprocessing map (section 2).

Every decision here is pinned rather than left to a library default, so the
preprocessing map is reproducible across Python versions and platforms. See
``docs/spec_addenda.md#g7``.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Final, Literal

__all__ = ["NormalisationConfig", "UnicodeForm", "normalise"]

#: The four Unicode normalisation forms. Typed as a Literal so a typo in a config
#: file is a static error rather than a silent runtime fallback.
UnicodeForm = Literal["NFC", "NFD", "NFKC", "NFKD"]

#: Unicode normalisation form. NFKC folds compatibility variants (ligatures,
#: full-width forms, superscripts) onto canonical ones, so visually identical
#: text produces identical tokens. Chosen over NFC because web-scraped corpora
#: routinely mix the two.
_FORM: Final[UnicodeForm] = "NFKC"

#: Characters deleted outright rather than mapped to a separator, since a
#: control character would otherwise collide with the n-gram joiner (ngrams.py).
#: `Cn` is excluded: the set of unassigned code points changes with
#: the Unicode version, so deleting them would tie the token stream to that.
_CONTROL_CATEGORIES: Final = frozenset({"Cc", "Cf", "Co", "Cs"})


@dataclass(frozen=True, slots=True)
class NormalisationConfig:
    """Pinned normalisation options for the preprocessing map.

    ``unicode_form`` is ``"NFKC"`` when normative; ``lowercase`` applies :meth:`str.lower`, as
    :meth:`str.casefold` needs ICU in C++ (spec_addenda G7); ``strip_control`` deletes
    non-whitespace Cc/Cf/Co/Cs characters; ``collapse_whitespace`` collapses and trims whitespace.
    """

    unicode_form: UnicodeForm = _FORM
    lowercase: bool = True
    strip_control: bool = True
    collapse_whitespace: bool = True


_DEFAULT = NormalisationConfig()


def _canonical_case(text: str, cfg: NormalisationConfig) -> str:
    """Normalisation form and case, in the order that makes them commute.

    Normalising first lets case mapping see canonical forms, since some compatibility
    characters case-map differently before and after NFKC. The second normalisation
    is required because ``lower()`` can denormalise characters, breaking idempotence.
    """
    out = unicodedata.normalize(cfg.unicode_form, text)
    # str.lower, not str.casefold. Rationale: docs/engineering_notes.md#lowercase-not-casefold
    if cfg.lowercase:
        out = unicodedata.normalize(cfg.unicode_form, out.lower())
    return out


def normalise(text: str, config: NormalisationConfig | None = None) -> str:
    """Apply the normalisation stage of the preprocessing map.

    Idempotent: ``normalise(normalise(t)) == normalise(t)`` for all ``t`` (property-tested),
    so the map does not depend on how often it is applied. Returns the normalised
    ``text``; ``config`` holds the pinned options and defaults to the normative ones.
    """
    cfg = config or _DEFAULT

    out = _canonical_case(text, cfg)

    if cfg.strip_control:
        stripped = "".join(
            ch for ch in out if ch.isspace() or unicodedata.category(ch) not in _CONTROL_CATEGORIES
        )
        if stripped != out:
            # Deleting a format character can bring a base character and a combining
            # mark into contact ("e", U+200B, U+0301 leaves non-NFKC "e" U+0301), so
            # the stripped text is recomposed. A single recomposition suffices.
            # Rationale: docs/engineering_notes.md#recompose-after-control-strip
            out = _canonical_case(stripped, cfg)

    if cfg.collapse_whitespace:
        out = " ".join(out.split())

    return out
