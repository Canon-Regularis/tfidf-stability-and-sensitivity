"""Stopword removal, backed by a frozen and hash-verified word list.

The list ``configs/default.yaml`` names has its SHA-256 verified at load and recorded in every
run manifest. Removed tokens become a :data:`~tfidf_stability.preprocessing.tokenise.GAP`
sentinel so n-grams cannot bridge them. See ``docs/spec_addenda.md#g7``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Final

from tfidf_stability.preprocessing.tokenise import GAP
from tfidf_stability.utils.validation import DataIntegrityError

__all__ = ["StopwordSet", "load_stopwords", "remove_stopwords"]


def _resolve_asset_dir(module: Path) -> Path:
    """Return the stopword asset directory for ``module``, installed or in-tree.

    Checks the wheel layout ``tfidf_stability/data/assets``, then the checkout's ``data/assets``.
    ``data/README.md`` states their licence; ``scripts/check_vendored.py`` verifies their digests.
    Rationale: docs/engineering_notes.md#stopword-asset-layout
    """
    packaged = module.resolve().parents[1] / "data" / "assets"
    if packaged.is_dir():
        return packaged
    return module.resolve().parents[3] / "data" / "assets"


_ASSET_DIR: Final = _resolve_asset_dir(Path(__file__))

#: The recorded digests, in the format ``scripts/check_vendored.py`` parses, so
#: the asset is covered both at load and by the repository-wide gate.
_MANIFEST: Final = _ASSET_DIR / "MANIFEST.sha256"

DEFAULT_STOPWORD_ASSET: Final[str] = "stopwords_en_v1.txt"


def _recorded_digest(asset: str) -> str:
    """The digest this asset is supposed to have, from the manifest."""
    for line in _MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, _, name = line.partition("  ")
        if name.strip() == asset:
            return digest.strip()
    raise DataIntegrityError(f"{asset} has no recorded digest in {_MANIFEST}")


class StopwordSet:
    """An immutable, identified set of stopwords.

    ``digest`` is provenance, from the raw file bytes via `load_stopwords`, for the run manifest.
    ``content_digest`` is identity, derived from the words and never supplied.
    Rationale: docs/engineering_notes.md#stopword-digests
    """

    __slots__ = ("_words", "content_digest", "digest", "name")

    def __init__(self, words: Iterable[str], name: str, digest: str) -> None:
        self._words = frozenset(words)
        self.name = name
        self.digest = digest
        #: SHA-256 over the canonical sorted words, in the form
        #: `from_iterable` digests. Derived, never supplied.
        payload = ("\n".join(sorted(self._words)) + "\n").encode("utf-8")
        self.content_digest = hashlib.sha256(payload).hexdigest()

    def __contains__(self, token: str) -> bool:
        return token in self._words

    def __len__(self) -> int:
        return len(self._words)

    def __iter__(self) -> Iterable[str]:
        return iter(sorted(self._words))

    def __repr__(self) -> str:
        return (
            f"StopwordSet(name={self.name!r}, n={len(self._words)}, digest={self.digest[:12]}...)"
        )

    def is_stopword(self, token: str) -> bool:
        """Whether ``token`` is a stopword. Expects an already-normalised token."""
        return token in self._words

    @classmethod
    def empty(cls) -> StopwordSet:
        """The empty set, for configurations that disable stopword removal."""
        return cls((), name="none", digest=hashlib.sha256(b"").hexdigest())

    @classmethod
    def from_iterable(cls, words: Iterable[str], name: str = "inline") -> StopwordSet:
        """Build from an in-memory iterable, digesting the canonical sorted form."""
        ws = sorted(set(words))
        payload = ("\n".join(ws) + "\n").encode("utf-8")
        return cls(ws, name=name, digest=hashlib.sha256(payload).hexdigest())


@lru_cache(maxsize=8)
def load_stopwords(asset: str = DEFAULT_STOPWORD_ASSET) -> StopwordSet:
    """Load the frozen stopword file ``asset`` from ``data/assets``, verified.

    Returns a cached :class:`StopwordSet` whose digest covers the raw file bytes.
    Raises FileNotFoundError if absent, DataIntegrityError if its digest is unrecorded or wrong.
    Rationale: docs/engineering_notes.md#stopword-load-verification
    """
    path = _ASSET_DIR / asset
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()

    recorded = _recorded_digest(asset)
    if digest != recorded:
        raise DataIntegrityError(
            f"{path} does not match its recorded digest: expected {recorded}, got {digest}. "
            f"The stopword list decides the vocabulary, so this changes every published number."
        )

    words: list[str] = []
    for line in raw.decode("utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            words.append(s)

    return StopwordSet(words, name=asset, digest=digest)


def remove_stopwords(
    tokens: Sequence[str],
    stopwords: StopwordSet,
    *,
    insert_gaps: bool = True,
) -> list[str]:
    """Remove ``stopwords`` from ``tokens``, leaving a gap sentinel by default.

    With ``insert_gaps=True`` (normative) each removed token becomes a :data:`GAP` so
    n-grams cannot span it; ``False`` drops it ("king of pop" gives "king pop"), for ablation.
    Gap runs collapse and edge gaps drop: inputs differing only in stopword runs match.
    """
    out: list[str] = []
    for t in tokens:
        if t == GAP or stopwords.is_stopword(t):
            if insert_gaps and out and out[-1] != GAP:
                out.append(GAP)
        else:
            out.append(t)

    while out and out[-1] == GAP:
        out.pop()
    return out
