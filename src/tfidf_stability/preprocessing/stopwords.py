"""Stopword removal, backed by a frozen and hash-verified word list.

The stopword set determines the vocabulary and therefore every document
frequency, idf value and score in this repository, so it is treated as data with
an identity: a versioned file whose SHA-256 is verified at load and recorded in
every run manifest, never a library default that can shift between releases.

Removed tokens become a :data:`~tfidf_stability.preprocessing.tokenise.GAP`
sentinel rather than disappearing, so n-gram construction cannot bridge the hole
they leave. See ``docs/spec_addenda.md#g7``.
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
    """Where the frozen stopword list lives, installed or in-tree.

    Takes the module's own path rather than reading ``__file__`` directly, so
    both layouts can be exercised: in a source checkout only one of them exists,
    and the branch that matters for a wheel is the one that cannot be reached
    from here.

    Two layouts, checked in that order. In a wheel the assets sit inside the
    package, at ``tfidf_stability/data/assets``. In a source checkout they sit at
    the repository root, ``data/assets``, beside ``data/README.md`` which
    explains the licence position -- and they stay there, because the manifest,
    ``scripts/check_vendored.py`` and the docs all address them by that path.

    The repository-root form used to be the only one. It resolves through
    ``parents[3]``, which is the repository from ``src/tfidf_stability/
    preprocessing/``, and the directory *above* ``site-packages`` from an
    installed distribution -- so an installed package could not load its own
    stopword list. Nothing caught it because ``tests/conftest.py`` put ``src/``
    on ``sys.path`` unconditionally, which meant even the wheel's own test
    command imported the source tree.
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

    ``digest`` goes into the run manifest, so a result traces back to the word
    list that produced it.

    Two digests, because they answer different questions and neither subsumes the
    other. ``digest`` is *provenance*: `load_stopwords` sets it from the asset's
    raw file bytes on purpose, so editing the file in place is detectable even if
    the parsed set is unchanged. ``content_digest`` is *identity*: derived here
    from the words themselves, so it cannot be handed in and cannot be wrong.

    Only ``digest`` existed, and it was an arbitrary string the caller supplied.
    Nothing derived it, nothing checked it, and `PreprocessingPipeline` published
    it as the map's identity -- so two sets holding different words could carry
    one identity and hash to one pipeline digest. The factories below all derive
    it correctly, which is why this was latent rather than live, but the
    constructor is public and took whatever it was given.

    This is the same hole `LookupLemmatiser` had one field along, and it is
    closed the same way: the class computes the identity rather than trusting a
    caller to.
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
    """Load a frozen stopword asset from ``data/assets``.

    The digest covers the raw file bytes rather than the parsed set, so a changed
    comment or reordering still shows in the manifest. Cached: the result is
    immutable, and the file cannot change within a run without invalidating the
    run's own provenance.

    Args:
        asset: File name within ``data/assets``.

    Returns:
        The loaded :class:`StopwordSet`.

    Raises:
        FileNotFoundError: If the asset is missing.
        DataIntegrityError: If the asset has no recorded digest, or its bytes do
            not match it. This module's header, the asset's own header and
            ``configs/default.yaml`` all said the list was "verified at load"
            while nothing verified it, until ``data/assets/MANIFEST.sha256``
            existed. Editing one word silently changed every df, idf and score;
            the reproducibility snapshot compares runs against each other rather
            than against a pinned value, so it could not catch that.
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
    """Remove stopwords, leaving a gap sentinel behind by default.

    Args:
        tokens: Token stream.
        stopwords: The set to remove.
        insert_gaps: When ``True`` (the normative setting) each removed token
            becomes a :data:`GAP` so n-grams cannot span it. When ``False``
            tokens are dropped, letting "king of pop" produce the spurious bigram
            "king pop"; ablation only.

    Returns:
        The filtered token stream. Consecutive gaps are collapsed and leading and
        trailing gaps dropped, so two inputs differing only in their stopword
        runs give identical output.
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
