"""The fixed, deterministic preprocessing map of README section 2.

Stages: normalise, tokenise, stopwords (gap sentinels, ``docs/spec_addenda.md#g7``),
lemmatise, n-grams. Every input is pinned in :class:`PreprocessingConfig` and hashed by
:meth:`PreprocessingConfig.digest`. Rationale: docs/engineering_notes.md#preprocessing-order
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from tfidf_stability.preprocessing.lemmatise import (
    Lemmatiser,
    LemmatiserKind,
    lemmatiser_identity,
    make_lemmatiser,
)
from tfidf_stability.preprocessing.ngrams import JOINER, generate_ngrams
from tfidf_stability.preprocessing.normalise import NormalisationConfig, normalise
from tfidf_stability.preprocessing.stopwords import (
    DEFAULT_STOPWORD_ASSET,
    StopwordSet,
    load_stopwords,
    remove_stopwords,
)
from tfidf_stability.preprocessing.tokenise import GAP, TokenisationConfig, tokenise

__all__ = [
    "PreprocessedDocument",
    "PreprocessingConfig",
    "PreprocessingPipeline",
    "preprocess_all",
    "preprocess_records",
]


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Every knob of the preprocessing map, in one hashable object.

    ``insert_gaps`` leaves a sentinel where stopwords are removed, so n-grams cannot bridge
    them (normative ``True``); ``cross_gaps=True`` bridges sentinels (ablation only).
    ``stopword_asset`` is a ``data/assets`` file or ``None`` (no removal); ``n_max`` is inclusive.
    """

    normalisation: NormalisationConfig = field(default_factory=NormalisationConfig)
    tokenisation: TokenisationConfig = field(default_factory=TokenisationConfig)
    lemmatiser: LemmatiserKind = LemmatiserKind.PORTER2
    stopword_asset: str | None = DEFAULT_STOPWORD_ASSET
    insert_gaps: bool = True
    n_min: int = 1
    n_max: int = 2
    cross_gaps: bool = False

    def __post_init__(self) -> None:
        """Normalise ``lemmatiser`` to its member; :func:`make_lemmatiser` selects on it."""
        object.__setattr__(self, "lemmatiser", LemmatiserKind(self.lemmatiser))

    def to_dict(self) -> dict[str, Any]:
        """A canonical, JSON-serialisable view, used for hashing and manifests."""
        return {
            "normalisation": {
                "unicode_form": self.normalisation.unicode_form,
                "lowercase": self.normalisation.lowercase,
                "strip_control": self.normalisation.strip_control,
                "collapse_whitespace": self.normalisation.collapse_whitespace,
            },
            "tokenisation": {
                "pattern": self.tokenisation.pattern,
                "min_token_length": self.tokenisation.min_token_length,
                "max_token_length": self.tokenisation.max_token_length,
            },
            "lemmatiser": str(self.lemmatiser),
            "stopword_asset": self.stopword_asset,
            "insert_gaps": self.insert_gaps,
            "n_min": self.n_min,
            "n_max": self.n_max,
            "cross_gaps": self.cross_gaps,
            "ngram_joiner": JOINER,
        }

    def digest(
        self,
        stopword_digest: str | None = None,
        lemmatiser_override: str | None = None,
    ) -> str:
        """SHA-256 over the canonical config, optionally binding the word list.

        ``stopword_digest`` binds the word-list contents, so editing the asset changes the digest;
        ``lemmatiser_override`` binds a backend injected into :class:`PreprocessingPipeline`
        that bypasses :attr:`lemmatiser`. Each key is omitted when ``None``.
        """
        payload = self.to_dict()
        if stopword_digest is not None:
            payload["stopword_digest"] = stopword_digest
        if lemmatiser_override is not None:
            payload["lemmatiser_override"] = lemmatiser_override
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def with_(self, **changes: Any) -> PreprocessingConfig:
        """Return a copy with fields replaced; used by the ablation sweeps."""
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class PreprocessedDocument:
    """A document's feature stream plus the intermediates section 1.2 requires."""

    doc_id: str
    features: tuple[str, ...]
    #: Tokens after normalisation and tokenisation, before stopword removal.
    raw_tokens: tuple[str, ...]
    #: Tokens after stopword removal and lemmatisation, before n-gram expansion.
    lemmas: tuple[str, ...]

    @property
    def n_features(self) -> int:
        return len(self.features)


class PreprocessingPipeline:
    """A configured, reusable preprocessing map.

    Stateless across documents: repeated :meth:`preprocess` calls agree and document
    order changes no result. ``tests/test_preprocessing_determinism.py`` asserts both;
    the determinism guarantee of section 3 depends on them.
    """

    __slots__ = ("_lemmatiser", "_stopwords", "config")

    def __init__(
        self,
        config: PreprocessingConfig | None = None,
        *,
        lemmatiser: Lemmatiser | None = None,
        stopwords: StopwordSet | None = None,
    ) -> None:
        self.config = config or PreprocessingConfig()

        if stopwords is not None:
            self._stopwords = stopwords
        elif self.config.stopword_asset is None:
            self._stopwords = StopwordSet.empty()
        else:
            self._stopwords = load_stopwords(self.config.stopword_asset)

        self._lemmatiser = lemmatiser or make_lemmatiser(self.config.lemmatiser)

    # --- identity -----------------------------------------------------------
    @property
    def stopwords(self) -> StopwordSet:
        return self._stopwords

    @property
    def lemmatiser(self) -> Lemmatiser:
        return self._lemmatiser

    def digest(self) -> str:
        """Identity of this exact preprocessing map, for the run manifest.

        Binds :func:`~tfidf_stability.preprocessing.lemmatise.lemmatiser_identity` unless it is the
        bare name of the config kind (:class:`IdentityLemmatiser`, :class:`Porter2Stemmer`);
        a :class:`~tfidf_stability.preprocessing.lemmatise.LookupLemmatiser` table is always bound.
        """
        identity = lemmatiser_identity(self._lemmatiser)
        if identity == self._lemmatiser.name and identity == str(self.config.lemmatiser):
            override = None
        else:
            override = identity
        # Both stopword digests, colon-joined as `lemmatiser_identity` joins name and hash:
        # `digest` hashes the raw file bytes (provenance: an in-place edit is detectable) and
        # `content_digest` hashes the words (identity). Neither subsumes the other.
        # Rationale: docs/engineering_notes.md#pipeline-digest-identity
        stopwords = f"{self._stopwords.digest}:{self._stopwords.content_digest}"
        return self.config.digest(stopword_digest=stopwords, lemmatiser_override=override)

    def fingerprint(self) -> dict[str, Any]:
        """Full human-readable provenance of the map."""
        return {
            "digest": self.digest(),
            "config": self.config.to_dict(),
            "stopwords": {
                "name": self._stopwords.name,
                "count": len(self._stopwords),
                "digest": self._stopwords.digest,
            },
            "lemmatiser": self._lemmatiser.name,
        }

    # --- the map ------------------------------------------------------------
    def _tokenise(self, text: str) -> list[str]:
        """Apply :func:`normalise` and tokenise; no gap sentinel exists yet."""
        cfg = self.config
        return tokenise(normalise(text, cfg.normalisation), cfg.tokenisation)

    def _after_tokenisation(self, tokens: Sequence[str]) -> tuple[list[str], list[str]]:
        """Stopwords, lemmatisation and n-grams, as ``(lemmas, features)``."""
        cfg = self.config
        kept = remove_stopwords(tokens, self._stopwords, insert_gaps=cfg.insert_gaps)
        lemmas = self._lemmatiser.apply(kept)
        features = generate_ngrams(
            lemmas, cfg.n_min, cfg.n_max, joiner=JOINER, cross_gaps=cfg.cross_gaps
        )
        return lemmas, features

    def preprocess(self, text: str) -> list[str]:
        """Apply the full map, returning the feature (n-gram) stream."""
        return self._after_tokenisation(self._tokenise(text))[1]

    def preprocess_fields(self, fields: Sequence[str]) -> list[str]:
        """As :meth:`preprocess`, with a hard field boundary unless gaps are ablated.

        Fields are tokenised alone, joined by :data:`~tfidf_stability.preprocessing.tokenise.GAP`
        and split there by :func:`~tfidf_stability.preprocessing.ngrams.generate_ngrams`.
        Rationale: docs/engineering_notes.md#field-boundaries
        """
        stream: list[str] = []
        for text in fields:
            if stream:
                stream.append(GAP)
            stream.extend(self._tokenise(text))
        return self._after_tokenisation(stream)[1]

    def preprocess_document(self, doc_id: str, text: str) -> PreprocessedDocument:
        """As :meth:`preprocess`, retaining the intermediate stages for inspection.

        Section 1.2 of the paper requires intermediate quantities to stay
        accessible; the chain starts here.
        """
        raw = self._tokenise(text)
        lemmas, features = self._after_tokenisation(raw)
        return PreprocessedDocument(
            doc_id=doc_id,
            features=tuple(features),
            raw_tokens=tuple(raw),
            lemmas=tuple(lemmas),
        )

    def preprocess_corpus(self, documents: Iterable[tuple[str, str]]) -> list[PreprocessedDocument]:
        """Preprocess ``(doc_id, text)`` pairs, preserving input order.

        Documents are independent: no result depends on another, or on arrival
        order, which lets the native backend parallelise this stage without
        moving a value.
        """
        return [self.preprocess_document(doc_id, text) for doc_id, text in documents]

    def __repr__(self) -> str:
        return (
            f"PreprocessingPipeline(lemmatiser={self._lemmatiser.name!r}, "
            f"stopwords={self._stopwords.name!r}, "
            f"ngrams=({self.config.n_min},{self.config.n_max}), "
            f"digest={self.digest()[:12]}...)"
        )


def preprocess_all(
    texts: Sequence[str], config: PreprocessingConfig | None = None
) -> list[list[str]]:
    """Convenience wrapper for callers that only need the feature streams."""
    pipeline = PreprocessingPipeline(config)
    return [pipeline.preprocess(t) for t in texts]


def preprocess_records(
    pipeline: PreprocessingPipeline, records: Iterable[Mapping[str, Any]]
) -> list[list[str]]:
    """Feature streams for corpus records, honouring a field boundary if present.

    A record with ``fields`` goes through :meth:`PreprocessingPipeline.preprocess_fields`,
    one with only ``text`` through :meth:`PreprocessingPipeline.preprocess`, so every
    driver reads a corpus, and its field boundaries, the same way.
    """
    return [
        pipeline.preprocess_fields([str(f) for f in record["fields"]])
        if record.get("fields")
        else pipeline.preprocess(str(record["text"]))
        for record in records
    ]
