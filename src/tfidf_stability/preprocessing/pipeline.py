"""The fixed, deterministic preprocessing map of README section 2.

    raw text -> normalise -> tokenise -> remove stopwords -> lemmatise -> n-grams

Section 2 requires this map to be "deterministic and fixed across all
perturbation experiments". That is a stronger requirement than it appears: it
must be stable not only within a run but across processes, Python versions,
platforms, and the C++ mirror of this code. Everything that could vary is
therefore pinned in :class:`PreprocessingConfig` and folded into
:meth:`PreprocessingConfig.digest`, which every run manifest records.

Order matters and is not arbitrary:

* Stopwords are removed **before** n-gram construction, leaving a gap sentinel,
  so that no n-gram spans a removed token (``docs/spec_addenda.md#g7``).
* Lemmatisation runs **after** stopword removal, so the stopword list is matched
  against surface forms rather than stems -- otherwise "willing" would stem to
  "will" and be silently deleted as a stopword.
* N-grams are built **last**, from lemmatised tokens, so that "running shoes"
  and "run shoe" collapse to the same bigram.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from tfidf_stability.preprocessing.lemmatise import (
    Lemmatiser,
    LemmatiserKind,
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
from tfidf_stability.preprocessing.tokenise import TokenisationConfig, tokenise

__all__ = ["PreprocessedDocument", "PreprocessingConfig", "PreprocessingPipeline"]


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Every knob of the preprocessing map, in one hashable object.

    Attributes:
        normalisation: Unicode and case-folding options.
        tokenisation: The pinned token pattern and length bounds.
        lemmatiser: Which lemmatisation backend to use.
        stopword_asset: File name in ``data/assets``; ``None`` disables removal.
        insert_gaps: Leave a sentinel where a stopword was removed, so n-grams
            cannot bridge it. The normative setting is ``True``.
        n_min: Smallest n-gram order.
        n_max: Largest n-gram order, inclusive.
        cross_gaps: Allow n-grams to span gap sentinels. Ablation only.
    """

    normalisation: NormalisationConfig = field(default_factory=NormalisationConfig)
    tokenisation: TokenisationConfig = field(default_factory=TokenisationConfig)
    lemmatiser: LemmatiserKind = LemmatiserKind.PORTER2
    stopword_asset: str | None = DEFAULT_STOPWORD_ASSET
    insert_gaps: bool = True
    n_min: int = 1
    n_max: int = 2
    cross_gaps: bool = False

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

    def digest(self, stopword_digest: str | None = None) -> str:
        """SHA-256 over the canonical config, optionally binding the word list.

        Passing ``stopword_digest`` makes the identity cover the *contents* of
        the stopword file, not merely its name -- so editing the asset changes
        the config digest and invalidates cached results, as it must.
        """
        payload = self.to_dict()
        if stopword_digest is not None:
            payload["stopword_digest"] = stopword_digest
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def with_(self, **changes: Any) -> PreprocessingConfig:
        """Return a copy with fields replaced -- handy for ablation sweeps."""
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class PreprocessedDocument:
    """A document's token stream plus the intermediates section 1.2 requires."""

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

    The instance is stateless with respect to the documents it processes: calling
    :meth:`preprocess` twice on the same text always yields the same result, and
    processing documents in a different order cannot change any of them. Both
    properties are asserted in ``tests/test_preprocessing_determinism.py``,
    because the determinism guarantee of section 3 depends on them.
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

    # -- identity ------------------------------------------------------------
    @property
    def stopwords(self) -> StopwordSet:
        return self._stopwords

    @property
    def lemmatiser(self) -> Lemmatiser:
        return self._lemmatiser

    def digest(self) -> str:
        """Identity of this exact preprocessing map, for the run manifest."""
        return self.config.digest(stopword_digest=self._stopwords.digest)

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

    # -- the map -------------------------------------------------------------
    def preprocess(self, text: str) -> list[str]:
        """Apply the full map, returning the feature (n-gram) stream."""
        cfg = self.config
        tokens = tokenise(normalise(text, cfg.normalisation), cfg.tokenisation)
        kept = remove_stopwords(tokens, self._stopwords, insert_gaps=cfg.insert_gaps)
        lemmas = self._lemmatiser.apply(kept)
        return generate_ngrams(
            lemmas, cfg.n_min, cfg.n_max, joiner=JOINER, cross_gaps=cfg.cross_gaps
        )

    def preprocess_document(self, doc_id: str, text: str) -> PreprocessedDocument:
        """As :meth:`preprocess`, retaining the intermediate stages for inspection.

        Section 1.2 of the paper requires intermediate quantities to remain
        accessible; this is where that begins.
        """
        cfg = self.config
        raw = tokenise(normalise(text, cfg.normalisation), cfg.tokenisation)
        kept = remove_stopwords(raw, self._stopwords, insert_gaps=cfg.insert_gaps)
        lemmas = self._lemmatiser.apply(kept)
        features = generate_ngrams(
            lemmas, cfg.n_min, cfg.n_max, joiner=JOINER, cross_gaps=cfg.cross_gaps
        )
        return PreprocessedDocument(
            doc_id=doc_id,
            features=tuple(features),
            raw_tokens=tuple(raw),
            lemmas=tuple(lemmas),
        )

    def preprocess_corpus(self, documents: Iterable[tuple[str, str]]) -> list[PreprocessedDocument]:
        """Preprocess ``(doc_id, text)`` pairs, preserving input order.

        Documents are independent: the result for one never depends on the
        others, nor on the order they arrive in. That independence is what lets
        the native backend parallelise this stage without affecting any value.
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
