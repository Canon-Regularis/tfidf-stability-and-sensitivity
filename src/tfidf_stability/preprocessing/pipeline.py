"""The fixed, deterministic preprocessing map of README section 2.

    raw text -> normalise -> tokenise -> remove stopwords -> lemmatise -> n-grams

Section 2 requires this map to be "deterministic and fixed across all
perturbation experiments", meaning stable across processes, Python versions,
platforms, and the C++ mirror of this code, and not merely within a run.
Everything that could vary is pinned in :class:`PreprocessingConfig` and folded
into :meth:`PreprocessingConfig.digest`, which every run manifest records.

The order is forced:

* Stopwords go before n-gram construction, leaving a gap sentinel, so no n-gram
  spans a removed token (``docs/spec_addenda.md#g7``).
* Lemmatisation runs after stopword removal, so the list is matched against
  surface forms; otherwise "willing" stems to "will" and is silently deleted as
  a stopword.
* N-grams are built last, from lemmatised tokens, so "running shoes" and "run
  shoe" collapse to the same bigram.
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

    def digest(
        self,
        stopword_digest: str | None = None,
        lemmatiser_override: str | None = None,
    ) -> str:
        """SHA-256 over the canonical config, optionally binding the word list.

        ``stopword_digest`` makes the identity cover the contents of the stopword
        file rather than its name alone, so editing the asset changes the config
        digest and invalidates cached results.

        ``lemmatiser_override`` covers the same hole one field along:
        :class:`PreprocessingPipeline` accepts a ready-made lemmatiser that
        bypasses :attr:`lemmatiser` entirely, and without this two pipelines
        producing different features share an identity. Both keys are omitted
        when absent, so a run with no override digests as before.
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

    Stateless with respect to the documents it processes: :meth:`preprocess`
    twice on the same text gives the same result, and document order cannot
    change any result. Both are asserted in
    ``tests/test_preprocessing_determinism.py``; the determinism guarantee of
    section 3 depends on them.
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

        The lemmatiser is bound only when an injected one disagrees with the
        config, the sole way the two can differ. The digest previously read the
        config alone, so ``PreprocessingPipeline(cfg)`` and
        ``PreprocessingPipeline(cfg, lemmatiser=IdentityLemmatiser())``, which
        turn "running cats" into ``run|cat`` and ``running|cats``, hashed to the
        same string. The sibling ``stopwords=`` injection was always bound by
        content. No current caller injects, so no recorded digest changes.

        Bound by :func:`~tfidf_stability.preprocessing.lemmatise.lemmatiser_identity`
        rather than by ``Lemmatiser.name``, because for
        :class:`~tfidf_stability.preprocessing.lemmatise.LookupLemmatiser` the
        name is the constant ``"lookup"`` while the output comes from a table
        handed in at construction. Reading the name bound all such pipelines to
        one digest: three of them producing ``cat|running``, ``feline|running``
        and ``cat|run`` from the same input reported the same identity, which is
        the failure this whole method exists to prevent, one field further in.

        A backend carrying content is bound *unconditionally*, not only when it
        disagrees with the config. The ``==`` short-circuit is sound exactly when
        the name is the whole identity, and it was the second half of the same
        hole: ``lemmatiser=LOOKUP`` in the config plus an injected table matched
        by name and so bound nothing at all.

        Recorded digests are unmoved. The short-circuit still applies to
        :class:`IdentityLemmatiser` and :class:`Porter2Stemmer`, whose output is
        fixed by their class, and no config-only route to a lookup pipeline
        exists: :func:`make_lemmatiser` refuses that kind without a table.
        """
        identity = lemmatiser_identity(self._lemmatiser)
        if identity == self._lemmatiser.name and identity == str(self.config.lemmatiser):
            override = None
        else:
            override = identity
        # Both stopword digests, joined the way `lemmatiser_identity` joins the
        # backend name to its content hash. `digest` is provenance -- the asset's
        # raw file bytes, so an edit in place is detectable even when the parsed
        # set is unchanged -- and `content_digest` is identity, derived from the
        # words and impossible to hand in. Binding provenance alone let a
        # hand-built `StopwordSet` carry any digest string it liked and publish
        # it as this map's identity, so two sets holding different words hashed
        # to one pipeline digest. Neither subsumes the other, so both go in.
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

        Section 1.2 of the paper requires intermediate quantities to stay
        accessible; the chain starts here.
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
