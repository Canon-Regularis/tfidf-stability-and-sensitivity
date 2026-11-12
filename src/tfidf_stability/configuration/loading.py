"""Load a run configuration and build the objects it describes.

``load_config`` reads ``configs/default.yaml`` when given no path. The builders
read the ``preprocessing``, ``vocabulary`` and ``numerics`` sections.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from tfidf_stability.preprocessing.lemmatise import LemmatiserKind
from tfidf_stability.preprocessing.normalise import NormalisationConfig
from tfidf_stability.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline
from tfidf_stability.preprocessing.tokenise import TokenisationConfig
from tfidf_stability.utils.hashing import hash_file
from tfidf_stability.utils.numerics import Reduction
from tfidf_stability.utils.paths import resolve_packaged_path
from tfidf_stability.utils.validation import ConfigError
from tfidf_stability.vectorisation.idf import LogImpl
from tfidf_stability.vectorisation.tfidf import TfidfVectoriser
from tfidf_stability.vectorisation.vocabulary import MaxFeaturesPolicy, VocabularyConfig

__all__ = ["load_config", "pipeline_from_config", "vectoriser_from_config"]

_E = TypeVar("_E", bound=Enum)


def _resolve_default_config(module: Path) -> Path:
    """Return the normative configuration path, installed or in-tree.

    The same two layouts as the stopword asset; see :func:`resolve_packaged_path`.
    """
    return resolve_packaged_path(module, "configs", "default.yaml", present=Path.is_file)


_DEFAULT_CONFIG = _resolve_default_config(Path(__file__))


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load a configuration, defaulting to the normative one.

    The returned ``_digest`` identifies the file down to its comments, which
    carry the spec_addenda citations. Every failure raises :class:`ConfigError`: an
    unreadable file, invalid YAML, a non-mapping top level or an unknown section.
    """
    import yaml

    target = Path(path) if path else _DEFAULT_CONFIG
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        # OSError rather than FileNotFoundError: a directory, a permission
        # failure and a missing file are the same mistake to the caller.
        raise ConfigError(f"cannot read config {target}: {exc}") from exc
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{target} is not valid YAML: {exc}") from exc
    if not isinstance(document, dict):
        raise ConfigError(
            f"{target} must hold a mapping at the top level, got {type(document).__name__}"
        )
    parsed: dict[str, Any] = document
    unknown = sorted(set(parsed) - _CONFIG_SECTIONS)
    if unknown:
        # `_section` rejects an unknown key within a section but not an unknown
        # section name. Without this guard a misspelled `preprocesing:` applies
        # nothing while the whole block is hashed into the manifest, so the
        # manifest records a setting the run did not use.
        raise ConfigError(
            f"unrecognised config section(s): {', '.join(unknown)}. "
            f"Known sections: {', '.join(sorted(_CONFIG_SECTIONS))}."
        )
    parsed["_source"] = str(target.name)
    parsed["_digest"] = hash_file(target, text=True)
    return parsed


#: Every top-level section `configs/default.yaml` declares. `ranking`,
#: `evaluation` and `profiles` are recognised but read by no module. They are
#: listed so the file validates; `test_cli.py` asserts the library defaults
#: still agree with the values pinned there.
_CONFIG_SECTIONS = frozenset(
    {"preprocessing", "vocabulary", "numerics", "ranking", "evaluation", "profiles"}
)

#: Every key each section of `configs/default.yaml` declares. An unlisted key is
#: rejected rather than read and ignored, so the manifest's `config` block
#: records what was applied rather than what was on disk.
_PREPROCESSING_KEYS = frozenset(
    {
        "unicode_form",
        "lowercase",
        "token_pattern",
        "min_token_length",
        "max_token_length",
        "stopword_asset",
        "lemmatiser",
        "insert_gaps",
        "n_min",
        "n_max",
        "cross_gaps",
    }
)
_VOCABULARY_KEYS = frozenset({"min_df", "max_df", "max_features", "max_features_policy"})
_NUMERICS_KEYS = frozenset({"reduction", "log_impl"})

_UNICODE_FORMS = frozenset({"NFC", "NFD", "NFKC", "NFKD"})

# Fallbacks come off the dataclasses rather than being restated, so an absent
# key takes the library default.
_NORM_DEFAULTS = NormalisationConfig()
_TOKEN_DEFAULTS = TokenisationConfig()
_PRE_DEFAULTS = PreprocessingConfig()
_VOCAB_DEFAULTS = VocabularyConfig()


def _section(config: dict[str, Any], name: str, allowed: frozenset[str]) -> dict[str, Any]:
    """Return one config section, rejecting any key that would be ignored."""
    # `config.get(name) or {}` would treat any falsy value as absent, so
    # `numerics: 0` and `vocabulary: false` would skip the type check below.
    # An absent or explicitly null section is the empty one; a scalar is an
    # error, whether it is falsy or not.
    section = config.get(name)
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise ConfigError(
            f"config section {name!r} must be a mapping, got {type(section).__name__}"
        )
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ConfigError(
            f"unrecognised key(s) in config section {name!r}: {', '.join(unknown)}. "
            f"Known keys: {', '.join(sorted(allowed))}."
        )
    return section


def _enum(kind: type[_E], value: Any, *, where: str) -> _E:
    """Convert a config string to an enum member, naming the key when it fails."""
    try:
        return kind(value)
    except ValueError as exc:
        options = ", ".join(str(member.value) for member in kind)
        raise ConfigError(f"{where}: {value!r} is not one of {options}") from exc


def pipeline_from_config(config: dict[str, Any]) -> PreprocessingPipeline:
    """Build the preprocessing pipeline the configuration describes."""
    pre = _section(config, "preprocessing", _PREPROCESSING_KEYS)

    form = pre.get("unicode_form", _NORM_DEFAULTS.unicode_form)
    if form not in _UNICODE_FORMS:
        known = ", ".join(sorted(_UNICODE_FORMS))
        raise ConfigError(f"preprocessing.unicode_form: {form!r} is not one of {known}")

    return PreprocessingPipeline(
        PreprocessingConfig(
            normalisation=NormalisationConfig(
                unicode_form=form,
                lowercase=bool(pre.get("lowercase", _NORM_DEFAULTS.lowercase)),
            ),
            tokenisation=TokenisationConfig(
                pattern=str(pre.get("token_pattern", _TOKEN_DEFAULTS.pattern)),
                min_token_length=int(pre.get("min_token_length", _TOKEN_DEFAULTS.min_token_length)),
                max_token_length=int(pre.get("max_token_length", _TOKEN_DEFAULTS.max_token_length)),
            ),
            lemmatiser=_enum(
                LemmatiserKind,
                pre.get("lemmatiser", _PRE_DEFAULTS.lemmatiser),
                where="preprocessing.lemmatiser",
            ),
            stopword_asset=pre.get("stopword_asset", _PRE_DEFAULTS.stopword_asset),
            insert_gaps=bool(pre.get("insert_gaps", _PRE_DEFAULTS.insert_gaps)),
            n_min=int(pre.get("n_min", _PRE_DEFAULTS.n_min)),
            n_max=int(pre.get("n_max", _PRE_DEFAULTS.n_max)),
            cross_gaps=bool(pre.get("cross_gaps", _PRE_DEFAULTS.cross_gaps)),
        )
    )


def vectoriser_from_config(config: dict[str, Any]) -> TfidfVectoriser:
    """Build the vectoriser the configuration describes.

    ``numerics.reduction`` fixes the summation order every norm uses and
    ``numerics.log_impl`` is G13's cross-platform bit-exactness switch. Both reach
    the vectoriser as well as the manifest.
    """
    vocab = _section(config, "vocabulary", _VOCABULARY_KEYS)
    numerics = _section(config, "numerics", _NUMERICS_KEYS)

    return TfidfVectoriser(
        vocabulary_config=VocabularyConfig(
            min_df=vocab.get("min_df", _VOCAB_DEFAULTS.min_df),
            max_df=vocab.get("max_df", _VOCAB_DEFAULTS.max_df),
            max_features=vocab.get("max_features", _VOCAB_DEFAULTS.max_features),
            max_features_policy=_enum(
                MaxFeaturesPolicy,
                vocab.get("max_features_policy", _VOCAB_DEFAULTS.max_features_policy),
                where="vocabulary.max_features_policy",
            ),
        ),
        log_impl=_enum(
            LogImpl, numerics.get("log_impl", LogImpl.CORRECTLY_ROUNDED), where="numerics.log_impl"
        ),
        reduction=_enum(
            Reduction, numerics.get("reduction", Reduction.NAIVE), where="numerics.reduction"
        ),
    )
