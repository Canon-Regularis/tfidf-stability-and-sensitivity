"""CLI command implementations.

Each command is a thin wrapper over a library function that emits a run
manifest. The manifest comes as a side effect of every command rather than on
request, so an undocumented result cannot be produced by accident.

stdlib-only (``argparse``), matching the reference backend.
"""

from __future__ import annotations

import argparse
import json
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from tfidf_stability.persistence.manifest import RunManifest, environment_block
from tfidf_stability.persistence.save_load import load_model, save_model
from tfidf_stability.preprocessing.lemmatise import LemmatiserKind
from tfidf_stability.preprocessing.normalise import NormalisationConfig
from tfidf_stability.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline
from tfidf_stability.preprocessing.tokenise import TokenisationConfig
from tfidf_stability.utils.hashing import hash_file, short
from tfidf_stability.utils.io import canonical_json, read_jsonl, write_json
from tfidf_stability.utils.logging import EventKind, get_logger, log_event
from tfidf_stability.utils.numerics import Reduction
from tfidf_stability.utils.validation import ConfigError
from tfidf_stability.vectorisation.idf import LogImpl
from tfidf_stability.vectorisation.tfidf import TfidfVectoriser
from tfidf_stability.vectorisation.vocabulary import MaxFeaturesPolicy, VocabularyConfig

__all__ = [
    "cmd_build_corpus",
    "cmd_info",
    "cmd_inspect",
    "cmd_verify",
    "load_config",
    "pipeline_from_config",
    "vectoriser_from_config",
]

_LOG = get_logger(__name__)

_E = TypeVar("_E", bound=Enum)

_REPO = Path(__file__).resolve().parents[3]


def _resolve_default_config(module: Path) -> Path:
    """The normative configuration, installed or in-tree.

    Same two layouts as the stopword asset, and the same reason: resolving only
    through ``parents[3]`` finds the repository from a checkout and the
    directory above ``site-packages`` from a wheel, so ``tfidf-stability
    build-corpus`` could not find its own default config once installed.
    """
    packaged = module.resolve().parents[1] / "configs" / "default.yaml"
    if packaged.is_file():
        return packaged
    return module.resolve().parents[3] / "configs" / "default.yaml"


_DEFAULT_CONFIG = _resolve_default_config(Path(__file__))


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load a configuration, defaulting to the normative one.

    The digest travels with the parsed content, so a manifest identifies the
    config down to its comments, which carry the spec_addenda citations.
    """
    import yaml

    target = Path(path) if path else _DEFAULT_CONFIG
    parsed: dict[str, Any] = yaml.safe_load(target.read_text(encoding="utf-8"))
    parsed["_source"] = str(target.name)
    parsed["_digest"] = hash_file(target, text=True)
    return parsed


# Every key ``configs/default.yaml`` declares, per section. A key listed here is
# honoured, anything else rejected. Unlisted keys used to be read and ignored,
# which made the manifest's ``config`` block a record of what was on disk rather
# than of what was applied.
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
# key behaves as it did before the config was wired up.
_NORM_DEFAULTS = NormalisationConfig()
_TOKEN_DEFAULTS = TokenisationConfig()
_PRE_DEFAULTS = PreprocessingConfig()
_VOCAB_DEFAULTS = VocabularyConfig()


def _section(config: dict[str, Any], name: str, allowed: frozenset[str]) -> dict[str, Any]:
    """Return one config section, rejecting any key that would be ignored."""
    section = config.get(name) or {}
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
    ``numerics.log_impl`` is G13's cross-platform bit-exactness switch. Both were
    once read from the file, hashed into the manifest, and then ignored.
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


def _read_corpus(path: Path) -> tuple[list[str], list[str]]:
    """Read a JSONL corpus of ``{doc_id, text}`` records."""
    ids: list[str] = []
    texts: list[str] = []
    for record in read_jsonl(path):
        ids.append(str(record["doc_id"]))
        texts.append(str(record["text"]))
    return ids, texts


def cmd_build_corpus(args: argparse.Namespace) -> int:
    """Preprocess a corpus, fit a model, and write it with its manifest."""
    config = load_config(args.config)
    pipeline = pipeline_from_config(config)

    ids, texts = _read_corpus(Path(args.corpus))
    features = [pipeline.preprocess(t) for t in texts]
    model = vectoriser_from_config(config).fit(features, ids)

    log_event(_LOG, EventKind.REDUCTION_POLICY, stage="vectorisation", policy=model.reduction)
    if model.zero_norm_documents:
        log_event(
            _LOG,
            EventKind.DEGENERATE,
            case="zero_norm_document",
            n=len(model.zero_norm_documents),
            n_documents=model.n_documents,
        )

    out = Path(args.output)

    manifest = RunManifest(
        run_kind="build_corpus",
        config=config,
        dataset={
            "path": Path(args.corpus).name,
            "sha256": hash_file(args.corpus, text=True),
            "n_documents": len(ids),
        },
        preprocessing=pipeline.fingerprint(),
        parameters={"reduction": str(model.reduction), "log_impl": str(model.idf.log_impl)},
    )
    # Refuse BEFORE writing anything. `save_model` used to run first and this
    # guard second, so on a fast-math or arch-tuned build the container and its
    # readable sidecar were already on disk when `require_reproducible` raised
    # and `manifest.write` was never reached. What that left behind is the worst
    # of the three possible states: a complete-looking `.tfsx` with a sidecar
    # full of digests, from a build this project declares unfit to produce
    # publishable numbers, and no manifest to say so. `is_reproducible_build`
    # reads only `environment["native"]`, so it needs nothing from `save_model`
    # and can be asked first.
    manifest.require_reproducible()

    provenance = save_model(model, out)
    manifest.model = provenance
    manifest.write(out.with_suffix(".manifest.json"))

    # Digests only, never the output path: the path is the caller's choice and
    # would put a machine-specific value into a reproducible record.
    for artefact, sha256 in (
        ("model", model.digest()),
        ("vocabulary", model.vocabulary.digest()),
        ("manifest", manifest.digest()),
    ):
        log_event(_LOG, EventKind.DIGEST, artefact=artefact, sha256=sha256)

    print(
        f"fitted {model.n_documents} documents, |V| = {model.n_features}, nnz = {model.matrix.nnz}"
    )
    print(f"  zero-norm documents : {len(model.zero_norm_documents)}")
    print(f"  model digest        : {short(model.digest(), 16)}")
    print(f"  manifest digest     : {short(manifest.digest(), 16)}")
    print(f"  written             : {out}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Print a document's intermediate quantities (README section 1.2).

    Section 1.2 requires the intermediates stay inspectable; this reaches them
    from a shell without a Python session.
    """
    model = load_model(args.model)
    if args.doc_id not in model.doc_ids:
        print(f"no document {args.doc_id!r}; corpus has {model.n_documents}")
        return 2
    print(canonical_json(model.intermediates(model.doc_ids.index(args.doc_id))))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Re-derive a saved model's digests and compare against its manifest.

    Does this file still contain what the manifest says it does? Answered
    without the original corpus.
    """
    model = load_model(args.model)
    manifest_path = Path(args.model).with_suffix(".manifest.json")
    if not manifest_path.exists():
        print(f"no manifest beside {args.model}")
        return 2

    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = recorded.get("model", {})
    failures = []
    for key, actual in (
        ("model_digest", model.digest()),
        ("vocabulary_digest", model.vocabulary.digest()),
    ):
        if key in expected and expected[key] != actual:
            failures.append(f"  {key}: manifest {expected[key][:16]}... != actual {actual[:16]}...")

    if failures:
        print("MISMATCH")
        print("\n".join(failures))
        return 1

    # A build with fast-math or architecture tuning may not reproduce published
    # numbers, so report what the manifest recorded. No native block means the
    # pure-Python reference produced this, reproducible on its own.
    native = recorded.get("environment", {}).get("native")
    reproducible = True if native is None else bool(native.get("reproducible", False))

    backend = "reference (pure Python)" if native is None else native["compiler_id"]
    print(f"verified: {model.n_documents} documents, |V| = {model.n_features}")
    print(f"  model digest    : {short(model.digest(), 16)}")
    print(f"  backend         : {backend}")
    print(f"  reproducible    : {reproducible}")
    return 0 if reproducible else 1


def cmd_info(args: argparse.Namespace) -> int:
    """Report the environment a run would execute in.

    The first question about a surprising number is which build produced it.
    This answers it without running an experiment.
    """
    payload: dict[str, Any] = {"environment": environment_block()}
    if args.config:
        payload["config"] = load_config(args.config)
    if args.json:
        print(canonical_json(payload))
        return 0

    env = payload["environment"]
    print(f"python     {env['python']} ({env['implementation']})")
    print(f"platform   {env['platform']}")
    native = env.get("native")
    if native is None:
        print("native     not built -- the pure-Python reference is normative and complete")
    else:
        print(
            f"native     {native['compiler_id']} {native['compiler_ver']} ({native['build_type']})"
        )
        print(f"           reproducible = {native['reproducible']}")
        print(f"           flags = {native['numeric_flags']}")
    float_env = env["float"]
    print(
        f"float      mantissa {float_env['mantissa_dig']} bits, "
        f"subnormals {'ok' if float_env['subnormals_supported'] else 'FLUSHED'}"
    )
    return 0


def cmd_schema(args: argparse.Namespace) -> int:
    """Print the ``.tfsx`` on-disk schema."""
    from tfidf_stability.persistence.model import describe_schema

    print(canonical_json(describe_schema()))
    return 0


def write_report(path: Path | str, payload: Any, manifest: RunManifest) -> None:
    """Write a result document and its manifest side by side."""
    target = Path(path)
    write_json(target, payload)
    manifest.results = {**manifest.results, "report_sha256": hash_file(target, text=True)}
    manifest.write(target.with_suffix(".manifest.json"))
