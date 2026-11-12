"""CLI command implementations.

Each command wraps a library function. A command that writes a result to disk
writes its run manifest beside it, so no stored result exists without one.
stdlib-only (``argparse``), matching the reference backend.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tfidf_stability.configuration.loading import (
    _DEFAULT_CONFIG,  # noqa: F401 - re-exported for tests/test_packaged_assets.py
    _resolve_default_config,  # noqa: F401 - likewise
    load_config,
    pipeline_from_config,
    vectoriser_from_config,
)
from tfidf_stability.persistence.manifest import (
    RunManifest,
    environment_block,
    is_reproducible_environment,
)
from tfidf_stability.persistence.save_load import load_model, save_model
from tfidf_stability.preprocessing.pipeline import preprocess_records
from tfidf_stability.utils.hashing import hash_bytes, hash_file, short
from tfidf_stability.utils.io import canonical_json, read_jsonl, write_json
from tfidf_stability.utils.logging import EventKind, get_logger, log_event
from tfidf_stability.utils.validation import ConfigError

# `load_config`, `pipeline_from_config` and `vectoriser_from_config` are defined in
# `tfidf_stability.configuration.loading` and are public API of this module as well.
__all__ = [
    "cmd_build_corpus",
    "cmd_info",
    "cmd_inspect",
    "cmd_schema",
    "cmd_verify",
    "load_config",
    "pipeline_from_config",
    "vectoriser_from_config",
    "write_report",
    "_DEFAULT_CONFIG",
    "_resolve_default_config",
]

_LOG = get_logger(__name__)


def _read_corpus(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Read a JSONL corpus of ``{doc_id, text}`` records.

    The whole record is returned rather than the text alone, so a corpus that
    also carries ``fields`` keeps its field boundaries; see
    :func:`~tfidf_stability.preprocessing.pipeline.preprocess_records`.
    """
    ids: list[str] = []
    records: list[dict[str, Any]] = []
    for record in read_jsonl(path):
        ids.append(str(record["doc_id"]))
        records.append(record)
    return ids, records


def cmd_build_corpus(args: argparse.Namespace) -> int:
    """Preprocess a corpus, fit a model, and write it with its manifest."""
    config = load_config(args.config)
    pipeline = pipeline_from_config(config)

    ids, records = _read_corpus(Path(args.corpus))
    features = preprocess_records(pipeline, records)
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
    # Refuse before writing anything. A build refused after `save_model` leaves
    # a complete `.tfsx` and its readable sidecar on disk, with no manifest
    # recording that the build is unfit. `is_reproducible_build` reads only
    # `environment["native"]`, so it needs nothing from `save_model`.
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

    Checks that the file still holds what the manifest records, without the
    original corpus.
    """
    model = load_model(args.model)
    manifest_path = Path(args.model).with_suffix(".manifest.json")
    if not manifest_path.exists():
        print(f"no manifest beside {args.model}")
        return 2

    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = recorded.get("model", {})
    failures = []

    # `container_sha256` first, because it is the only recorded value covering
    # the whole file. `TfidfModel.digest()` hashes the vocabulary, the IDF and
    # every weight, but `reduction`, `doc_ids` and `lengths` round-trip through
    # the container without reaching it.
    actual_container = hash_bytes(Path(args.model).read_bytes())
    checks: tuple[tuple[str, str], ...] = (
        ("container_sha256", actual_container),
        ("model_digest", model.digest()),
        ("vocabulary_digest", model.vocabulary.digest()),
    )

    # A manifest recording none of the three cannot be compared against, so
    # reporting "verified" would be an empty claim. `save_model` writes all
    # three, so a file missing them did not come from this project.
    missing = [key for key, _ in checks if key not in expected]
    if len(missing) == len(checks):
        print(f"the manifest beside {args.model} records no digest to compare against")
        return 2

    for key, actual in checks:
        if key not in expected:
            failures.append(f"  {key}: absent from the manifest, so nothing was compared")
        elif expected[key] != actual:
            failures.append(f"  {key}: manifest {expected[key][:16]}... != actual {actual[:16]}...")

    if failures:
        print("MISMATCH")
        print("\n".join(failures))
        return 1

    # A build with fast-math or architecture tuning may not reproduce published
    # numbers, so report what the manifest recorded. The rule lives in
    # `is_reproducible_environment` so this command and
    # `RunManifest.is_reproducible_build` answer one block the same way.
    environment = recorded.get("environment", {})
    if not isinstance(environment, dict):
        raise ConfigError(
            f"the manifest's environment block is {type(environment).__name__}, "
            f"not a mapping, so the build cannot be described"
        )
    native = environment.get("native")
    reproducible = is_reproducible_environment(environment)

    backend = (
        native["compiler_id"]
        if isinstance(native, dict) and "compiler_id" in native
        else "reference (pure Python)"
    )
    print(f"verified: {model.n_documents} documents, |V| = {model.n_features}")
    print(f"  model digest    : {short(model.digest(), 16)}")
    print(f"  backend         : {backend}")
    print(f"  reproducible    : {reproducible}")
    return 0 if reproducible else 1


def _native_diagnosis() -> list[str]:
    """Why the compiled backend did not load, in the order a reader needs it.

    Extensions are listed as found beside the package and again as filtered by
    :data:`importlib.machinery.EXTENSION_SUFFIXES`, because the usual cause is a
    build tagged for another interpreter, and the two lists differ exactly then.
    """
    import importlib
    import importlib.machinery

    from tfidf_stability import _native

    beside = Path(_native.__file__).parent
    present = sorted(p.name for p in beside.glob("_tfidf_native*") if p.is_file())
    loadable = sorted(
        {
            candidate.name
            for suffix in importlib.machinery.EXTENSION_SUFFIXES
            for candidate in beside.glob(f"_tfidf_native{suffix}")
        }
    )

    lines = [f"the compiled backend did not load: {_native.unavailable_reason() or 'unknown'}"]
    lines += [
        f"{label + ':':<31}{value}"
        for label, value in (
            ("required ABI", _native.REQUIRED_ABI),
            ("extensions beside the package", present or "NONE"),
            ("tagged for this interpreter", loadable or "NONE"),
        )
    ]
    try:
        importlib.import_module("tfidf_stability._native._tfidf_native")
    except Exception as exc:  # the raw error is the whole diagnostic
        lines.append(f"raw import error: {type(exc).__name__}: {exc}")
    return lines


def cmd_info(args: argparse.Namespace) -> int:
    """Report the environment a run would execute in.

    Names the build behind a number without running an experiment.
    ``--require-native`` exits 1 when the extension does not load: every native
    test is guarded on ``native_available()`` and pytest exits 0 when all skip.
    """
    payload: dict[str, Any] = {"environment": environment_block()}
    if args.config:
        payload["config"] = load_config(args.config)

    env = payload["environment"]
    native = env.get("native")
    unmet = getattr(args, "require_native", False) and native is None

    if args.json:
        print(canonical_json(payload))
    else:
        print(f"python     {env['python']} ({env['implementation']})")
        print(f"platform   {env['platform']}")
        if native is None:
            print(
                "native     not built"
                if unmet
                else "native     not built -- the pure-Python reference is normative and complete"
            )
        else:
            print(
                f"native     {native['compiler_id']} {native['compiler_ver']} "
                f"({native['build_type']})"
            )
            print(f"           reproducible = {native['reproducible']}")
            print(f"           flags = {native['numeric_flags']}")
        float_env = env["float"]
        print(
            f"float      mantissa {float_env['mantissa_dig']} bits, "
            f"subnormals {'ok' if float_env['subnormals_supported'] else 'FLUSHED'}"
        )

    if unmet:
        # stderr, so the diagnosis never lands in the JSON on stdout.
        for line in _native_diagnosis():
            print(line, file=sys.stderr)
        return 1
    return 0


def cmd_schema(args: argparse.Namespace) -> int:
    """Print the ``.tfsx`` on-disk schema."""
    from tfidf_stability.persistence.model import (
        BLOCK_SEPARATOR,
        describe_header,
        describe_schema,
    )

    # The header and the separator go out with the arrays. A reader given the
    # arrays alone cannot recover the header's `flags` or `reduction`, nor the
    # byte between the token and document-id blocks, and misparses every
    # container.
    print(
        canonical_json(
            {
                "header": describe_header(),
                "arrays": describe_schema(),
                "block_separator": BLOCK_SEPARATOR.decode("ascii"),
            }
        )
    )
    return 0


def write_report(path: Path | str, payload: Any, manifest: RunManifest) -> None:
    """Write a result document and its manifest side by side."""
    target = Path(path)
    write_json(target, payload)
    manifest.results = {**manifest.results, "report_sha256": hash_file(target, text=True)}
    manifest.write(target.with_suffix(".manifest.json"))
