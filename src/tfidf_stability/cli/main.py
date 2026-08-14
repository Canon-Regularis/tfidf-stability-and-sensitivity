"""``tfidf-stability`` command-line entry point.

``argparse`` rather than a CLI framework: the normative backend is
standard-library only, and a dependency for a handful of subcommands would
undermine that.

Every result-producing command writes a run manifest beside its output, so no
flag can be forgotten into an unrecorded result.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from tfidf_stability.cli.commands import (
    cmd_build_corpus,
    cmd_info,
    cmd_inspect,
    cmd_schema,
    cmd_verify,
)
from tfidf_stability.utils.logging import configure

__all__ = ["build_parser", "main"]

#: Accepted ``--log-level`` values, lowercased for the command line.
_LOG_LEVELS = ("debug", "info", "warning", "error")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="tfidf-stability",
        description=(
            "Numerical stability and perturbation behaviour in TF-IDF similarity "
            "systems. Every result-producing command writes a run manifest beside "
            "its output."
        ),
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    parser.add_argument(
        "--log-level",
        default=None,
        choices=_LOG_LEVELS,
        help="emit provenance events on stderr at this level (default: no logging)",
    )
    parser.add_argument(
        "--log-timestamps",
        action="store_true",
        help="prefix log lines with wall-clock time; makes the output non-reproducible",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    build = sub.add_parser(
        "build-corpus", help="preprocess a JSONL corpus, fit a model, write it with a manifest"
    )
    build.add_argument("corpus", help="JSONL file of {doc_id, text} records")
    build.add_argument("-o", "--output", required=True, help="destination .tfsx path")
    build.add_argument(
        "-c",
        "--config",
        default=None,
        help="config YAML (default: configs/default.yaml)",
    )
    build.set_defaults(func=cmd_build_corpus)

    inspect = sub.add_parser(
        "inspect", help="print one document's intermediate quantities (README section 1.2)"
    )
    inspect.add_argument("model", help="a .tfsx file")
    inspect.add_argument("doc_id", help="document identifier")
    inspect.set_defaults(func=cmd_inspect)

    verify = sub.add_parser(
        "verify", help="re-derive a model's digests and compare against its manifest"
    )
    verify.add_argument("model", help="a .tfsx file")
    verify.set_defaults(func=cmd_verify)

    info = sub.add_parser("info", help="report the build and floating-point environment")
    info.add_argument("-c", "--config", default=None, help="also resolve and print a config")
    info.add_argument("--json", action="store_true", help="machine-readable output")
    info.set_defaults(func=cmd_info)

    schema = sub.add_parser("schema", help="print the .tfsx on-disk schema")
    schema.set_defaults(func=cmd_schema)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Only on request: an unconfigured package logger leaves the logging system
    # as the caller had it, so this entry point is safe to call from a test or a
    # notebook as well as from a shell.
    if getattr(args, "log_level", None):
        configure(
            level=args.log_level.upper(),
            timestamps=getattr(args, "log_timestamps", False),
        )
        from tfidf_stability._native import log_backend_selection

        log_backend_selection()

    if getattr(args, "version", False):
        from tfidf_stability import __version__

        print(__version__)
        return 0

    if not getattr(args, "func", None):
        parser.print_help()
        return 1

    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
