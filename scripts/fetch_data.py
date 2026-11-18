#!/usr/bin/env python3
"""Download a dataset archive, verify its digest, and parse it.

The only code in the repository that opens a network connection. Without a
pin the script prints the line to paste into the dataset's module.
Rationale: docs/engineering_notes.md#movielens-digest-pin
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tfidf_stability.datasets import beir, movielens  # noqa: E402
from tfidf_stability.utils.validation import DataIntegrityError  # noqa: E402

RAW = REPO / "data" / "raw"
_CHUNK = 1 << 16


@dataclass(frozen=True)
class _Source:
    """Where one dataset comes from, and how its pin is read and reported."""

    url: str
    destination: Path
    pin: Callable[[], str | None]
    pin_line: str
    upstream: Callable[[str], str]
    summary: Callable[[bytes], str]


def _movielens_summary(data: bytes) -> str:
    corpus = movielens.parse_archive(data)
    return (
        f"{corpus.n_documents} documents, {corpus.n_ratings} ratings, "
        f"{corpus.n_users} users, {corpus.n_unrated} unrated"
    )


def _beir_source(variant: str) -> _Source:
    def summary(data: bytes) -> str:
        corpus = beir.parse_archive(data)
        return (
            f"{corpus.n_documents} documents, {len(corpus.queries)} judged queries, "
            f"{len(corpus.qrels)} judgements"
        )

    return _Source(
        url=beir.BEIR_URL.format(variant=variant),
        destination=RAW / f"beir-{variant}.zip",
        pin=lambda: beir.PINNED_SHA256.get(variant),
        pin_line=(
            f'src/tfidf_stability/datasets/beir.py, in PINNED_SHA256:\n    "{variant}": "{{}}",'
        ),
        upstream=lambda url: (
            f"The {variant} archive at {url}\n"
            "no longer matches the pin, so upstream has most likely changed."
        ),
        summary=summary,
    )


#: Each fetchable dataset, by its registered name. Pins are read when the
#: script runs, so a test can replace one.
SOURCES: dict[str, _Source] = {
    "movielens_small": _Source(
        url=movielens.MOVIELENS_URL,
        destination=RAW / "ml-latest-small.zip",
        pin=lambda: movielens.MOVIELENS_SHA256,
        pin_line='src/tfidf_stability/datasets/movielens.py\n    MOVIELENS_SHA256 = "{}"',
        upstream=lambda url: (
            "GroupLens updates ml-latest-small in place, so upstream has most likely\nchanged."
        ),
        summary=_movielens_summary,
    ),
    "beir_scifact": _beir_source("scifact"),
    "beir_nfcorpus": _beir_source("nfcorpus"),
}


def _download(url: str, dest: Path, expected: str | None) -> tuple[str, bool]:
    """Stream to a temporary file, verify the digest, then rename into place.

    Returns the digest and whether the download was placed at ``dest``. An
    interrupted transfer leaves nothing at ``dest``. A mismatched download is
    kept beside ``dest`` with a ``.rejected`` suffix and ``dest`` is untouched.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    digest = hashlib.sha256()

    print(f"downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "tfidf-stability/0.2"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        seen = 0
        while chunk := response.read(_CHUNK):
            digest.update(chunk)
            out.write(chunk)
            seen += len(chunk)
            if total:
                print(f"\r  {seen / 1e6:.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()

    actual = digest.hexdigest()
    if expected is not None and actual != expected:
        rejected = dest.with_suffix(dest.suffix + ".rejected")
        shutil.move(str(partial), str(rejected))
        print(f"  kept the rejected download at {rejected}")
        return actual, False

    shutil.move(str(partial), str(dest))
    return actual, True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=tuple(SOURCES), default="movielens_small")
    parser.add_argument("--output", type=Path, default=None, help="default: under data/raw/")
    parser.add_argument("--url", default=None, help="default: the dataset's published URL")
    parser.add_argument("--force", action="store_true", help="re-download if present")
    parser.add_argument(
        "--print-digest", action="store_true", help="digest an existing file and exit"
    )
    args = parser.parse_args(argv)

    source = SOURCES[args.dataset]
    dest: Path = args.output or source.destination
    url: str = args.url or source.url

    if args.print_digest:
        if not dest.exists():
            print(f"{dest} does not exist", file=sys.stderr)
            return 1
        print(hashlib.sha256(dest.read_bytes()).hexdigest())
        return 0

    # Read before the download, because it decides whether the download is
    # allowed to replace what is already at `dest`.
    pinned = source.pin()

    if dest.exists() and not args.force:
        print(f"{dest} already exists (use --force to re-download)")
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    else:
        had_archive = dest.exists()
        try:
            digest, placed = _download(url, dest, pinned)
        except OSError as exc:
            print(f"download failed: {exc}", file=sys.stderr)
            print(
                "\nIf you are offline or behind a proxy, fetch the archive manually from\n"
                f"  {url}\n"
                f"and place it at {dest}.",
                file=sys.stderr,
            )
            return 1

        if not placed:
            print(f"sha256  {digest}")
            print(
                f"\nDIGEST MISMATCH\n  expected {pinned}\n  actual   {digest}\n\n"
                f"{source.upstream(url)} The download was NOT moved into place.\n"
                + (
                    f"{dest} is untouched and still holds the archive it held before.\n"
                    if had_archive
                    else f"Nothing was written to {dest}.\n"
                )
                + "Published numbers were computed against the pinned archive, so either\n"
                "keep that archive, or re-run the experiments and update the pin in the\n"
                "same commit.",
                file=sys.stderr,
            )
            return 1

    print(f"sha256  {digest}")
    return _check_archive(source, dest, url, digest, pinned)


def _check_archive(source: _Source, dest: Path, url: str, digest: str, pinned: str | None) -> int:
    """Compare ``dest`` with its pin and parse it; 0 when it may be used."""
    # Reached only for a file already at `dest`: a mismatched download returned above.
    if pinned is not None and digest != pinned:
        print(
            f"\nDIGEST MISMATCH\n  expected {pinned}\n  actual   {digest}\n\n"
            f"{dest} was not downloaded again and does not match the pin. It may be\n"
            "truncated, another dataset or an older copy. Published numbers were computed\n"
            f"against the pinned archive; re-run with --force to fetch\n  {url}",
            file=sys.stderr,
        )
        return 1

    # Parsed before any pin is offered, so a corrupt archive fails here rather
    # than mid-experiment, and is never offered as the digest to pin.
    try:
        summary = source.summary(dest.read_bytes())
    except (DataIntegrityError, zipfile.BadZipFile) as exc:
        print(f"{dest} is not a valid archive: {exc}", file=sys.stderr)
        return 1

    if pinned is None:
        print(
            "\nNo digest is pinned yet. To pin this archive, set in\n  "
            + source.pin_line.format(digest)
            + "\nand commit. Every later run then verifies against it."
        )
    else:
        print("digest matches the pin")
    print(
        f"\nparsed: {summary}\nstored at {dest} (gitignored -- do not commit, see data/README.md)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
