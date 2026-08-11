#!/usr/bin/env python3
"""Download MovieLens, verify it, and record the digest.

The only code in this repository that opens a network connection. Kept in one
short script, out of the library, so that "does this package phone home?" is
answerable by reading one file -- and so the library itself stays usable in an
air-gapped or reviewer-sandboxed environment.

What it does not do
-------------------
It does not put the archive anywhere git tracks. ``data/raw/`` is gitignored, and
this script refuses to write outside it without ``--output``. The MovieLens
licence prohibits redistribution, and an accidental ``git add -A`` is exactly how
that gets violated.

The pin
-------
GroupLens replaces ``ml-latest-small.zip`` **in place**. So on first run there is
no digest to check against; the script downloads, prints the digest, and tells
you to paste it into ``movielens.MOVIELENS_SHA256``. On every subsequent run the
digest is checked and a mismatch is fatal.

That order matters: pinning *after* the fact is the only honest option when
upstream publishes no digest of its own. What the pin buys is not authenticity
but *stability* -- a guarantee that the corpus underneath a published number has
not moved.

Usage::

    python scripts/fetch_data.py                    # download and verify
    python scripts/fetch_data.py --print-digest     # digest an existing file
    python scripts/fetch_data.py --output PATH      # somewhere else
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tfidf_stability.datasets import movielens  # noqa: E402

DEFAULT_DEST = REPO / "data" / "raw" / "ml-latest-small.zip"
_CHUNK = 1 << 16


def _download(url: str, dest: Path) -> str:
    """Stream to a temporary file, then rename. Returns the digest.

    Streamed rather than read whole so an interrupted transfer cannot leave a
    truncated file at the destination path -- which would then fail the digest
    check confusingly, rather than simply being absent.
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

    shutil.move(str(partial), str(dest))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--url", default=movielens.MOVIELENS_URL)
    parser.add_argument("--force", action="store_true", help="re-download if present")
    parser.add_argument(
        "--print-digest", action="store_true", help="digest an existing file and exit"
    )
    args = parser.parse_args()

    dest: Path = args.output

    if args.print_digest:
        if not dest.exists():
            print(f"{dest} does not exist", file=sys.stderr)
            return 1
        print(hashlib.sha256(dest.read_bytes()).hexdigest())
        return 0

    if dest.exists() and not args.force:
        print(f"{dest} already exists (use --force to re-download)")
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    else:
        try:
            digest = _download(args.url, dest)
        except OSError as exc:
            print(f"download failed: {exc}", file=sys.stderr)
            print(
                "\nIf you are offline or behind a proxy, fetch the archive manually from\n"
                f"  {args.url}\n"
                f"and place it at {dest}.",
                file=sys.stderr,
            )
            return 1

    print(f"sha256  {digest}")

    pinned = movielens.MOVIELENS_SHA256
    if pinned is None:
        print(
            "\nNo digest is pinned yet. To pin this archive, set in\n"
            "  src/tfidf_stability/datasets/movielens.py\n"
            f'    MOVIELENS_SHA256 = "{digest}"\n'
            "and commit. Every later run then verifies against it."
        )
    elif digest != pinned:
        print(
            f"\nDIGEST MISMATCH\n  expected {pinned}\n  actual   {digest}\n\n"
            "GroupLens updates ml-latest-small in place, so upstream has most likely\n"
            "changed. This is not something to paper over: published numbers were\n"
            "computed against the pinned archive. Either obtain that archive, or\n"
            "re-run the experiments and update the pin in the same commit.",
            file=sys.stderr,
        )
        return 1
    else:
        print("digest matches the pin")

    # Parse it, so a corrupt-but-correctly-sized archive fails here rather than
    # in the middle of an experiment.
    corpus = movielens.parse_archive(dest.read_bytes())
    print(
        f"\nparsed: {corpus.n_documents} documents, {corpus.n_ratings} ratings, "
        f"{corpus.n_users} users, {corpus.n_unrated} unrated\n"
        f"stored at {dest} (gitignored -- do not commit, see data/README.md)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
