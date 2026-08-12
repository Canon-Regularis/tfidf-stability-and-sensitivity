"""MovieLens ``ml-latest-small`` to a corpus, without ever redistributing it.

The licence position, which shapes the whole module
---------------------------------------------------
The MovieLens usage licence permits research use and **prohibits
redistribution**. So the data cannot be committed, cannot be an test fixture, and
cannot be in CI. What *is* committed is this parser, a pinned SHA-256, and the
derived statistics -- everything needed to reproduce the numbers given a copy of
the archive obtained from GroupLens directly.

GroupLens updates ``ml-latest-small`` **in place**, at the same URL. A pin is
therefore not paranoia: without it the corpus underneath a published result can
change silently. :func:`load` verifies the digest and refuses to proceed on a
mismatch rather than warning, because a warning would be filtered out by the very
scripts that need to see it.

Determinism on real data
------------------------
Three specific hazards, all handled here rather than downstream:

1. **Row order.** CSV row order is an artefact of how GroupLens exported the
   file. Everything is sorted by ``movieId`` as an integer, so document indices
   are a property of the data rather than of the file.
2. **Ratings are 0.5-quantised**, so ``2 * rating`` is always an exact integer.
   That is what makes ``docs/spec_addenda.md#g8``'s exact ``(2*sum, count)``
   representation available on real data -- the mean is never formed as a float,
   so two equal means can never compare unequal and two distinct means can never
   collide. Parsed via integer arithmetic on the decimal text, never through
   ``float`` -- ``float("3.5") * 2`` is exact here, but relying on that is a
   silent dependence on the input alphabet.
3. **Unrated films exist.** ``movies.csv`` lists films that appear in no row of
   ``ratings.csv``, so their mean rating is genuinely undefined. They become
   ``has_value = False`` and sort last -- the missing-value path from G8
   exercised on real data rather than on a constructed fixture. The count is
   reported by ``scripts/fetch_data.py`` on first download rather than asserted
   here, since this repository never holds the archive to check it against.

Text
----
``title``, ``genres`` and the free-text ``tags`` are concatenated. Genres arrive
pipe-separated (``Action|Sci-Fi``) and the release year is parenthesised in the
title (``Heat (1995)``); both are split into tokens here rather than left for the
tokeniser, since neither is natural language and the tokeniser should not have to
know about this dataset's conventions.
"""

from __future__ import annotations

import csv
import io
import math
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tfidf_stability.utils.hashing import hash_bytes
from tfidf_stability.utils.validation import DataIntegrityError

__all__ = [
    "MOVIELENS_SHA256",
    "MOVIELENS_URL",
    "MovieLensCorpus",
    "load",
    "parse_archive",
]

#: Canonical source. GroupLens overwrites this URL in place -- hence the pin.
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

#: Pinned digest of the archive these results were computed against. Recorded
#: from a verified first download on 2026-08-12; a mismatch is an error, never a
#: warning, because GroupLens replaces this file in place at a stable URL.
#:
#: The archive it identifies: 9742 films, 100836 ratings, 610 users, 18 films
#: with no rating at all.
MOVIELENS_SHA256: str | None = "696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436"

#: Rating at or above which an interaction counts as positive (G10 item 5).
DEFAULT_MIN_WEIGHT = 4.0

_EXPECTED_MEMBERS = ("movies.csv", "ratings.csv", "tags.csv")


@dataclass(frozen=True, slots=True)
class MovieLensCorpus:
    """Documents, attributes and interactions derived from the archive."""

    archive_sha256: str
    doc_ids: tuple[str, ...]
    texts: tuple[str, ...]
    attributes: tuple[dict[str, Any], ...]
    interactions: tuple[tuple[str, str, float], ...]
    n_ratings: int
    n_users: int
    n_unrated: int

    @property
    def n_documents(self) -> int:
        return len(self.doc_ids)

    def records(self) -> list[dict[str, Any]]:
        """Corpus rows in the shape :class:`AttributeTable` and the CLI expect."""
        return [
            {"doc_id": doc_id, "text": text, **attributes}
            for doc_id, text, attributes in zip(
                self.doc_ids, self.texts, self.attributes, strict=True
            )
        ]


def _double_rating(text: str) -> int:
    """``"3.5"`` to ``7``, in exact integer arithmetic.

    MovieLens ratings are quantised to halves, so twice a rating is an integer.
    Going via ``float`` would happen to be exact for this alphabet, but that is a
    property of the input rather than of the code, and it is the kind of accident
    that stops holding when someone points this parser at ``ml-25m``. Doing it in
    integers makes the guarantee structural, and the ``ValueError`` below is what
    detects an input that breaks the assumption instead of silently rounding it.
    """
    whole, _, frac = text.strip().partition(".")
    doubled = int(whole) * 2
    if not frac:
        return doubled
    if frac in ("0", "00"):
        return doubled
    if frac in ("5", "50"):
        return doubled + 1
    raise DataIntegrityError(
        f"rating {text!r} is not a multiple of 0.5; the exact (2*sum, count) "
        f"representation from spec_addenda G8 assumes half-integer ratings"
    )


def _title_tokens(title: str) -> str:
    """Split the parenthesised year off a title.

    ``"Heat (1995)"`` to ``"Heat 1995"``. The tokeniser would otherwise have to
    treat brackets as a dataset-specific convention, which it should not.
    """
    return title.replace("(", " ").replace(")", " ")


def parse_archive(data: bytes, *, min_weight: float = DEFAULT_MIN_WEIGHT) -> MovieLensCorpus:
    """Parse the archive bytes. Pure: no network, no filesystem, no clock.

    Args:
        data: The raw ``ml-latest-small.zip`` bytes.
        min_weight: Rating at or above which a rating becomes a positive
            interaction (G10 item 5). The threshold is compared in the doubled
            integer domain, so ``4.0`` means exactly 4.0 and not
            "4.0 give or take a rounding error".
    """
    digest = hash_bytes(data)

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = {Path(n).name: n for n in archive.namelist() if not n.endswith("/")}
        missing = [m for m in _EXPECTED_MEMBERS if m not in names]
        if missing:
            raise DataIntegrityError(
                f"archive is missing {missing}; expected an ml-latest-small layout, "
                f"found {sorted(names)}"
            )

        def rows(member: str) -> list[dict[str, str]]:
            with archive.open(names[member]) as handle:
                # utf-8-sig: GroupLens ships a BOM, which would otherwise become
                # part of the first column's *name* and break every lookup.
                text = io.TextIOWrapper(handle, encoding="utf-8-sig", newline="")
                return list(csv.DictReader(text))

        movie_rows = rows("movies.csv")
        rating_rows = rows("ratings.csv")
        tag_rows = rows("tags.csv")

    # -- ratings: exact sums, kept as integers throughout -----------------
    sum2: defaultdict[str, int] = defaultdict(int)
    count: defaultdict[str, int] = defaultdict(int)
    users: set[str] = set()
    interactions: list[tuple[str, str, float]] = []
    # Compared as integers. `min_weight * 2` is exact for any half-integer
    # threshold, and math.ceil handles a threshold between two representable
    # ratings (4.3 admits 4.5 and above) without a float comparison per row.
    min_doubled = math.ceil(min_weight * 2)
    for row in rating_rows:
        movie, user = row["movieId"], row["userId"]
        doubled = _double_rating(row["rating"])
        sum2[movie] += doubled
        count[movie] += 1
        users.add(user)
        if doubled >= min_doubled:
            interactions.append((f"u{user}", f"m{movie}", doubled / 2.0))

    # -- tags: text and the engagement attribute --------------------------
    tags: defaultdict[str, list[str]] = defaultdict(list)
    for row in tag_rows:
        tags[row["movieId"]].append(row["tag"])

    # Sorted by movieId as an *integer*, so ordering is a property of the data
    # and not of CSV row order or of string collation ("10" before "9").
    movie_rows.sort(key=lambda r: int(r["movieId"]))

    doc_ids: list[str] = []
    texts: list[str] = []
    attributes: list[dict[str, Any]] = []
    n_unrated = 0
    for row in movie_rows:
        movie = row["movieId"]
        n = count[movie]
        n_unrated += n == 0
        doc_ids.append(f"m{movie}")
        texts.append(
            " ".join(
                (
                    _title_tokens(row["title"]),
                    row["genres"].replace("|", " "),
                    *tags[movie],
                )
            )
        )
        attributes.append(
            {
                "popularity": n,
                # G8's exact pair. A film with no ratings has count 0, which is
                # the missing case -- not a mean of zero, which would rank it
                # alongside genuinely awful films rather than last.
                "rating_sum2": sum2[movie],
                "rating_count": n,
                "engagement": len(tags[movie]),
            }
        )

    # Interaction order follows ratings.csv, which is grouped by user and sorted
    # by timestamp within a user -- deterministic, but re-sorted anyway so the
    # result does not depend on that continuing to hold.
    interactions.sort()

    return MovieLensCorpus(
        archive_sha256=digest,
        doc_ids=tuple(doc_ids),
        texts=tuple(texts),
        attributes=tuple(attributes),
        interactions=tuple(interactions),
        n_ratings=len(rating_rows),
        n_users=len(users),
        n_unrated=n_unrated,
    )


def load(
    archive: Path | str,
    *,
    expect_sha256: str | None = MOVIELENS_SHA256,
    min_weight: float = DEFAULT_MIN_WEIGHT,
) -> MovieLensCorpus:
    """Load a local archive, verifying its digest.

    Args:
        archive: Path to ``ml-latest-small.zip``.
        expect_sha256: Digest to require. ``None`` skips the check and is only
            appropriate the first time, when the pin is being established.
        min_weight: Positive-interaction threshold; see :func:`parse_archive`.

    Raises:
        DataIntegrityError: If the digest does not match, or the archive does not
            look like ml-latest-small.
    """
    path = Path(archive)
    if not path.exists():
        raise DataIntegrityError(
            f"{path} not found. MovieLens may not be redistributed, so it is not "
            f"in the repository -- run 'python scripts/fetch_data.py' to download it."
        )

    data = path.read_bytes()
    digest = hash_bytes(data)
    if expect_sha256 is not None and digest != expect_sha256:
        raise DataIntegrityError(
            f"{path} does not match the pinned digest.\n"
            f"    expected {expect_sha256}\n"
            f"    actual   {digest}\n"
            f"GroupLens updates ml-latest-small in place, so this most likely means "
            f"the upstream data changed. Reconcile deliberately: re-run the "
            f"experiments and update MOVIELENS_SHA256, or obtain the pinned archive."
        )
    return parse_archive(data, min_weight=min_weight)
