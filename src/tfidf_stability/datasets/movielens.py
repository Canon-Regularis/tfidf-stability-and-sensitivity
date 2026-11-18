"""Parse the MovieLens ``ml-latest-small`` archive into a corpus.

The archive is not redistributed; ``scripts/fetch_data.py`` fetches a copy and :func:`load`
checks it against a pinned SHA-256. Films sort by integer ``movieId``; ratings sum as doubled
integers (``docs/spec_addenda.md#g8``). Rationale: docs/engineering_notes.md#movielens-ingestion
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

#: Canonical source. GroupLens overwrites this URL in place, hence the pin.
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

#: Pinned digest of the archive these results were computed against, recorded
#: on 2026-08-12. A mismatch is an error, since GroupLens replaces the file in
#: place at a stable URL. The archive holds 9742 films, 100836 ratings, 610
#: users and 18 films with no rating.
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
    #: The same documents before the fields were joined: title, genres, then one
    #: entry per tag. Carried so n-grams do not span a seam between two of them,
    #: which the joined text alone cannot express (``spec_addenda.md#g7``).
    fields: tuple[tuple[str, ...], ...]
    attributes: tuple[dict[str, Any], ...]
    interactions: tuple[tuple[str, str, float], ...]
    #: Rating at or above which a rating becomes a positive interaction. Carried
    #: so the manifest records the threshold behind the interaction set;
    #: changing it moves every profile built from the dataset.
    min_weight: float
    n_ratings: int
    n_users: int
    n_unrated: int

    @property
    def n_documents(self) -> int:
        return len(self.doc_ids)

    def records(self) -> list[dict[str, Any]]:
        """Corpus rows in the shape :class:`AttributeTable` and the CLI expect.

        ``text`` is the joined form; ``fields`` is the same content unjoined, which
        :func:`~tfidf_stability.preprocessing.pipeline.preprocess_records` prefers.
        """
        return [
            {"doc_id": doc_id, "text": text, "fields": list(parts), **attributes}
            for doc_id, text, parts, attributes in zip(
                self.doc_ids, self.texts, self.fields, self.attributes, strict=True
            )
        ]


def _double_rating(text: str) -> int:
    """Convert ``"3.5"`` to ``7`` in exact integer arithmetic.

    MovieLens ratings are quantised to halves, so twice a rating is an integer.
    Parsing the decimal text avoids relying on ``float`` being exact for this
    input; a rating off the half grid raises instead of being rounded.
    """
    stripped = text.strip()
    whole, _, frac = stripped.partition(".")
    # `int()` accepts PEP 515 underscores and non-ASCII decimal digits: without
    # this guard "1_0" would parse as ten and "٣.5" as three point five.
    # The archive is pinned, so this parser takes `d`, `d.0`, `d.00`, `d.5` and
    # `d.50` only.
    digits = whole.removeprefix("-")  # one sign; `lstrip` would accept "--3"
    if not whole.isascii() or not digits.isdigit():
        raise DataIntegrityError(f"rating {text!r} is not a plain decimal number")
    # The half moves the rating away from zero, so it takes the sign of the
    # whole part. `int("-0")` is 0, so `doubled` alone does not carry that sign
    # for "-0.5". Each result sums into the `2*sum` half of G8's
    # `(2*sum, count)` pair.
    sign = -1 if whole.startswith("-") else 1
    doubled = int(whole) * 2
    if not frac or frac in ("0", "00"):
        return doubled
    if frac in ("5", "50"):
        return doubled + sign
    raise DataIntegrityError(
        f"rating {text!r} is not a multiple of 0.5; the exact (2*sum, count) "
        f"representation from spec_addenda G8 assumes half-integer ratings"
    )


def _title_tokens(title: str) -> str:
    """Split the parenthesised year off a title.

    ``"Heat (1995)"`` to ``"Heat 1995"``, so the tokeniser need not treat
    brackets as a dataset-specific convention.
    """
    return title.replace("(", " ").replace(")", " ")


def parse_archive(data: bytes, *, min_weight: float = DEFAULT_MIN_WEIGHT) -> MovieLensCorpus:
    """Parse the archive bytes with no network, filesystem or clock access.

    Args:
        data: The raw ``ml-latest-small.zip`` bytes.
        min_weight: Inclusive positive-rating threshold (G10 item 5), as ``ceil(2 * min_weight)``.
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
                # GroupLens ships a BOM; without utf-8-sig it joins the first
                # column's name and breaks every lookup.
                text = io.TextIOWrapper(handle, encoding="utf-8-sig", newline="")
                return list(csv.DictReader(text))

        movie_rows = rows("movies.csv")
        rating_rows = rows("ratings.csv")
        tag_rows = rows("tags.csv")

    # ratings: exact sums, kept as integers throughout ====================
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

    # tags: text and the engagement attribute =============================
    tags: defaultdict[str, list[str]] = defaultdict(list)
    for row in tag_rows:
        tags[row["movieId"]].append(row["tag"])
    # Sorted, as the movie rows and the interactions below already are. Tags are
    # concatenated into the document text, so their order decides which
    # tag-to-tag n-grams exist. Under `tags.csv` row order the vocabulary would
    # be a property of GroupLens' export, not of the data.
    for movie in tags:
        tags[movie].sort()

    # movieId as an integer, so ordering is a property of the data rather than
    # of CSV row order or string collation ("10" before "9").
    movie_rows.sort(key=lambda r: int(r["movieId"]))

    doc_ids: list[str] = []
    texts: list[str] = []
    fields: list[tuple[str, ...]] = []
    attributes: list[dict[str, Any]] = []
    n_unrated = 0
    for row in movie_rows:
        movie = row["movieId"]
        n = count[movie]
        n_unrated += n == 0
        doc_ids.append(f"m{movie}")
        # Title, genres and each tag are separate fields, not one passage: an
        # n-gram bridging two of them is a product of this concatenation rather
        # than of the data. Both forms are carried, and `preprocess_records`
        # takes the unjoined one.
        parts = (_title_tokens(row["title"]), row["genres"].replace("|", " "), *tags[movie])
        fields.append(parts)
        texts.append(" ".join(parts))
        attributes.append(
            {
                "popularity": n,
                # G8's exact pair. No ratings gives count 0, the missing case; a
                # mean of zero would rank the film beside genuinely awful ones
                # instead of last.
                "rating_sum2": sum2[movie],
                "rating_count": n,
                "engagement": len(tags[movie]),
            }
        )

    # ratings.csv is grouped by user and sorted by timestamp within a user, so
    # interaction order is already deterministic; re-sorted so the result does
    # not depend on that continuing to hold.
    interactions.sort()

    return MovieLensCorpus(
        archive_sha256=digest,
        doc_ids=tuple(doc_ids),
        texts=tuple(texts),
        fields=tuple(fields),
        attributes=tuple(attributes),
        interactions=tuple(interactions),
        min_weight=min_weight,
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
    """Load ``ml-latest-small.zip`` from the path ``archive``, verifying its digest.

    ``expect_sha256=None`` skips the check (first download only); ``min_weight`` is as
    in :func:`parse_archive`. Raises ``DataIntegrityError`` if the file or a member is
    missing, the digest differs or a rating is malformed.
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
