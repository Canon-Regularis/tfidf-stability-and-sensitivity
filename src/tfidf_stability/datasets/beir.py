"""Parse a BEIR archive into documents, queries and relevance judgements.

The archives are not redistributed; ``scripts/fetch_data.py`` fetches a copy and :func:`load`
checks it against :data:`PINNED_SHA256`. Documents and queries sort by UTF-8 identifier.
Rationale: docs/engineering_notes.md#beir-ingestion
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from tfidf_stability.utils.hashing import hash_bytes
from tfidf_stability.utils.validation import DataIntegrityError

__all__ = [
    "BEIR_URL",
    "PINNED_SHA256",
    "BeirCorpus",
    "load",
    "parse_archive",
]

#: Where each archive is published; ``{variant}`` is the top-level folder name.
BEIR_URL: Final = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{variant}.zip"

#: Pinned digest of each archive, recorded on 2026-11-18. SciFact holds 5183
#: documents and 300 judged test queries; NFCorpus holds 3633 and 323.
PINNED_SHA256: Final[dict[str, str]] = {
    "scifact": "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
    "nfcorpus": "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b",
}

_QRELS_HEADER: Final = "query-id\tcorpus-id\tscore"
_GRADE: Final = re.compile(r"-?[0-9]+")


def _utf8(identifier: str) -> bytes:
    return identifier.encode("utf-8")


@dataclass(frozen=True, slots=True)
class BeirCorpus:
    """Documents, judged queries and judgements from one split of an archive."""

    archive_sha256: str
    variant: str
    split: str
    doc_ids: tuple[str, ...]
    texts: tuple[str, ...]
    #: Title and body, each only when non-blank, so no n-gram spans the seam.
    fields: tuple[tuple[str, ...], ...]
    #: ``(query_id, text)`` for every query the split judges, by UTF-8 identifier.
    queries: tuple[tuple[str, str], ...]
    #: ``(query_id, doc_id, grade)``, by UTF-8 query then document identifier.
    qrels: tuple[tuple[str, str, int], ...]

    @property
    def n_documents(self) -> int:
        return len(self.doc_ids)

    @property
    def n_queries_without_relevant(self) -> int:
        """Judged queries with no grade above zero: nDCG and recall are undefined."""
        relevant = {query_id for query_id, _, grade in self.qrels if grade > 0}
        return sum(1 for query_id, _ in self.queries if query_id not in relevant)

    def records(self) -> list[dict[str, Any]]:
        """Corpus rows, in the shape ``preprocess_records`` expects."""
        return [
            {"doc_id": doc_id, "text": text, "fields": list(parts)}
            for doc_id, text, parts in zip(self.doc_ids, self.texts, self.fields, strict=True)
        ]

    def query_texts(self) -> dict[str, str]:
        return dict(self.queries)

    def judgements(self) -> dict[str, dict[str, int]]:
        """Grade by document, per judged query, including grades of zero or below."""
        out: dict[str, dict[str, int]] = {query_id: {} for query_id, _ in self.queries}
        for query_id, doc_id, grade in self.qrels:
            out[query_id][doc_id] = grade
        return out


def _lines(text: str) -> list[str]:
    """Split on LF alone, as JSON Lines does, dropping a CR before it.

    ``str.splitlines`` also breaks on U+0085, U+2028 and U+2029, which JSON
    allows unescaped inside a string.
    """
    return [line.removesuffix("\r") for line in text.split("\n")]


def _decode(data: bytes, member: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DataIntegrityError(f"{member}: not UTF-8 ({exc.reason} at byte {exc.start})") from exc


def _jsonl(text: str, member: str) -> list[dict[str, Any]]:
    """Parse JSON Lines, skipping blank lines; every row must be an object."""
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(_lines(text), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataIntegrityError(f"{member} line {number}: malformed JSON ({exc.msg})") from exc
        if not isinstance(row, dict):
            raise DataIntegrityError(f"{member} line {number}: expected a JSON object")
        rows.append(row)
    return rows


def _text_field(row: dict[str, Any], key: str, member: str, *, required: bool) -> str:
    """Read a string field; a missing optional field reads as the empty string."""
    if key not in row:
        if required:
            raise DataIntegrityError(f"{member}: a row lacks {key!r}: {row.get('_id')!r}")
        return ""
    value = row[key]
    if not isinstance(value, str):
        raise DataIntegrityError(f"{member}: {key!r} of {row.get('_id')!r} is not a string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        # JSON can escape a lone surrogate, which no UTF-8 hash or sort key accepts.
        raise DataIntegrityError(
            f"{member}: {key!r} of {row.get('_id')!r} is not valid text"
        ) from exc
    return value


def _identified(rows: list[dict[str, Any]], member: str) -> dict[str, dict[str, Any]]:
    """Index rows by a non-empty, unique ``_id``."""
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = _text_field(row, "_id", member, required=True)
        if not identifier:
            raise DataIntegrityError(f"{member}: a row has an empty '_id'")
        if identifier in out:
            raise DataIntegrityError(f"{member}: duplicate '_id' {identifier!r}")
        out[identifier] = row
    return out


def _qrels(text: str, member: str) -> list[tuple[str, str, int]]:
    """Parse a tab-separated qrels file under its fixed header."""
    lines = _lines(text)
    if lines[0] != _QRELS_HEADER:
        found = lines[0]
        raise DataIntegrityError(
            f"{member}: expected the header {_QRELS_HEADER!r}, found {found!r}"
        )
    rows: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str]] = set()
    for number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        cells = line.split("\t")
        if len(cells) != 3 or not cells[0] or not cells[1] or not _GRADE.fullmatch(cells[2]):
            raise DataIntegrityError(
                f"{member} line {number}: expected 'query doc grade': {line!r}"
            )
        query_id, doc_id = cells[0], cells[1]
        if (query_id, doc_id) in seen:
            raise DataIntegrityError(
                f"{member} line {number}: {query_id!r} judges {doc_id!r} twice"
            )
        seen.add((query_id, doc_id))
        rows.append((query_id, doc_id, int(cells[2])))
    return rows


def parse_archive(data: bytes, *, split: str = "test") -> BeirCorpus:
    """Parse the archive bytes with no network, filesystem or clock access.

    Members are read by exact path under the archive's single top-level folder. Every
    query the split judges is kept; a judgement naming an unknown query or document,
    a duplicate identifier or a malformed row raises ``DataIntegrityError``.
    """
    digest = hash_bytes(data)
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise DataIntegrityError(f"not a zip archive: {exc}") from exc
    with archive:
        files = [name for name in archive.namelist() if not name.endswith("/")]
        tops = sorted({name.split("/", 1)[0] for name in files})
        if len(tops) != 1:
            raise DataIntegrityError(f"expected one top-level folder, found {tops}")
        variant = tops[0]
        members = {
            "corpus": f"{variant}/corpus.jsonl",
            "queries": f"{variant}/queries.jsonl",
            "qrels": f"{variant}/qrels/{split}.tsv",
        }
        missing = [path for path in members.values() if path not in files]
        if missing:
            raise DataIntegrityError(f"archive is missing {missing}; found {sorted(files)}")
        texts = {key: _decode(archive.read(path), path) for key, path in members.items()}

    corpus = _identified(_jsonl(texts["corpus"], members["corpus"]), members["corpus"])
    all_queries = _identified(_jsonl(texts["queries"], members["queries"]), members["queries"])
    judged = _qrels(texts["qrels"], members["qrels"])

    unknown_queries = sorted({q for q, _, _ in judged if q not in all_queries}, key=_utf8)
    unknown_docs = sorted({d for _, d, _ in judged if d not in corpus}, key=_utf8)
    if unknown_queries or unknown_docs:
        raise DataIntegrityError(
            f"{members['qrels']} names {len(unknown_queries)} unknown queries "
            f"{unknown_queries[:3]} and {len(unknown_docs)} unknown documents {unknown_docs[:3]}"
        )

    doc_ids = sorted(corpus, key=_utf8)
    fields: list[tuple[str, ...]] = []
    for doc_id in doc_ids:
        row = corpus[doc_id]
        title = _text_field(row, "title", members["corpus"], required=False)
        body = _text_field(row, "text", members["corpus"], required=True)
        fields.append(tuple(part for part in (title, body) if part.strip()))

    query_ids = sorted({q for q, _, _ in judged}, key=_utf8)
    queries = tuple(
        (q, _text_field(all_queries[q], "text", members["queries"], required=True))
        for q in query_ids
    )
    qrels = tuple(sorted(judged, key=lambda row: (_utf8(row[0]), _utf8(row[1]))))

    return BeirCorpus(
        archive_sha256=digest,
        variant=variant,
        split=split,
        doc_ids=tuple(doc_ids),
        texts=tuple(" ".join(parts) for parts in fields),
        fields=tuple(fields),
        queries=queries,
        qrels=qrels,
    )


def load(
    archive: Path | str, *, variant: str, expect_sha256: str | None, split: str = "test"
) -> BeirCorpus:
    """Load the ``variant`` archive from the path ``archive``, verifying its digest.

    ``expect_sha256=None`` skips the check (first download only). Raises
    ``DataIntegrityError`` if the file is missing, the digest differs, the archive
    holds another variant or :func:`parse_archive` refuses it.
    """
    path = Path(archive)
    if not path.exists():
        raise DataIntegrityError(
            f"{path} not found. BEIR archives are not in the repository -- run "
            f"'python scripts/fetch_data.py --dataset beir_{variant}' to download it."
        )
    data = path.read_bytes()
    digest = hash_bytes(data)
    if expect_sha256 is not None and digest != expect_sha256:
        raise DataIntegrityError(
            f"{path} does not match the pinned digest.\n"
            f"    expected {expect_sha256}\n"
            f"    actual   {digest}\n"
            f"Reconcile deliberately: re-run the experiments and update PINNED_SHA256, "
            f"or obtain the pinned archive."
        )
    corpus = parse_archive(data, split=split)
    if corpus.variant != variant:
        raise DataIntegrityError(f"{path} holds {corpus.variant!r}, not {variant!r}")
    return corpus
