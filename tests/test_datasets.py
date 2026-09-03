"""Dataset generation, parsing and loading.

The MovieLens tests build a synthetic archive: the licence forbids committing the
real one, so a test needing it would skip everywhere including CI.

The archive reproduces the quirks of ``ml-latest-small`` the parser copes with:

* a UTF-8 BOM on every CSV (corrupts the first column's name, leaving its value
  intact);
* pipe-separated genres and parenthesised release years;
* films with no ratings at all, which upstream has;
* half-star ratings, which make G8's exact integer pair available;
* ``movieId`` ordering where string and integer collation disagree.
"""

from __future__ import annotations

import bisect
import hashlib
import importlib.util
import io
import json
import random as random_module
import sys
import zipfile
from itertools import pairwise
from pathlib import Path
from types import ModuleType

import pytest

from tfidf_stability.datasets import movielens, synthetic
from tfidf_stability.datasets.loaders import load_dataset, load_jsonl_corpus
from tfidf_stability.preprocessing.pipeline import PreprocessingPipeline
from tfidf_stability.ranking.attributes import AttributeTable
from tfidf_stability.similarity.cosine import cosine_against_corpus
from tfidf_stability.utils.hashing import hash_bytes, hash_text
from tfidf_stability.utils.io import canonical_json
from tfidf_stability.utils.validation import DataIntegrityError
from tfidf_stability.vectorisation.tfidf import TfidfVectoriser

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# A fake ml-latest-small
# ---------------------------------------------------------------------------
_MOVIES = """movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children
2,Jumanji (1995),Adventure|Children|Fantasy
10,GoldenEye (1995),Action|Adventure|Thriller
3,Grumpier Old Men (1995),Comedy|Romance
9,Sudden Death (1995),Action
"""

# 3.5 and 4.5: half-stars are what G8's exact (2*sum, count) pair is for.
_RATINGS = """userId,movieId,rating,timestamp
1,1,4.0,964982703
1,2,3.5,964981247
2,1,5.0,964982224
2,10,3.0,964983815
3,1,4.5,964982931
3,3,2.0,964982400
"""

_TAGS = """userId,movieId,tag,timestamp
1,1,pixar,1445714994
2,1,fun,1445714996
3,10,bond,1445715000
"""


def _archive(
    movies: str = _MOVIES,
    ratings: str = _RATINGS,
    tags: str = _TAGS,
    *,
    prefix: str = "ml-latest-small/",
) -> bytes:
    """Build a MovieLens-shaped zip, BOM and all."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, body in (("movies.csv", movies), ("ratings.csv", ratings), ("tags.csv", tags)):
            # utf-8-sig: GroupLens does ship a BOM, and a parser reading plain
            # utf-8 leaves it glued to the front of the first column name.
            archive.writestr(f"{prefix}{name}", body.encode("utf-8-sig"))
    return buffer.getvalue()


def _spec(n_docs: int, vocab_size: int, **kw) -> synthetic.SyntheticSpec:
    """A spec whose duplicate and twin counts scale with its size.

    The generator refuses a spec whose special-case documents outnumber its
    documents.
    """
    return synthetic.SyntheticSpec(
        n_docs=n_docs,
        vocab_size=vocab_size,
        n_exact_duplicates=kw.pop("n_exact_duplicates", max(2, n_docs // 20)),
        n_twin_pairs=kw.pop("n_twin_pairs", max(2, n_docs // 20)),
        **kw,
    )


# ---------------------------------------------------------------------------
# MovieLens parsing
# ---------------------------------------------------------------------------
def test_parses_the_archive_layout() -> None:
    corpus = movielens.parse_archive(_archive())
    assert corpus.n_documents == 5
    assert corpus.n_ratings == 6
    assert corpus.n_users == 3


def test_documents_are_ordered_by_integer_movie_id() -> None:
    """Neither string order nor CSV row order.

    The file lists 3 and 9 after 10, and string collation puts "10" before "9".
    Either alternative indexes documents stably but arbitrarily, so a result
    stops reproducing against a re-exported file.
    """
    corpus = movielens.parse_archive(_archive())
    assert corpus.doc_ids == ("m1", "m2", "m3", "m9", "m10")


def test_the_byte_order_mark_does_not_become_part_of_a_column_name() -> None:
    corpus = movielens.parse_archive(_archive())
    assert corpus.doc_ids[0] == "m1"  # would be "mNone" under plain utf-8


def test_ratings_are_kept_as_an_exact_integer_pair() -> None:
    """G8's representation.

    Toy Story: 4.0 + 5.0 + 4.5 = 13.5, a mean of 4.5 over three ratings, stored as
    ``(27, 3)``. 13.5/3 happens to be exact in binary64; the code never forms it,
    so the guarantee does not rest on that.
    """
    corpus = movielens.parse_archive(_archive())
    by_id = dict(zip(corpus.doc_ids, corpus.attributes, strict=True))
    assert by_id["m1"]["rating_sum2"] == 27
    assert by_id["m1"]["rating_count"] == 3
    assert all(isinstance(a["rating_sum2"], int) for a in corpus.attributes)


def test_a_rating_that_is_not_a_half_star_is_rejected() -> None:
    """The assumption the exact pair rests on is checked at parse time."""
    with pytest.raises(DataIntegrityError, match=r"multiple of 0\.5"):
        movielens.parse_archive(_archive(ratings="userId,movieId,rating,timestamp\n1,1,3.7,0\n"))


def test_an_unrated_film_has_count_zero_rather_than_a_mean_of_zero() -> None:
    """The missing-value path: a mean of zero would rank an unrated film alongside
    a genuinely terrible one. ``count == 0`` makes the attribute table treat the
    rating as absent and sort it last.
    """
    corpus = movielens.parse_archive(_archive())
    by_id = dict(zip(corpus.doc_ids, corpus.attributes, strict=True))
    assert by_id["m9"]["rating_count"] == 0
    assert by_id["m9"]["rating_sum2"] == 0
    assert corpus.n_unrated == 1


def test_genres_and_years_are_split_into_tokens() -> None:
    corpus = movielens.parse_archive(_archive())
    text = dict(zip(corpus.doc_ids, corpus.texts, strict=True))["m1"]
    assert "Adventure" in text
    assert "Animation" in text
    assert "|" not in text  # genres were split
    assert "1995" in text
    assert "(" not in text  # the year was unbracketed
    # Tags contribute to the document text as well as the engagement count.
    assert "pixar" in text
    assert "fun" in text


def test_only_ratings_at_or_above_the_threshold_become_interactions() -> None:
    """G10 item 5: the boundary is inclusive, so 4.0 is in and 3.5 is out."""
    corpus = movielens.parse_archive(_archive())
    assert corpus.interactions == (
        ("u1", "m1", 4.0),
        ("u2", "m1", 5.0),
        ("u3", "m1", 4.5),
    )


def test_parsing_is_a_pure_function_of_the_bytes() -> None:
    data = _archive()
    assert movielens.parse_archive(data) == movielens.parse_archive(data)


def test_a_flat_archive_without_the_directory_prefix_still_parses() -> None:
    corpus = movielens.parse_archive(_archive(prefix=""))
    assert corpus.n_documents == 5


def test_an_archive_missing_a_member_is_rejected_by_name() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("ml-latest-small/movies.csv", _MOVIES)
    with pytest.raises(DataIntegrityError, match=r"ratings\.csv"):
        movielens.parse_archive(buffer.getvalue())


def test_a_missing_file_names_the_fetch_script(tmp_path) -> None:
    """The archive is absent by design, so the error has to say how to get it."""
    with pytest.raises(DataIntegrityError, match=r"fetch_data\.py"):
        movielens.load(tmp_path / "absent.zip")


def test_a_digest_mismatch_is_fatal_rather_than_a_warning(tmp_path) -> None:
    """GroupLens updates the archive in place, so this is the realistic failure."""
    path = tmp_path / "ml.zip"
    path.write_bytes(_archive())
    with pytest.raises(DataIntegrityError, match="pinned digest"):
        movielens.load(path, expect_sha256="0" * 64)


def test_the_digest_is_reported_even_when_unpinned(tmp_path) -> None:
    path = tmp_path / "ml.zip"
    path.write_bytes(_archive())
    corpus = movielens.load(path, expect_sha256=None)
    assert len(corpus.archive_sha256) == 64


def test_movielens_attributes_are_accepted_by_the_attribute_table() -> None:
    """Real-shaped data reaches the ranking layer intact, the unrated film
    included: its rating must arrive missing rather than zero."""
    corpus = movielens.parse_archive(_archive())
    table = AttributeTable.from_records(corpus.records())
    assert table.n_documents == 5
    assert len(table.digest()) == 64


# ---------------------------------------------------------------------------
# The synthetic generator
# ---------------------------------------------------------------------------
def test_generation_is_deterministic() -> None:
    spec = _spec(60, 120, n_users=10)
    assert synthetic.generate(spec) == synthetic.generate(spec)


def test_a_different_seed_gives_a_different_corpus() -> None:
    base = _spec(60, 120, n_users=10)
    other = _spec(60, 120, n_users=10, seed=base.seed + 1)
    assert synthetic.generate(base).documents != synthetic.generate(other).documents


def test_exact_duplicates_are_genuinely_identical() -> None:
    """The tau = 0 case: only the tie-break can separate these."""
    corpus = synthetic.generate(_spec(80, 150, n_users=10))
    texts = corpus.features_by_doc()
    assert corpus.exact_duplicate_pairs
    for a, b in corpus.exact_duplicate_pairs:
        assert texts[a] == texts[b]


def test_twins_differ_by_exactly_one_token() -> None:
    corpus = synthetic.generate(_spec(80, 150, n_users=10))
    texts = corpus.features_by_doc()
    assert corpus.twins
    for a, b, _ in corpus.twins:
        assert len(texts[b]) == len(texts[a]) + 1
        assert texts[b][: len(texts[a])] == texts[a]


def test_no_document_is_empty() -> None:
    """An empty document has a zero norm, a separate G3 case."""
    corpus = synthetic.generate(_spec(100, 200, n_users=10))
    assert all(len(d) >= corpus.spec.len_min for d in corpus.documents)


def test_document_ids_are_unique() -> None:
    """The precondition of the permutation-identity claim."""
    corpus = synthetic.generate(_spec(200, 300, n_users=10))
    assert len(set(corpus.doc_ids)) == corpus.n_documents


def test_the_generator_never_calls_an_unstable_prng_method(monkeypatch) -> None:
    """``choice``/``sample``/``shuffle`` are not stable across CPython versions.

    Asserted by making them explode rather than by reading the source, so a call
    added later still fails.
    """
    for name in ("choice", "choices", "sample", "shuffle", "randrange", "randint"):
        monkeypatch.setattr(
            random_module.Random,
            name,
            lambda *a, **k: pytest.fail("used a PRNG method that is not version-stable"),
        )
    synthetic.generate(_spec(40, 80, n_users=5))


# ---------------------------------------------------------------------------
# Near-tie structure (G22)
# ---------------------------------------------------------------------------
def test_find_near_ties_orders_by_gap() -> None:
    scores = [1.0, 0.9, 0.85, 0.2]
    found = synthetic.find_near_ties(scores, limit=2)
    assert [round(p.gap, 10) for p in found] == [0.05, 0.1]
    assert found[0].rank == 2  # 1-indexed: the pair (r2, r3)


def test_find_near_ties_can_report_the_exact_tie_block() -> None:
    """The case that dominates in practice; see G22."""
    scores = [1.0, 1.0, 0.5, 0.0, 0.0, 0.0]
    assert synthetic.find_near_ties(scores, strictly_positive=True)[0].gap == 0.5
    exact = synthetic.find_near_ties(scores, limit=10, strictly_positive=False)
    assert sum(p.is_exact for p in exact) == 3


def test_find_near_ties_handles_degenerate_lengths() -> None:
    assert synthetic.find_near_ties([]) == []
    assert synthetic.find_near_ties([1.0]) == []


def test_a_single_token_edit_cannot_produce_a_fine_near_tie() -> None:
    """G22 as an executable statement.

    Section 2.2's ``tf = count / L`` makes a one-token edit a ``1/(L+1)`` relative
    perturbation, so document length floors the separation it induces. Hence
    section 7.4 identifies a near-tie instead of constructing one, and the twin
    grid bottoms out around 1e-3.
    """
    corpus = synthetic.generate(_spec(300, 600, n_users=20, len_max=40))
    longest = max(len(d) for d in corpus.documents)
    assert 1.0 / (longest + 1) > 1e-6, (
        "a corpus whose documents were long enough to reach 1e-6 by a single-token "
        "edit would falsify G22's argument"
    )


# ---------------------------------------------------------------------------
# The loader façade
# ---------------------------------------------------------------------------
def test_load_dataset_records_provenance() -> None:
    data = load_dataset("synthetic_tiny")
    assert data.provenance["kind"] == "synthetic"
    assert data.provenance["redistributable"] is True
    assert len(data.provenance["spec_digest"]) == 64


def test_the_corpus_digest_is_over_documents_not_provenance() -> None:
    """Two loads producing identical documents must agree, however they got there."""
    assert load_dataset("synthetic_tiny").digest() == load_dataset("synthetic_tiny").digest()


def test_an_unknown_dataset_name_lists_the_valid_ones() -> None:
    with pytest.raises(DataIntegrityError, match="synthetic_small"):
        load_dataset("no_such_dataset")


def test_movielens_without_an_archive_says_how_to_get_one() -> None:
    with pytest.raises(DataIntegrityError, match=r"fetch_data\.py"):
        load_dataset("movielens_small")


def test_a_jsonl_corpus_round_trips(tmp_path) -> None:
    source = load_dataset("synthetic_tiny")
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in source.records), encoding="utf-8"
    )
    loaded = load_jsonl_corpus(path)
    assert loaded.n_documents == source.n_documents
    assert loaded.digest() == source.digest()


def test_a_jsonl_corpus_missing_required_fields_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"doc_id": "a", "text": "x"}\n{"doc_id": "b"}\n', encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="line 2"):
        load_jsonl_corpus(path)


def test_an_empty_jsonl_corpus_is_rejected(tmp_path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="no records"):
        load_jsonl_corpus(path)


def test_the_interaction_threshold_is_configurable_and_exact() -> None:
    """The config declares `interaction_min_weight`; the code must honour it.

    Compared in the doubled integer domain, so a threshold of 3.5 admits the 3.5
    rating rather than depending on how `3.5 * 2` rounds.
    """
    at_35 = movielens.parse_archive(_archive(), min_weight=3.5)
    assert ("u1", "m2", 3.5) in at_35.interactions
    assert len(at_35.interactions) == 4

    # A threshold between two representable ratings rounds up to the next one.
    at_43 = movielens.parse_archive(_archive(), min_weight=4.3)
    assert {w for _, _, w in at_43.interactions} == {4.5, 5.0}

    assert movielens.parse_archive(_archive(), min_weight=0.0).interactions


@pytest.mark.slow
def test_the_near_tie_interval_below_tau_is_empty() -> None:
    """A property of the synthetic generator alone.

    On a seeded synthetic corpus adjacent score gaps land at zero or well above
    1e-9, with the interval between them empty. The CI experiments run on this
    generator, so the lattice is pinned here.

    It does not generalise, and an earlier version of G22 said it did. Measured on
    MovieLens under the normative naive reduction: 197 of 114,504 adjacent pairs
    fall in (0, 4.44e-16) and the smallest strictly-positive gap is 8.67e-19,
    below the arithmetic noise floor. Recomputed exactly those gaps disappear, so
    they are artefacts of naive summation. Section 7.4's regime differs on real
    data; G22 now records both.
    """
    corpus = synthetic.generate(_spec(600, 1200, n_users=40))
    pipeline = PreprocessingPipeline()
    features = [pipeline.preprocess(" ".join(d)) for d in corpus.documents]
    model = TfidfVectoriser().fit(features, list(corpus.doc_ids))
    documents = [model.document(i) for i in range(model.n_documents)]

    query = TfidfVectoriser.transform_query(list(corpus.documents[0])[:6], model)
    scores = sorted(cosine_against_corpus(query, documents, model.norms), reverse=True)

    gaps = [above - below for above, below in pairwise(scores)]
    assert sum(g == 0.0 for g in gaps) > 0, "expected an exact-tie block"

    positive = [g for g in gaps if g > 0.0]
    assert positive, "expected some strictly-positive gaps"
    assert not [g for g in positive if g < 1e-9], (
        "a gap landed in (0, 1e-9); G22's claim that the near-tie interval is "
        "empty -- and section 7.4's reliance on it -- would need revisiting"
    )
    assert min(positive) > 1e-12


@pytest.mark.slow
def test_the_gap_table_g22_publishes_is_the_one_this_corpus_produces() -> None:
    """G22 prints a five-row table of measured gap counts. This recomputes it.

    Its sibling above pins the row the section's argument rests on, `(0, 1e-9)`
    being empty, and nothing pinned the rest. Three of them drifted: the
    addendum read 104, 2323 and 19 with a smallest positive gap of 2.0e-08,
    against the 130, 2292, 24 and 1.575e-08 measured here. The two figures that
    identify the corpus -- 553 exact ties and 2524 documents scoring above zero
    -- never moved, and no other seed reproduces either, so the generator was
    never in question; the drift is downstream of it.

    The wording drifted with the numbers. "Seed as specified" and "the first six
    tokens of document 0" did not determine a run: document 0 is five words,
    which become nine features once bigrams are added, so "six tokens" has no
    reading in raw words and the two available readings give different tables.
    G22 now names the spec and writes the query as the expression below.

    Marked slow for the 3000-document fit, about six seconds, almost all of it
    preprocessing. There is no cheaper form -- the table is a property of that
    corpus at that size -- and the fast sibling above covers the load-bearing
    claim on a 600-document corpus, so no line depends on this test alone.
    """
    corpus = synthetic.generate(
        synthetic.SyntheticSpec(
            n_docs=3000,
            vocab_size=5000,
            n_exact_duplicates=30,
            n_twin_pairs=60,
            seed=20260811,
        )
    )
    pipeline = PreprocessingPipeline()
    features = [pipeline.preprocess(str(r["text"])) for r in corpus.records()]
    model = TfidfVectoriser().fit(features, list(corpus.doc_ids))
    documents = [model.document(i) for i in range(model.n_documents)]

    query = TfidfVectoriser.transform_query(list(features[0])[:6], model)
    scores = cosine_against_corpus(query, documents, model.norms)
    gaps = [above - below for above, below in pairwise(sorted(scores, reverse=True))]

    positive = [g for g in gaps if g > 0.0]
    counted = {
        "exactly 0": sum(1 for g in gaps if g == 0.0),
        "(0, 1e-9)": sum(1 for g in positive if g < 1e-9),
        "[1e-9, 1e-6)": sum(1 for g in positive if 1e-9 <= g < 1e-6),
        "[1e-6, 1e-3)": sum(1 for g in positive if 1e-6 <= g < 1e-3),
        ">= 1e-3": sum(1 for g in positive if g >= 1e-3),
    }
    published = {
        "exactly 0": 553,
        "(0, 1e-9)": 0,
        "[1e-9, 1e-6)": 130,
        "[1e-6, 1e-3)": 2292,
        ">= 1e-3": 24,
    }

    assert len(gaps) == 2999, "3000 documents give 2999 adjacent pairs"
    assert sum(counted.values()) == len(gaps), "the bands partition the gaps"
    assert counted == published, (
        "G22's table no longer matches this corpus. Update the table in "
        "docs/spec_addenda.md and this test together, and say in the addendum "
        "what moved -- a silent edit to either one is how it drifted before."
    )
    # Exact, not approximate: the corpus is seeded and the arithmetic is fixed,
    # so this is a reproducibility assertion like every other float in this
    # repository. A tolerance would let the value drift the way the counts did.
    assert min(positive) == 1.575373397705304e-08, (
        "G22 quotes the smallest strictly-positive gap as 1.575e-08"
    )
    assert sum(1 for s in scores if s > 0.0) == 2524, (
        "G22: 2524 of 3000 documents score above zero, so the exact-tie mass is "
        "not merely the zero block"
    )


# ---------------------------------------------------------------------------
# Writing the corpus out: the committed bytes are the artefact
# ---------------------------------------------------------------------------
# Nothing regenerates from the spec at experiment time, so PRNG portability stays
# out of the reproducibility surface and the written files are what downstream
# reads. That makes write_corpus part of the provenance chain rather than a
# convenience, and it had no test at all.
def test_writing_a_corpus_produces_the_three_files_and_a_manifest(tmp_path: Path) -> None:
    corpus = synthetic.generate(_spec(n_docs=12, vocab_size=30))
    digests = synthetic.write_corpus(corpus, tmp_path)

    for name in ("corpus.jsonl", "interactions.jsonl", "spec.json"):
        assert (tmp_path / name).is_file(), f"{name} was not written"
        assert name in digests, f"{name} is missing from the returned digests"
    assert (tmp_path / "MANIFEST.sha256").is_file()


def test_the_manifest_records_the_digest_of_every_file_beside_it(tmp_path: Path) -> None:
    """A manifest that disagreed with the files would make the corpus
    unverifiable while looking verified."""
    corpus = synthetic.generate(_spec(n_docs=12, vocab_size=30))
    digests = synthetic.write_corpus(corpus, tmp_path)

    recorded = {}
    for line in (tmp_path / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        recorded[name.strip()] = digest.strip()

    assert recorded == digests
    for name, digest in recorded.items():
        assert hash_text((tmp_path / name).read_text(encoding="utf-8")) == digest


def test_the_manifest_is_written_with_lf_endings_on_every_platform(tmp_path: Path) -> None:
    """The manifest is itself hashed by the repository-wide gate, so a CRLF here
    would make the same corpus verify on Linux and fail on Windows."""
    corpus = synthetic.generate(_spec(n_docs=12, vocab_size=30))
    synthetic.write_corpus(corpus, tmp_path)
    assert b"\r\n" not in (tmp_path / "MANIFEST.sha256").read_bytes()


def test_the_manifest_lists_its_entries_in_sorted_order(tmp_path: Path) -> None:
    """Dictionary iteration order would otherwise leak into a committed file."""
    corpus = synthetic.generate(_spec(n_docs=12, vocab_size=30))
    synthetic.write_corpus(corpus, tmp_path)
    names = [
        line.partition("  ")[2].strip()
        for line in (tmp_path / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
    ]
    assert names == sorted(names)


def test_the_written_spec_records_the_constructed_cases_it_planted(tmp_path: Path) -> None:
    """An experiment finds the twins and duplicates by reading this rather than
    searching for them, so the record has to match what was generated."""
    spec = _spec(n_docs=16, vocab_size=30, n_twin_pairs=2, n_exact_duplicates=2)
    corpus = synthetic.generate(spec)
    synthetic.write_corpus(corpus, tmp_path)
    written = json.loads((tmp_path / "spec.json").read_text(encoding="utf-8"))

    assert written["spec_digest"] == corpus.spec.digest()
    assert written["n_documents"] == corpus.n_documents
    assert len(written["twins"]) == len(corpus.twins)
    assert len(written["exact_duplicate_pairs"]) == len(corpus.exact_duplicate_pairs)


def test_writing_the_same_corpus_twice_produces_identical_bytes(tmp_path: Path) -> None:
    """The committed artefact must not depend on when it was written."""
    corpus = synthetic.generate(_spec(n_docs=12, vocab_size=30))
    first = synthetic.write_corpus(corpus, tmp_path / "a")
    second = synthetic.write_corpus(corpus, tmp_path / "b")
    assert first == second


def test_writing_into_a_directory_that_does_not_exist_creates_it(tmp_path: Path) -> None:
    corpus = synthetic.generate(_spec(n_docs=12, vocab_size=30))
    target = tmp_path / "deep" / "nested"
    synthetic.write_corpus(corpus, target)
    assert (target / "corpus.jsonl").is_file()


# ---------------------------------------------------------------------------
# Generation boundaries
# ---------------------------------------------------------------------------
def test_a_corpus_too_small_for_its_constructed_cases_is_refused() -> None:
    """The planted duplicates and twins consume documents, so a spec asking for
    more of them than it has room for is a caller error rather than a corpus
    with no ordinary documents in it."""
    with pytest.raises(ValueError, match="too small for"):
        synthetic.generate(_spec(n_docs=4, vocab_size=20, n_exact_duplicates=2, n_twin_pairs=2))


def test_a_non_integer_zipf_exponent_is_permitted_but_goes_through_the_platform() -> None:
    """The integer path is exact; a fractional exponent uses pow and therefore
    the platform libm, which is why the generated files rather than the spec
    become the artefact."""
    corpus = synthetic.generate(_spec(n_docs=12, vocab_size=30, zipf_exponent=1.2))
    assert corpus.n_documents == 12
    assert all(doc for doc in corpus.documents), "no document may come out empty"


def test_a_missing_jsonl_corpus_is_named_rather_than_raising_an_oserror(tmp_path) -> None:
    """Every failure out of the loader façade is a DataIntegrityError, so a
    caller catching that one type cannot be surprised by a FileNotFoundError."""
    with pytest.raises(DataIntegrityError, match="corpus file not found"):
        load_jsonl_corpus(tmp_path / "absent.jsonl")


def test_the_jsonl_prefix_reaches_the_same_loader_as_calling_it_directly(tmp_path) -> None:
    """`jsonl:<path>` is the documented escape hatch from the registered names,
    and it must not be a second, divergent parse."""
    source = load_dataset("synthetic_tiny")
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in source.records), encoding="utf-8"
    )
    assert load_dataset(f"jsonl:{path}").digest() == load_jsonl_corpus(path).digest()


def test_the_doc_ids_of_a_loaded_corpus_are_its_records_in_order() -> None:
    """Downstream indexes documents by position and reports them by id, so the
    two orders being the same is what makes a reported rank meaningful."""
    data = load_dataset("synthetic_tiny")
    assert data.doc_ids == [str(r["doc_id"]) for r in data.records]
    assert len(data.doc_ids) == data.n_documents
    assert len(set(data.doc_ids)) == data.n_documents, "doc ids must identify a document"


def test_loading_movielens_through_the_facade_records_that_it_is_not_redistributable(
    tmp_path, monkeypatch
) -> None:
    """The provenance is the only place a reader learns that this result cannot
    be reproduced from the repository alone.

    The digest pin is repointed at the fixture archive rather than removed: the
    pin's own behaviour is tested separately, and what is under test here is the
    wiring from a parsed corpus into a LoadedDataset.
    """
    data = _archive()
    path = tmp_path / "ml-latest-small.zip"
    path.write_bytes(data)
    monkeypatch.setattr(
        movielens.load,
        "__kwdefaults__",
        {**movielens.load.__kwdefaults__, "expect_sha256": hash_bytes(data)},
    )

    loaded = load_dataset("movielens_small", archive=path)

    assert loaded.provenance["kind"] == "movielens"
    assert loaded.provenance["redistributable"] is False
    assert loaded.provenance["archive_sha256"] == hash_bytes(data)
    assert loaded.n_documents == 5
    assert loaded.interactions, "the ratings must survive the façade"


def test_a_whole_star_rating_written_with_a_zero_decimal_is_the_same_rating() -> None:
    """GroupLens writes 4.0, not 4. The doubled-integer domain has to treat the
    trailing zero as absent rather than as a fraction it cannot represent."""
    ratings = "userId,movieId,rating,timestamp\n1,1,4.00,964982703\n1,2,4,964981247\n"
    corpus = movielens.parse_archive(_archive(ratings=ratings))
    assert {w for _, _, w in corpus.interactions} == {4.0}


# ---------------------------------------------------------------------------
# The deterministic sampling primitives
# ---------------------------------------------------------------------------
# Every existing test of the generator asserts that the same spec produces the
# same corpus. A corrupted sampler is still deterministic, so that property
# cannot detect one: mutation testing broke the binary search, inverted the Zipf
# division and shifted the inclusive bounds, and the suite stayed green at 50%.
#
# These test the primitives directly. `_pick` and `_uniform_int` take their
# entropy from an injected `getrandbits`, so a stub for that -- and only that --
# drives the arithmetic through every value it can see.
class _FixedBits:
    """A `random.Random` stand-in whose draws are scripted.

    Only `getrandbits` is used by the code under test; `_pick` and
    `_uniform_int` are written against it precisely so the sampling does not
    depend on anything CPython does not promise to keep stable.
    """

    def __init__(self, *draws: int) -> None:
        self._draws = list(draws)

    def getrandbits(self, k: int) -> int:
        assert self._draws, "the code under test asked for more entropy than scripted"
        return self._draws.pop(0)


def test_the_zipf_weights_are_exact_integer_division_at_exponent_one() -> None:
    """Integers so the cumulative distribution and every comparison against it
    are exact, which is what makes the sampling reproducible without depending
    on how a float division rounds."""
    weights = synthetic._zipf_weights(5, 1.0)
    scale = synthetic._ZIPF_SCALE
    assert weights == [scale // 1, scale // 2, scale // 3, scale // 4, scale // 5]
    assert weights[0] > weights[1] > weights[2], "rank 0 is the most frequent"


def test_a_zipf_weight_never_falls_below_one() -> None:
    """The floor keeps every token reachable. A weight of zero would make its
    cumulative entry equal to its predecessor's, and the token could never be
    drawn -- silently shrinking the effective vocabulary.

    Reached with a steep exponent rather than a huge vocabulary: at exponent 1
    the scale is 2**40, so the quotient only reaches zero past a rank no corpus
    will ever have.
    """
    weights = synthetic._zipf_weights(8, 20.0)
    assert weights[0] == synthetic._ZIPF_SCALE, "rank 0 is untouched by the floor"
    assert min(weights) == 1
    assert all(w >= 1 for w in weights)
    assert weights[-1] == 1, "and the tail has bottomed out"


def test_a_fractional_exponent_still_falls_away_with_rank() -> None:
    """The platform-libm path. Its exact values are not pinned -- that is the
    point of the warning against it -- but it is still a Zipf law."""
    weights = synthetic._zipf_weights(6, 2.0)
    assert weights == sorted(weights, reverse=True)
    assert weights[0] > weights[-1]
    assert weights != synthetic._zipf_weights(6, 1.0), "the exponent has to matter"


def test_the_cumulative_distribution_is_the_running_total() -> None:
    assert synthetic._cumulative([3, 1, 4, 1]) == [3, 4, 8, 9]
    assert synthetic._cumulative([]) == []
    assert synthetic._cumulative([7]) == [7]


def test_picking_agrees_with_a_linear_scan_for_every_draw() -> None:
    """The binary search is the one piece of real algorithm here, and a corpus
    generated by a broken one still regenerates identically.

    Exhaustive over the whole draw space rather than sampled: the interesting
    values are the boundaries between adjacent cumulative entries, and there are
    few enough of them to check all of them.
    """
    cumulative = synthetic._cumulative([3, 1, 4, 1, 5])
    total = cumulative[-1]
    assert total == 14

    for draw in range(total):
        expected = bisect.bisect_right(cumulative, draw)
        assert synthetic._pick(_FixedBits(draw), cumulative) == expected, f"draw={draw}"


def test_picking_rejects_a_draw_at_or_above_the_total() -> None:
    """Rejection sampling is what makes the result uniform without a float
    division. A draw equal to the total is out of range, not the last index."""
    cumulative = [3, 4]
    # 4 and 5 are rejected, 2 is accepted and lands in the first bucket.
    assert synthetic._pick(_FixedBits(4, 5, 2), cumulative) == 0


def test_picking_from_a_single_bucket_always_returns_it() -> None:
    assert synthetic._pick(_FixedBits(0), [1]) == 0


def test_a_uniform_integer_covers_both_ends_of_its_range() -> None:
    """Inclusive at both ends, so the span is `high - low + 1`. Dropping the
    `+ 1` makes the top value unreachable, which for `_uniform_int(rng, 0, 4)`
    silently removes 5.0 from the interaction weights."""
    span_draws = list(range(5))
    assert [synthetic._uniform_int(_FixedBits(d), 10, 14) for d in span_draws] == [
        10,
        11,
        12,
        13,
        14,
    ]


def test_a_uniform_integer_rejects_a_draw_outside_its_span() -> None:
    """`bit_length` rounds up to a power of two, so draws above the span happen
    and have to be discarded rather than folded back in."""
    assert synthetic._uniform_int(_FixedBits(5, 6, 7, 3), 0, 4) == 3


def test_a_degenerate_range_returns_its_only_value() -> None:
    assert synthetic._uniform_int(_FixedBits(0), 7, 7) == 7


# ---------------------------------------------------------------------------
# What the generator builds out of them
# ---------------------------------------------------------------------------
def test_the_document_budget_counts_two_documents_for_every_twin_pair() -> None:
    """A twin pair is two documents, not one, so the budget is
    `n_docs - duplicates - 2 * pairs`. Counting one apiece lets a spec through
    that then produces more documents than it asked for."""
    # 10 - 2 - 2*4 = 0 base documents: one short.
    with pytest.raises(ValueError, match="too small for"):
        synthetic.generate(_spec(n_docs=10, vocab_size=30, n_exact_duplicates=2, n_twin_pairs=4))

    # One more document and it is exactly satisfiable.
    corpus = synthetic.generate(
        _spec(n_docs=11, vocab_size=30, n_exact_duplicates=2, n_twin_pairs=4)
    )
    assert corpus.n_documents == 11


def test_the_twin_token_is_picked_by_rank_so_its_frequency_is_intended() -> None:
    """`vocab_size // target_df - 1`, clamped into the vocabulary. Drawing from
    the Zipf head gives a high document frequency and from the tail a low one,
    so the rank is what makes the twin's separation tunable rather than
    incidental."""
    corpus = synthetic.generate(
        _spec(n_docs=20, vocab_size=64, n_twin_pairs=2, n_exact_duplicates=2)
    )
    by_id = dict(zip(corpus.doc_ids, corpus.documents, strict=True))

    assert corpus.twins, "the premise: twins were planted"
    for a_id, b_id, target_df in corpus.twins:
        a, b = by_id[a_id], by_id[b_id]
        assert len(b) == len(a) + 1, "b is a plus exactly one token"
        assert b[:-1] == a
        extra = b[-1]
        expected_rank = min(64 - 1, max(0, 64 // target_df - 1))
        assert extra == f"w{expected_rank:05d}"


def test_every_planted_duplicate_is_an_exact_copy_of_a_base_document() -> None:
    """They tie exactly, which is the tau = 0 baseline the whole tie-break study
    rests on. A source index off the end of the base documents would copy a
    duplicate or a twin instead."""
    spec = _spec(n_docs=20, vocab_size=40, n_exact_duplicates=3, n_twin_pairs=2)
    corpus = synthetic.generate(spec)
    by_id = dict(zip(corpus.doc_ids, corpus.documents, strict=True))
    n_base = 20 - 3 - 2 * 2

    assert len(corpus.exact_duplicate_pairs) == 3
    for source_id, copy_id in corpus.exact_duplicate_pairs:
        assert by_id[source_id] == by_id[copy_id]
        assert source_id.startswith("d"), "sources come from the base documents"
        assert int(source_id[1:]) < n_base


def test_the_attributes_stay_inside_the_ranges_the_generator_documents() -> None:
    """G8's exact pair: `rating_sum2` is twice a sum of 0.5-quantised ratings,
    so it is a product of two integers and never a float mean."""
    corpus = synthetic.generate(_spec(n_docs=40, vocab_size=60, n_users=5))
    assert len(corpus.attributes) == corpus.n_documents

    for row in corpus.attributes:
        assert 0 <= row["popularity"] <= 500
        assert 1 <= row["rating_count"] <= 20
        assert 0 <= row["engagement"] <= 50
        assert 2 * 1 <= row["rating_sum2"] <= 10 * 20
        assert isinstance(row["rating_sum2"], int), "no float mean is ever formed"


def test_interaction_weights_are_the_five_half_star_values_from_three_up() -> None:
    """`3.0 + u(0, 4) * 0.5`. Every one of the five has to be reachable: the
    interaction threshold is compared exactly, so a missing 5.0 would quietly
    change which interactions count as positive."""
    corpus = synthetic.generate(_spec(n_docs=60, vocab_size=80, n_users=60))
    weights = {w for _, _, w in corpus.interactions}
    assert weights == {3.0, 3.5, 4.0, 4.5, 5.0}


def test_a_user_interacts_at_least_once_and_never_more_than_the_cap() -> None:
    """`1 + _pick(...)` over a heavy-tailed distribution of length
    `max_interactions_per_user`, so the count is in `[1, cap]` -- and a user with
    no interactions at all would have no profile to build a query from."""
    cap = 6
    corpus = synthetic.generate(
        _spec(n_docs=40, vocab_size=60, n_users=40, max_interactions_per_user=cap)
    )
    per_user: dict[str, int] = {}
    for user, _, _ in corpus.interactions:
        per_user[user] = per_user.get(user, 0) + 1

    assert len(per_user) == 40, "every user interacted at least once"
    assert min(per_user.values()) >= 1
    assert max(per_user.values()) <= cap
    assert max(per_user.values()) > min(per_user.values()), "the tail is heavy, not flat"


def test_a_user_never_interacts_with_the_same_document_twice() -> None:
    """The candidate is redrawn into a list that rejects repeats, so an
    interaction list is a set of documents rather than a multiset."""
    corpus = synthetic.generate(_spec(n_docs=30, vocab_size=50, n_users=30))
    seen: set[tuple[str, str]] = set()
    for user, doc_id, _ in corpus.interactions:
        assert (user, doc_id) not in seen
        seen.add((user, doc_id))
    assert seen, "the corpus produced interactions at all"


def test_every_interaction_names_a_document_in_the_corpus() -> None:
    """The index is drawn over `len(doc_ids)`, which includes the duplicates and
    twins appended after the base documents."""
    corpus = synthetic.generate(_spec(n_docs=30, vocab_size=50, n_users=20))
    ids = set(corpus.doc_ids)
    assert {d for _, d, _ in corpus.interactions} <= ids
    assert any(d.startswith(("dup", "twin")) for _, d, _ in corpus.interactions), (
        "the constructed documents are reachable too, not just the base ones"
    )


def test_the_tiny_dataset_is_the_tiny_spec_and_not_the_default_one() -> None:
    """`synthetic_tiny` exists so tests and CI have a corpus that generates in
    under a second. Falling back to the default spec gives 2000 documents under
    the same name -- a slow suite and, worse, a different corpus behind a name
    that other results were recorded against."""
    data = load_dataset("synthetic_tiny")
    assert data.n_documents == 120
    assert len(data.doc_ids) == 120


def test_an_explicit_spec_overrides_the_registered_one() -> None:
    """The `spec` argument is the documented way to vary the generator without
    inventing a dataset name. Ignoring it would silently return the registered
    corpus, so an experiment sweeping the spec would sweep nothing."""
    custom = synthetic.SyntheticSpec(
        n_docs=30, vocab_size=60, n_users=3, n_exact_duplicates=2, n_twin_pairs=2
    )
    data = load_dataset("synthetic_tiny", spec=custom)
    assert data.n_documents == 30
    assert data.provenance["spec_digest"] == custom.digest()
    assert data.digest() != load_dataset("synthetic_tiny").digest()


# ---------------------------------------------------------------------------
# The generator as a fixed artefact
# ---------------------------------------------------------------------------
#: Digests of `synthetic_tiny` as this generator produces it. Recorded rather
#: than recomputed, because "the same spec gives the same corpus" is satisfied by
#: any deterministic generator including a broken one -- which is exactly what
#: mutation testing demonstrated, walking changes through the twin source index,
#: the attribute ranges and the interaction sampler with every other test green.
#:
#: Safe to pin: the generator draws only from `random.Random.getrandbits`, whose
#: Mersenne Twister stream CPython promises to keep stable, and every weight and
#: comparison on the way is integer. That is the reason the module is written the
#: way it is, and this is the assertion that holds it to it.
#:
#: If a deliberate change to the generator moves these, update them in the same
#: commit and re-run anything that quotes a number from this corpus. They are
#: meant to be a speed bump, not a wall.
_TINY_CORPUS_DIGEST = "bc098582b26cc92840efb7cf8ca97a56ff9d81cfa79ce74d05e07bdf17be75c2"
_TINY_SPEC_DIGEST = "f760dab267bfac16066245fc0aae8fdb3c5859fac7d798f1d7dd4d044e274e8a"
_TINY_INTERACTIONS_DIGEST = "7dae269fb86344965a833f583305947e9121188bc0d6fcb13026f478b80b4e00"


def test_the_tiny_corpus_is_byte_for_byte_what_it_has_always_been() -> None:
    """One assertion over the whole generation pipeline.

    Every constant, index and comparison in `generate` reaches this digest, so a
    change to any of them shows up here rather than as a quietly different
    corpus behind an unchanged name.
    """
    data = load_dataset("synthetic_tiny")
    assert data.digest() == _TINY_CORPUS_DIGEST
    assert data.provenance["spec_digest"] == _TINY_SPEC_DIGEST


def test_the_tiny_interaction_set_is_fixed_too() -> None:
    """Digested separately because the corpus digest is taken over the records
    alone -- deliberately, so two loads producing identical documents agree
    however they got there. The interactions therefore need their own.
    """
    data = load_dataset("synthetic_tiny")
    payload = canonical_json([list(i) for i in data.interactions], indent=None)
    assert hash_text(payload) == _TINY_INTERACTIONS_DIGEST
    assert len(data.interactions) == 67


def test_the_spec_defaults_are_the_ones_every_recorded_result_assumed() -> None:
    """The spec is hashed into every manifest, so its defaults are part of the
    identity of any run that did not override them. The Zipf exponent in
    particular selects between the exact integer path and the platform one --
    which the module docstring warns makes the files, not the spec, the artefact.
    """
    spec = synthetic.SyntheticSpec()
    assert spec.zipf_exponent == 1.0, "the exact integer path"
    assert spec.twin_extra_token_df == (1, 2, 4, 8, 16, 32, 64, 128)
    assert spec.len_min == 3
    assert spec.len_max == 40
    assert spec.max_interactions_per_user == 12


def test_the_twin_frequency_grid_doubles() -> None:
    """A grid of near-tie magnitudes rather than a single value: the extra
    token's document frequency is what controls how far a twin pair separates,
    so the grid spacing is the resolution of the whole near-tie study."""
    grid = synthetic.SyntheticSpec().twin_extra_token_df
    assert list(grid) == [2**i for i in range(len(grid))]
    assert len(grid) == 8


def test_an_inverted_document_length_range_is_refused_rather_than_hanging() -> None:
    """`_uniform_int` rejection-samples until a draw lands below `high - low + 1`.

    With `len_max < len_min` that bound is non-positive, `bit_length()` still
    returns 1, and every draw from `{0, 1}` fails the test -- so the generator
    spins forever. A hang is worse than a crash: it cannot be caught by a test
    that has to finish, and in CI it presents as a timeout with no stack.
    """
    with pytest.raises(ValueError, match="len_max=3 is below len_min=5"):
        synthetic.generate(_spec(n_docs=12, vocab_size=30, len_min=5, len_max=3))


@pytest.mark.parametrize("len_min", [0, -1])
def test_a_document_length_below_one_is_refused(len_min: int) -> None:
    """Zero reaches the same non-terminating loop through `_pick`, which is
    called `length` times; a negative one produces documents of no tokens at
    all, which the vocabulary builder would then reject far from the cause."""
    with pytest.raises(ValueError, match=f"len_min must be at least 1, got {len_min}"):
        synthetic.generate(_spec(n_docs=12, vocab_size=30, len_min=len_min))


def test_the_length_range_is_checked_before_the_document_budget() -> None:
    """Both are spec errors; the length range is the one that would hang, so it
    is reported first even when the budget is also wrong."""
    with pytest.raises(ValueError, match="len_max"):
        synthetic.generate(
            _spec(
                n_docs=1, vocab_size=30, n_exact_duplicates=4, n_twin_pairs=4, len_min=9, len_max=2
            )
        )


class _BoundedRandom:
    """A random source that refuses to be asked forever.

    The samplers below reject-and-redraw, so a degenerate range never terminates
    and a real `Random` would hang the suite rather than fail it. A sound call
    rejects before drawing at all, or succeeds within a handful of draws, so a
    draw past the limit is a fault. Local by house convention.
    """

    def __init__(self, limit: int = 64) -> None:
        self.calls = 0
        self.limit = limit

    def getrandbits(self, k: int) -> int:
        self.calls += 1
        if self.calls > self.limit:
            raise AssertionError(
                f"still drawing after {self.limit} attempts: the sampler accepted a "
                f"range it cannot satisfy, which against a real source is a hang"
            )
        return 0


@pytest.mark.parametrize(("low", "high"), [(5, 3), (5, 4), (0, -1), (-3, -9)])
def test_a_range_with_no_integer_in_it_is_refused_by_the_sampler_itself(
    low: int, high: int
) -> None:
    """The guard above protects `generate`; this protects `_uniform_int`.

    The tests above reach the sampler only through `generate`, which validates
    the one spec field feeding it. The sampler itself is safe only because every
    call site passes a range derived from an already-checked count, and that is
    an invariant a new caller can break.

    Both degenerate shapes are covered, because they fail to terminate for
    different reasons and a guard written for one can miss the other. At
    `span == 0`, `(0).bit_length()` is 0, so `getrandbits(0)` returns 0 and
    `0 < 0` is false. At `span < 0`, `(-1).bit_length()` is 1, so the draw is 0
    or 1 and never below -1. `(5, 4)` is the first shape; the rest are the
    second.

    Driven by `_BoundedRandom` so that a weakened guard fails here rather than
    hanging the suite.
    """
    source = _BoundedRandom()

    with pytest.raises(ValueError, match=rf"the range \[{low}, {high}\] contains no integer"):
        synthetic._uniform_int(source, low, high)

    assert source.calls == 0, "the range is refused before any draw is asked for"


def test_a_distribution_with_no_mass_is_refused_rather_than_sampled_forever() -> None:
    """`_pick`'s own version of the same hang, which no spec guard reaches.

    `_pick` rejection-samples below `cumulative[-1]`. At a total of zero the same
    `bit_length()` of 0 applies: `getrandbits(0)` is 0, `0 < 0` is false, and the
    loop never leaves. It is reachable whenever every Zipf weight rounds to zero,
    which the spec cannot rule out because the weights are computed rather than
    given. Bounded for the same reason as the test above.
    """
    source = _BoundedRandom()

    with pytest.raises(ValueError, match="carries no mass"):
        synthetic._pick(source, [0])

    assert source.calls == 0, "refused before any draw is asked for"


def test_an_empty_vocabulary_names_the_field_rather_than_indexing_off_the_end() -> None:
    """`vocab_size=0` is refused by name rather than by an `IndexError`.

    Unguarded it fails two calls down in `_cumulative`, as `IndexError: list
    index out of range`, which does not say which spec field was wrong. Every
    sibling guard here names its field.
    """
    with pytest.raises(ValueError, match="vocab_size must be at least 1, got 0"):
        synthetic.generate(_spec(n_docs=12, vocab_size=0))


# ---------------------------------------------------------------------------
# Parallel arrays: a short one is a loud failure, never a shortened corpus
# ---------------------------------------------------------------------------
# Both corpus types keep documents as several tuples indexed in step, and both
# zip them under `strict=True`. Without it `zip` stops at the shortest and the
# corpus silently loses its tail: `N` shrinks, and `N` is inside the idf of every
# term, so every weight in the corpus moves. The failure would surface as a set
# of plausible numbers rather than as an error.
def _movielens_corpus(*, n_texts: int = 2, n_attributes: int = 2) -> movielens.MovieLensCorpus:
    """Two documents, with the parallel arrays independently sized so a test can
    make one short. Local by house convention."""
    return movielens.MovieLensCorpus(
        archive_sha256="0" * 64,
        doc_ids=("m1", "m2"),
        texts=tuple(f"t{i}" for i in range(n_texts)),
        attributes=tuple({"popularity": i} for i in range(n_attributes)),
        interactions=(),
        n_ratings=0,
        n_users=0,
        n_unrated=2,
    )


def test_a_movielens_corpus_with_arrays_in_step_renders_every_document() -> None:
    """The passing side, so the refusals below are shown to be about the
    mismatch rather than about the constructor."""
    records = _movielens_corpus().records()

    assert [r["doc_id"] for r in records] == ["m1", "m2"]
    assert records[0]["text"] == "t0"
    assert records[0]["popularity"] == 0


@pytest.mark.parametrize(
    ("label", "kwargs"),
    [
        ("one text short", {"n_texts": 1}),
        ("one attribute short", {"n_attributes": 1}),
        ("one text too many", {"n_texts": 3}),
    ],
)
def test_a_movielens_corpus_whose_arrays_disagree_is_refused(
    label: str, kwargs: dict[str, int]
) -> None:
    """`strict=True`. Truncating instead would drop `m2` from a two-film corpus
    and report `n_documents = 2` beside a one-row record set."""
    with pytest.raises(ValueError, match="zip"):
        _movielens_corpus(**kwargs).records()


def _synthetic_corpus(*, n_documents: int = 2, n_attributes: int = 2) -> synthetic.SyntheticCorpus:
    """The generated corpus's counterpart, built by hand rather than generated so
    the arrays can be put out of step."""
    return synthetic.SyntheticCorpus(
        spec=synthetic.SyntheticSpec(),
        doc_ids=("d0", "d1"),
        documents=tuple(("w",) * (i + 1) for i in range(n_documents)),
        attributes=tuple({"popularity": i} for i in range(n_attributes)),
        interactions=(),
        twins=(),
        exact_duplicate_pairs=(),
    )


def test_a_synthetic_corpus_whose_attributes_run_short_is_refused() -> None:
    """The same guard on the generated side. A generator that planted fewer
    attributes than documents would otherwise emit a corpus one document
    shorter than the spec it records beside it."""
    with pytest.raises(ValueError, match="zip"):
        _synthetic_corpus(n_attributes=1).records()


def test_a_synthetic_corpus_whose_documents_run_short_is_refused() -> None:
    """`features_by_doc` pairs ids with documents and nothing else, so it has its
    own use of the same guard."""
    with pytest.raises(ValueError, match="zip"):
        _synthetic_corpus(n_documents=1).features_by_doc()


def test_a_synthetic_corpus_in_step_pairs_each_id_with_its_own_features() -> None:
    """The passing side, and the property the mapping exists for: the pairing is
    positional, so an off-by-one would silently attribute one document's tokens
    to its neighbour."""
    by_doc = _synthetic_corpus().features_by_doc()

    assert by_doc == {"d0": ("w",), "d1": ("w", "w")}


# ---------------------------------------------------------------------------
# The Zipf exponent selects between two arithmetics, and both ends must hold
# ---------------------------------------------------------------------------
def test_an_exponent_of_zero_is_a_flat_vocabulary_not_a_zipf_one() -> None:
    """`if exponent == 1.0` picks the exact integer path; everything else goes
    through `pow`. At exponent 0 that yields `scale / 1` for every rank -- a
    uniform vocabulary, the ablation with no skew at all.

    Pinned because the branch is on a float equality: routing exponent 0 into
    the integer path would silently substitute the 1/rank distribution and the
    ablation would measure the baseline twice.
    """
    flat = synthetic._zipf_weights(6, 0.0)

    assert len(set(flat)) == 1, "every rank equally likely"
    assert flat != synthetic._zipf_weights(6, 1.0)


def test_a_larger_exponent_concentrates_the_vocabulary_harder() -> None:
    """The parameter's direction, so the two arithmetics are shown to agree
    about what the exponent means. Only the head is fixed; every later rank
    falls away faster."""
    gentle = synthetic._zipf_weights(6, 1.0)
    steep = synthetic._zipf_weights(6, 2.0)

    assert gentle[0] == steep[0], "rank one is the scale itself either way"
    assert all(s < g for s, g in zip(steep[1:], gentle[1:], strict=True))


def test_the_exact_and_platform_arithmetics_agree_over_a_full_vocabulary() -> None:
    """Exponent 1 is computed in integers so the corpus regenerates identically
    on any interpreter; `pow` would reach the platform libm, which
    `docs/spec_addenda.md#g13` shows disagrees across systems.

    They happen to agree numerically at double the default vocabulary size, so
    the integer path is chosen for its provenance rather than for its answer --
    which is worth recording, since a reader comparing the two branches would
    otherwise wonder what the difference buys.
    """
    exact = synthetic._zipf_weights(8000, 1.0)
    via_pow = [max(1, int((1 << 40) / (rank + 1) ** 1.0)) for rank in range(8000)]

    assert exact == via_pow


def test_a_document_length_of_exactly_one_is_accepted() -> None:
    """The boundary the guard is written around. `len_min = 1` is the shortest
    document `_uniform_int` can be asked for and is a legitimate corpus: a
    one-token document has `L = 1`, so `tf = count / L` is exactly 1 and its
    weights are the idf alone.

    Rejecting it would refuse the very corpus a length sweep starts from.
    """
    corpus = synthetic.generate(_spec(n_docs=12, vocab_size=30, len_min=1, len_max=4))

    assert corpus.n_documents == 12
    assert min(len(d) for d in corpus.documents) >= 1
    assert all(d for d in corpus.documents), "no document is empty"


def test_a_single_length_range_is_accepted_and_fixes_every_document_length() -> None:
    """`len_max < len_min` is the guard; `len_max == len_min` is a degenerate
    but valid range, and a useful one -- with every document the same length,
    `tf = count / L` has a common denominator and a margin sweep isolates the
    document-frequency effect from the length effect.

    Twins carry one extra token by construction, so they are the one documented
    exception to the fixed length.
    """
    corpus = synthetic.generate(_spec(n_docs=12, vocab_size=30, len_min=5, len_max=5))
    twin_bs = {b for _, b, _ in corpus.twins}
    by_doc = corpus.features_by_doc()

    assert {len(by_doc[d]) for d in corpus.doc_ids if d not in twin_bs} == {5}
    assert all(len(by_doc[b]) == 6 for b in twin_bs), "a twin is its source plus one token"


def test_a_twin_token_too_frequent_for_the_vocabulary_falls_back_to_its_head() -> None:
    """`vocabulary[min(vocab_size - 1, max(0, vocab_size // target_df - 1))]`.

    The extra token is picked by Zipf rank so its document frequency is intended
    rather than incidental. Asking for a df the vocabulary cannot supply --
    `target_df` above `vocab_size`, so the rank computes to -1 -- clamps to rank
    zero, the most frequent term available and the closest the vocabulary gets
    to what was asked for.

    Without the lower clamp the rank is negative and Python reads from the far
    end, so the *rarest* term would stand in for the most frequent one and the
    twin pair would separate by the opposite of the intended amount.
    """
    spec = synthetic.SyntheticSpec(
        n_docs=40, vocab_size=30, n_users=0, n_exact_duplicates=2, n_twin_pairs=8
    )
    corpus = synthetic.generate(spec)
    by_doc = corpus.features_by_doc()
    extras = {df: by_doc[b][-1] for _, b, df in corpus.twins}

    assert extras[1] == "w00029", "df 1 wants the rarest term, at the tail"
    assert extras[16] == "w00000", "rank 30 // 16 - 1 = 0, the head, without any clamping"
    for exhausted in (32, 64, 128):
        assert extras[exhausted] == "w00000", f"df {exhausted} clamps to the head, not past it"


def test_the_twin_token_rank_is_clamped_at_the_rare_end_too() -> None:
    """The rank chosen for each target df, pinned per pair.

    The extra token is `vocabulary[min(V - 1, max(0, V // target_df - 1))]`.
    Asserting only that the token is somewhere in the vocabulary holds for any
    index, including a wrong one, so the whole expression goes unchecked.

    On a 30-word vocabulary the default df grid gives ranks 29, 14, 6, 2 and
    then 0 four times: once `target_df` exceeds `V`, `V // target_df` is 0 and
    the `max(0, ...)` clamp is what keeps the index off `vocabulary[-1]`, which
    is the head of the Zipf distribution and the opposite of the rare token the
    pair is meant to carry.

    The `min(V - 1, ...)` clamp never binds: `V // target_df - 1 <= V - 1` for
    every `target_df >= 1`, with equality at 1.
    """
    spec = synthetic.SyntheticSpec(
        n_docs=40, vocab_size=30, n_users=0, n_exact_duplicates=2, n_twin_pairs=8
    )
    corpus = synthetic.generate(spec)
    by_doc = corpus.features_by_doc()

    extras = [by_doc[b][-1] for _, b, _ in corpus.twins]

    assert extras == [
        "w00029",  # target_df 1   -> rank 29, the rarest token
        "w00014",  # target_df 2   -> rank 14
        "w00006",  # target_df 4   -> rank 6
        "w00002",  # target_df 8   -> rank 2
        "w00000",  # target_df 16  -> rank 0
        "w00000",  # target_df 32  -> 30 // 32 - 1 is -1, clamped to 0
        "w00000",  # target_df 64  -> likewise
        "w00000",  # target_df 128 -> likewise
    ]
    assert all(e in {f"w{i:05d}" for i in range(30)} for e in extras), "and every one is in range"


def test_find_near_ties_skips_exact_ties_unless_asked_for_them() -> None:
    """`strictly_positive` defaults to True, and the default is what every
    caller that does not think about it gets.

    G22 measured the regime: at tau = 1e-9 essentially every within-tau pair has
    a gap of exactly zero, and 17.2% of adjacent gaps are zero outright. A
    "closest pairs" list that included them would be a list of zeros on any real
    corpus -- it would report that exact ties exist, which is already known, and
    nothing about the near-tie structure section 7.4 is looking for.

    Both existing exact-tie tests pass the flag explicitly, so the default was
    the one path never taken.
    """
    scores = [1.0, 1.0, 0.5, 0.0, 0.0]

    by_default = synthetic.find_near_ties(scores, limit=10)
    assert all(p.gap > 0.0 for p in by_default)
    assert not any(p.is_exact for p in by_default)
    assert [p.gap for p in by_default] == [0.5, 0.5]

    asked_for = synthetic.find_near_ties(scores, limit=10, strictly_positive=False)
    assert sum(p.is_exact for p in asked_for) == 2, "the pairs the default dropped"


# ---------------------------------------------------------------------------
# scripts/fetch_data.py: a bad download must not destroy a good archive
# ---------------------------------------------------------------------------
# The archive is not re-obtainable: GroupLens replaces ml-latest-small.zip in
# place, so a contributor holding the pinned copy holds the only copy the
# published numbers were computed against. `_download` therefore verifies the
# digest before the rename, and `--force` cannot overwrite on a mismatch.
def _fetch_script() -> ModuleType:
    """scripts/fetch_data.py as a module. Local by house convention."""
    path = REPO / "scripts" / "fetch_data.py"
    spec = importlib.util.spec_from_file_location("fetch_data_under_test", path)
    assert spec is not None, "the script must be loadable as a module"
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Response:
    """The slice of urlopen's result that `_download` uses."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.headers = {"Content-Length": str(len(payload))}

    def read(self, size: int) -> bytes:
        chunk, self._payload = self._payload[:size], self._payload[size:]
        return chunk

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _serve(monkeypatch: pytest.MonkeyPatch, fetch: ModuleType, payload: bytes) -> None:
    monkeypatch.setattr(
        fetch.urllib.request, "urlopen", lambda request: _Response(payload), raising=True
    )


def test_a_download_that_fails_the_pin_leaves_the_existing_archive_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The data-loss case: the digest is verified before the rename.

    The bytes at `dest` before the call are the pinned archive; the bytes served
    are what upstream now returns. The pinned copy must survive.
    """
    fetch = _fetch_script()
    dest = tmp_path / "ml-latest-small.zip"
    dest.write_bytes(b"the pinned archive, not obtainable again")
    _serve(monkeypatch, fetch, b"what upstream serves today")

    digest, placed = fetch._download("https://example.invalid/ml.zip", dest, "0" * 64)

    assert placed is False, "a download that fails the pin must not be moved into place"
    assert dest.read_bytes() == b"the pinned archive, not obtainable again", (
        "the archive the published numbers were computed against must survive a bad download"
    )
    assert digest == hashlib.sha256(b"what upstream serves today").hexdigest()


def test_the_rejected_download_is_kept_rather_than_discarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rejected bytes are kept rather than deleted.

    An upstream reissue and a proxy serving an error page are told apart by
    their content, so the mismatching download is preserved for inspection.
    """
    fetch = _fetch_script()
    dest = tmp_path / "ml-latest-small.zip"
    dest.write_bytes(b"pinned")
    _serve(monkeypatch, fetch, b"different")

    fetch._download("https://example.invalid/ml.zip", dest, "0" * 64)

    rejected = tmp_path / "ml-latest-small.zip.rejected"
    assert rejected.read_bytes() == b"different"
    assert not (tmp_path / "ml-latest-small.zip.partial").exists(), "no partial is left behind"


def test_a_download_matching_the_pin_is_placed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contrastive with the two above, so neither passes by refusing everything."""
    fetch = _fetch_script()
    dest = tmp_path / "ml-latest-small.zip"
    payload = b"the archive upstream still serves"
    _serve(monkeypatch, fetch, payload)

    digest, placed = fetch._download(
        "https://example.invalid/ml.zip", dest, hashlib.sha256(payload).hexdigest()
    )

    assert placed is True
    assert dest.read_bytes() == payload
    assert digest == hashlib.sha256(payload).hexdigest()


def test_an_unpinned_download_is_placed_so_the_first_fetch_can_pin_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`MOVIELENS_SHA256 = None` is the documented first-download case.

    There is nothing to verify against yet and the script prints the digest to
    pin, so that path must still deliver a file.
    """
    fetch = _fetch_script()
    dest = tmp_path / "ml-latest-small.zip"
    _serve(monkeypatch, fetch, b"first ever download")

    digest, placed = fetch._download("https://example.invalid/ml.zip", dest, None)

    assert placed is True
    assert dest.read_bytes() == b"first ever download"
    assert digest == hashlib.sha256(b"first ever download").hexdigest()
