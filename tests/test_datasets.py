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
import io
import json
import random as random_module
import zipfile
from itertools import pairwise
from pathlib import Path

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
