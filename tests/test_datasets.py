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

import io
import json
import random as random_module
import zipfile
from itertools import pairwise

import pytest

from tfidf_stability.datasets import movielens, synthetic
from tfidf_stability.datasets.loaders import load_dataset, load_jsonl_corpus
from tfidf_stability.preprocessing.pipeline import PreprocessingPipeline
from tfidf_stability.ranking.attributes import AttributeTable
from tfidf_stability.similarity.cosine import cosine_against_corpus
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
