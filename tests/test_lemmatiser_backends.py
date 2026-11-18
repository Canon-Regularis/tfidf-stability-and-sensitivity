"""Tests for the three lemmatiser backends in `preprocessing/lemmatise.py`.

The backend decides what a token becomes before it reaches a vocabulary, so it
is part of a run's identity. Every backend passes the gap sentinel from
`tokenise.py` through untouched, and `make_lemmatiser` refuses the lookup kind.
"""

from __future__ import annotations

import unicodedata

import pytest

from tfidf_stability.preprocessing.lemmatise import (
    IdentityLemmatiser,
    LemmatiserKind,
    LookupLemmatiser,
    Porter2Stemmer,
    lemmatiser_identity,
    make_lemmatiser,
    porter2_stem,
)
from tfidf_stability.preprocessing.tokenise import GAP


# ---------------------------------------------------------------------------
# The identity backend: the ablation baseline
# ---------------------------------------------------------------------------
def test_the_identity_backend_returns_every_token_unchanged() -> None:
    lemmatiser = IdentityLemmatiser()
    for token in ("running", "cats", "happy", "", "été"):
        assert lemmatiser(token) == token


def test_the_identity_backend_leaves_a_token_porter2_would_have_stemmed() -> None:
    """The identity backend is an ablation, so it must differ from Porter2."""
    assert porter2_stem("running") != "running"
    assert IdentityLemmatiser()("running") == "running"


def test_the_identity_backend_preserves_the_gap_sentinel() -> None:
    tokens = ["king", GAP, "pop"]
    assert IdentityLemmatiser().apply(tokens) == tokens


# ---------------------------------------------------------------------------
# The lookup backend: a recorded mapping, with a fallback
# ---------------------------------------------------------------------------
def test_a_table_hit_returns_the_recorded_lemma() -> None:
    lemmatiser = LookupLemmatiser({"mice": "mouse", "geese": "goose"})
    assert lemmatiser("mice") == "mouse"
    assert lemmatiser("geese") == "goose"


def test_a_table_miss_falls_through_to_the_secondary_backend() -> None:
    """A table miss passes the token through the default identity fallback."""
    lemmatiser = LookupLemmatiser({"mice": "mouse"})
    assert lemmatiser("running") == "running", "a miss must not vanish"


def test_an_explicit_fallback_is_used_for_the_tokens_the_table_does_not_cover() -> None:
    """A table for the irregulars and a stemmer fallback cover every token."""
    lemmatiser = LookupLemmatiser({"mice": "mouse"}, fallback=Porter2Stemmer())
    assert lemmatiser("mice") == "mouse", "the table wins where it has an entry"
    assert lemmatiser("running") == porter2_stem("running"), "and the stemmer takes the rest"


def test_the_table_wins_over_the_fallback_even_where_both_have_an_answer() -> None:
    """The recorded mapping is normative and overrides the fallback."""
    # Porter2 is a suffix stripper and leaves the irregular "better" alone, so
    # the table is the only source of the lemma "good".
    lemmatiser = LookupLemmatiser({"better": "good"}, fallback=Porter2Stemmer())
    assert porter2_stem("better") == "better", "the premise: the stemmer has no answer"
    assert lemmatiser("better") == "good", "the recorded mapping is normative, not advisory"


def test_the_lookup_backend_preserves_the_gap_sentinel_rather_than_looking_it_up() -> None:
    """The sentinel is passed through even when the table has an entry for it.

    The sentinel marks a removed stopword; rewriting it would let an n-gram
    span the hole that tokenise.py opened.
    """
    lemmatiser = LookupLemmatiser({GAP: "SHOULD_NEVER_BE_USED", "mice": "mouse"})
    assert lemmatiser.apply(["mice", GAP, "geese"]) == ["mouse", GAP, "geese"]


def test_an_empty_table_degrades_to_the_fallback_for_everything() -> None:
    lemmatiser = LookupLemmatiser({}, fallback=Porter2Stemmer())
    assert lemmatiser.apply(["running", "cats"]) == [porter2_stem("running"), porter2_stem("cats")]


# ---------------------------------------------------------------------------
# The factory
# ---------------------------------------------------------------------------
def test_the_factory_builds_the_two_backends_that_need_no_configuration() -> None:
    assert isinstance(make_lemmatiser(LemmatiserKind.NONE), IdentityLemmatiser)
    assert isinstance(make_lemmatiser(LemmatiserKind.PORTER2), Porter2Stemmer)


def test_the_factory_accepts_the_string_form_a_config_file_carries() -> None:
    """The YAML string form and the enum member select the same backend."""
    assert isinstance(make_lemmatiser("none"), IdentityLemmatiser)
    assert isinstance(make_lemmatiser("porter2"), Porter2Stemmer)


def test_the_default_backend_is_porter2() -> None:
    assert isinstance(make_lemmatiser(), Porter2Stemmer)


def test_the_lookup_backend_is_refused_rather_than_built_with_an_empty_table() -> None:
    """The factory refuses lookup rather than build it with an empty table.

    An empty table degrades silently to the identity backend, and no feature
    would be lemmatised.
    """
    with pytest.raises(ValueError, match="needs an explicit table"):
        make_lemmatiser(LemmatiserKind.LOOKUP)


def test_an_unknown_backend_name_is_refused_by_the_enum() -> None:
    with pytest.raises(ValueError, match="not_a_backend"):
        make_lemmatiser("not_a_backend")


# ---------------------------------------------------------------------------
# Every backend names itself, because the name reaches the manifest
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        (IdentityLemmatiser(), "none"),
        (Porter2Stemmer(), "porter2"),
        (LookupLemmatiser({}), "lookup"),
    ],
)
def test_each_backend_reports_the_name_the_manifest_records(backend: object, expected: str) -> None:
    assert backend.name == expected  # type: ignore[attr-defined]


def test_the_three_backend_names_are_distinct() -> None:
    """Distinct names keep the three preprocessing maps apart in a manifest."""
    names = {IdentityLemmatiser().name, Porter2Stemmer().name, LookupLemmatiser({}).name}
    assert len(names) == 3


# ---------------------------------------------------------------------------
# Porter2 memoises, and the cache must not change the answer
# ---------------------------------------------------------------------------
def test_stemming_the_same_token_twice_returns_the_same_answer_from_the_cache() -> None:
    """The per-instance memo returns the same answer on the second call.

    A cache that returned something else would make a document's features
    depend on how often a token had already been seen in the run.
    """
    stemmer = Porter2Stemmer()
    first = stemmer("running")
    second = stemmer("running")
    assert first == second == porter2_stem("running")


def test_a_cached_and_an_uncached_instance_agree() -> None:
    warmed = Porter2Stemmer()
    warmed("running")
    assert warmed("running") == Porter2Stemmer()("running")


def test_the_porter2_backend_preserves_the_gap_sentinel() -> None:
    """The sentinel is not a word and is passed through unstemmed.

    Stemming it would let an n-gram span the hole a removed stopword left.
    """
    assert Porter2Stemmer().apply(["running", GAP, "cats"]) == [
        porter2_stem("running"),
        GAP,
        porter2_stem("cats"),
    ]


def test_every_backend_applies_elementwise_over_an_empty_token_list() -> None:
    for backend in (IdentityLemmatiser(), Porter2Stemmer(), LookupLemmatiser({})):
        assert backend.apply([]) == []


# ---------------------------------------------------------------------------
# Identity: the name says which backend, not which instance of it
# ---------------------------------------------------------------------------


# `test_the_three_backend_names_are_distinct` above states the invariant a run
# manifest depends on across the three kinds. Within one kind `name` is a class
# attribute on `LookupLemmatiser`, whose answers come from a table handed in at
# construction, so the name alone cannot identify an instance.
def test_two_lookup_tables_that_disagree_have_different_identities() -> None:
    """Two lookup instances with different tables have different identities.

    Both report ``name == "lookup"``. The table is what produces the features,
    so the identity a manifest records has to differ with it.
    """
    cat = LookupLemmatiser({"cats": "cat"})
    feline = LookupLemmatiser({"cats": "feline"})

    assert cat("cats") != feline("cats"), "the premise: these produce different features"
    assert cat.name == feline.name == "lookup", "and the name cannot tell them apart"
    assert lemmatiser_identity(cat) != lemmatiser_identity(feline)


def test_the_fallback_is_part_of_the_identity_and_not_only_the_table() -> None:
    """The fallback is part of the identity, not only the table.

    On a real corpus most tokens miss the table, so the fallback decides most
    features and two instances that differ only in fallback differ in identity.
    """
    identical_table = {"cats": "cat"}
    passthrough = LookupLemmatiser(identical_table)
    stemming = LookupLemmatiser(identical_table, fallback=Porter2Stemmer())

    assert passthrough("running") != stemming("running"), (
        "the premise: a table miss is handled differently"
    )
    assert passthrough("cats") == stemming("cats") == "cat", "while a hit is not"
    assert lemmatiser_identity(passthrough) != lemmatiser_identity(stemming)


def test_the_same_table_gives_the_same_identity_whatever_order_it_was_built_in() -> None:
    """The identity covers the sorted mapping, not the insertion order.

    A digest over `dict` insertion order would make a manifest depend on how
    the table's file was read, so two runs of one experiment could disagree.
    """
    forwards = LookupLemmatiser({"cats": "cat", "mice": "mouse", "geese": "goose"})
    backwards = LookupLemmatiser({"geese": "goose", "mice": "mouse", "cats": "cat"})

    assert list(forwards._table) != list(backwards._table), "the premise: insertion order differs"
    assert lemmatiser_identity(forwards) == lemmatiser_identity(backwards)


def test_a_backend_whose_class_fixes_its_output_is_identified_by_name_alone() -> None:
    """A backend with no configuration is identified by its name alone.

    `IdentityLemmatiser` and `Porter2Stemmer` take no arguments that can change
    an answer (Porter2's memo caches a pure function), so neither carries a
    `digest`. Qualifying their identity would move every recorded digest.
    """
    for backend, expected in ((IdentityLemmatiser(), "none"), (Porter2Stemmer(), "porter2")):
        assert lemmatiser_identity(backend) == expected
        assert not hasattr(backend, "digest"), f"{expected} needs no content digest"

    assert lemmatiser_identity(LookupLemmatiser({})).startswith("lookup:"), (
        "while the backend that does need one says so in its identity"
    )


def test_an_identity_is_stable_across_two_constructions_of_the_same_table() -> None:
    """The digest does not depend on per-instance state such as an address.

    Two processes fitting the same experiment must agree on the identity.
    """
    first = LookupLemmatiser({"cats": "cat"}, fallback=Porter2Stemmer())
    second = LookupLemmatiser({"cats": "cat"}, fallback=Porter2Stemmer())

    assert first is not second
    assert lemmatiser_identity(first) == lemmatiser_identity(second)


# ---------------------------------------------------------------------------
# The identity's serialisation is a contract, not an implementation detail
# ---------------------------------------------------------------------------

# The tests above compare identities against each other, so they hold under any
# canonical form. The literals below pin the form itself: a failure means the
# canonical form moved, and every recorded identity moves with it.
# Rationale: docs/engineering_notes.md#lookup-lemmatiser-digest-literals
_PINNED_IDENTITIES = {
    "empty table": (
        LookupLemmatiser({}),
        "f4cf92413f98e0975b55412da824873aaa8ae1572901360fa8e83ca98afcc62e",
    ),
    "ascii table": (
        LookupLemmatiser({"cats": "cat", "mice": "mouse"}),
        "711ea4ce892d7c4ff0c32863ead450c4ebf833156c7aa5960771d59fa9bdc876",
    ),
    # The non-ASCII case is the only one `ensure_ascii` is visible from: under
    # `ensure_ascii=True` the table serialises as escape sequences and gives a
    # different digest for the same mapping.
    "non-ascii table": (
        LookupLemmatiser({"café": "cafe", "naïve": "naive"}),
        "efa0acd32e054aa157801d0207e75a6460eb201dc96977d5939bf9ad3dfd25d7",
    ),
    "table with a stemming fallback": (
        LookupLemmatiser({"cats": "cat"}, fallback=Porter2Stemmer()),
        "a9add8e75a9c513543a761fafd61c624c78dcb659c8645c03fb6d873bdfbd765",
    ),
}


def test_the_recorded_identity_of_a_known_table_is_the_value_this_repository_pins() -> None:
    """Each pinned table digests to the literal recorded above.

    The count is asserted outside the loop, so a case dropped from the table
    fails rather than shrinking the sweep.
    """
    checked = 0
    for name, (lemmatiser, expected) in _PINNED_IDENTITIES.items():
        assert lemmatiser.digest == expected, f"{name}: the canonical form moved"
        assert lemmatiser_identity(lemmatiser) == f"lookup:{expected}", f"{name}: identity"
        checked += 1

    assert checked == 4, "every pinned case was exercised"


def test_a_table_holding_the_same_mapping_written_differently_digests_the_same() -> None:
    """Insertion order is not part of the digest.

    Two runs reading the same table file in a different order are the same
    experiment, so the digest is taken with `sort_keys=True`.
    """
    forwards = LookupLemmatiser({"cats": "cat", "mice": "mouse"})
    backwards = LookupLemmatiser({"mice": "mouse", "cats": "cat"})

    assert list(forwards._table) != list(backwards._table), "the premise: order differs"
    assert forwards.digest == backwards.digest
    assert forwards.digest == _PINNED_IDENTITIES["ascii table"][1]


def test_a_non_ascii_table_is_digested_as_its_characters_rather_than_as_escapes() -> None:
    """The digest serialises the table with `ensure_ascii=False`.

    NFC and NFD spellings of one key are different mappings and digest
    differently under either flag; normalisation is the pipeline's job
    upstream. The pinned literal is what detects `ensure_ascii=True`.
    """
    composed = unicodedata.normalize("NFC", "café")
    decomposed = unicodedata.normalize("NFD", "café")

    assert composed != decomposed, "the premise: two spellings of one word"
    assert LookupLemmatiser({composed: "cafe"}).digest != (
        LookupLemmatiser({decomposed: "cafe"}).digest
    )
    assert (
        LookupLemmatiser({composed: "cafe", "naïve": "naive"}).digest
        == (_PINNED_IDENTITIES["non-ascii table"][1])
    ), "and the pinned case is the composed form, as written in this file"
