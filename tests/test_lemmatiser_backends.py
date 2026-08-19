"""The three lemmatiser backends, and what choosing one commits you to.

`preprocessing/lemmatise.py` decides what a token becomes before it ever reaches
a vocabulary, so the backend is part of the run's identity: swapping it changes
every feature, every df, and therefore every published number. Only the Porter2
path was exercised. The identity backend existed as an ablation nobody ran, and
the lookup backend, whose whole point is that the mapping is recorded provenance
rather than an ambient property of the machine, had never been constructed.

Two properties carry the weight.

The gap sentinel survives every backend. `tokenise.py` inserts a sentinel where a
stopword was removed so an n-gram cannot span the hole, and a lemmatiser that
stemmed the sentinel would destroy that. Each backend's `apply` has to pass it
through untouched, and only Porter2's did so under test.

A backend that needs configuration says so rather than guessing. `make_lemmatiser`
builds the two that are self-contained and refuses the lookup backend, because
inventing an empty table would silently degrade it to the identity backend and
every feature would quietly stop being lemmatised.
"""

from __future__ import annotations

import pytest

from tfidf_stability.preprocessing.lemmatise import (
    IdentityLemmatiser,
    LemmatiserKind,
    LookupLemmatiser,
    Porter2Stemmer,
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
    """The premise of having it as an ablation: if the two agreed there would be
    nothing to ablate."""
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
    """The default fallback is the identity backend, so a miss is a pass-through
    rather than a dropped token."""
    lemmatiser = LookupLemmatiser({"mice": "mouse"})
    assert lemmatiser("running") == "running", "a miss must not vanish"


def test_an_explicit_fallback_is_used_for_the_tokens_the_table_does_not_cover() -> None:
    """Table for the irregulars, stemmer for the rest: the configuration the
    class exists to support."""
    lemmatiser = LookupLemmatiser({"mice": "mouse"}, fallback=Porter2Stemmer())
    assert lemmatiser("mice") == "mouse", "the table wins where it has an entry"
    assert lemmatiser("running") == porter2_stem("running"), "and the stemmer takes the rest"


def test_the_table_wins_over_the_fallback_even_where_both_have_an_answer() -> None:
    """Otherwise the recorded mapping would be advisory rather than normative."""
    # "better" is a true lemma the stemmer cannot reach: Porter2 is a suffix
    # stripper, so it leaves irregular comparatives alone. That is exactly the
    # gap a lookup table exists to fill.
    lemmatiser = LookupLemmatiser({"better": "good"}, fallback=Porter2Stemmer())
    assert porter2_stem("better") == "better", "the premise: the stemmer has no answer"
    assert lemmatiser("better") == "good", "the recorded mapping is normative, not advisory"


def test_the_lookup_backend_preserves_the_gap_sentinel_rather_than_looking_it_up() -> None:
    """A table containing the sentinel must not be consulted for it: the sentinel
    marks a removed stopword, and rewriting it would let an n-gram span the hole
    that tokenise.py opened precisely to prevent that."""
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
    """These names come out of YAML, so the string and the member must select
    the same backend or a config would build something else."""
    assert isinstance(make_lemmatiser("none"), IdentityLemmatiser)
    assert isinstance(make_lemmatiser("porter2"), Porter2Stemmer)


def test_the_default_backend_is_porter2() -> None:
    assert isinstance(make_lemmatiser(), Porter2Stemmer)


def test_the_lookup_backend_is_refused_rather_than_built_with_an_empty_table() -> None:
    """An invented empty table would degrade silently to the identity backend,
    and every feature would quietly stop being lemmatised."""
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
    """A shared name would make two different preprocessing maps indistinguishable
    in a run manifest."""
    names = {IdentityLemmatiser().name, Porter2Stemmer().name, LookupLemmatiser({}).name}
    assert len(names) == 3


# ---------------------------------------------------------------------------
# Porter2 memoises, and the cache must not change the answer
# ---------------------------------------------------------------------------
def test_stemming_the_same_token_twice_returns_the_same_answer_from_the_cache() -> None:
    """The stemmer memoises per instance. A cache that returned something else
    on the second call would make a document's features depend on how often a
    token had already been seen in the run.
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
    """The sentinel is not a word, and stemming it would let an n-gram span the
    hole a removed stopword left."""
    assert Porter2Stemmer().apply(["running", GAP, "cats"]) == [
        porter2_stem("running"),
        GAP,
        porter2_stem("cats"),
    ]


def test_every_backend_applies_elementwise_over_an_empty_token_list() -> None:
    for backend in (IdentityLemmatiser(), Porter2Stemmer(), LookupLemmatiser({})):
        assert backend.apply([]) == []
