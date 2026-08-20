"""The preprocessing map is deterministic and fixed (README sections 2 and 3).

Section 3 promises "all stages of the pipeline are deterministic given a fixed
corpus, configuration, and software environment". Tested at the stricter
reading: stability within one run is too weak, so the map must also be stable
across processes and hash seeds. Otherwise the perturbation experiments would
be measuring interpreter noise.
"""

from __future__ import annotations

import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tfidf_stability.preprocessing.lemmatise import (
    IdentityLemmatiser,
    LemmatiserKind,
    make_lemmatiser,
    porter2_stem,
)
from tfidf_stability.preprocessing.ngrams import (
    JOINER,
    generate_ngrams,
    iter_gap_free,
    ngram_order,
    split_ngram,
)
from tfidf_stability.preprocessing.normalise import NormalisationConfig, normalise
from tfidf_stability.preprocessing.pipeline import (
    PreprocessingConfig,
    PreprocessingPipeline,
    preprocess_all,
)
from tfidf_stability.preprocessing.stopwords import (
    StopwordSet,
    load_stopwords,
    remove_stopwords,
)
from tfidf_stability.preprocessing.tokenise import (
    GAP,
    TokenisationConfig,
    tokenise,
    tokenise_with_offsets,
)
from tfidf_stability.utils.validation import DataIntegrityError

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(st.text(max_size=200))
def test_normalisation_is_idempotent(text: str) -> None:
    """Otherwise the map depends on how many times it has been applied, which is
    an irreproducibility that is hard to locate."""
    once = normalise(text)
    assert normalise(once) == once


@pytest.mark.property
@given(st.text(max_size=200))
def test_normalisation_output_is_nfkc(text: str) -> None:
    out = normalise(text)
    assert unicodedata.normalize("NFKC", out) == out


def test_normalisation_folds_compatibility_forms() -> None:
    assert normalise("ﬁle") == "file"  # ligature
    assert normalise("Ｆｕｌｌ") == "full"  # full-width
    assert normalise("①②③") == "123"  # circled digits
    assert normalise("  a   b  ") == "a b"  # whitespace collapse
    assert normalise("a\u200bb") == "ab"  # zero-width space removed


# ---------------------------------------------------------------------------
# Porter2 against the official Snowball vectors
# ---------------------------------------------------------------------------
def test_porter2_matches_official_snowball_vectors(snowball_vectors) -> None:  # type: ignore[no-untyped-def]
    """The strongest available check on the stemmer: 42 649 published word pairs.

    Justifies vendoring the generated implementation instead of hand-porting it,
    and catches any future divergence between the Python and C++ backends.
    """
    voc, out = snowball_vectors
    assert len(voc) == len(out) > 40_000
    bad = [(w, e, porter2_stem(w)) for w, e in zip(voc, out, strict=True) if porter2_stem(w) != e]
    assert not bad, f"{len(bad)} mismatches, first five: {bad[:5]}"


def test_vendored_snowball_matches_its_recorded_digests() -> None:
    """Provenance check: the vendored bytes are the ones we recorded."""
    import hashlib

    d = REPO / "src" / "tfidf_stability" / "preprocessing" / "_snowball"
    manifest = (d / "MANIFEST.sha256").read_text(encoding="utf-8")
    checked = 0
    for line in manifest.splitlines():
        if not line or line.startswith("#"):
            continue
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == digest, name
        checked += 1
    assert checked == 3


def test_stemming_is_a_pure_function() -> None:
    """Memoisation must not be observable."""
    s = make_lemmatiser(LemmatiserKind.PORTER2)
    first = [s(w) for w in ("running", "running", "flies", "running")]
    assert first == [s(w) for w in ("running", "running", "flies", "running")]
    assert first[0] == first[1] == first[3]


def test_identity_lemmatiser_is_the_identity() -> None:
    assert IdentityLemmatiser().apply(["running", "flies"]) == ["running", "flies"]


# ---------------------------------------------------------------------------
# N-grams (spec_addenda G7)
# ---------------------------------------------------------------------------
def test_ngrams_never_span_a_removed_stopword() -> None:
    """ "king of pop" must not manufacture the bigram "king pop"."""
    tokens = ["king", GAP, "pop"]
    assert generate_ngrams(tokens, 1, 2) == ["king", "pop"]
    assert f"king{JOINER}pop" not in generate_ngrams(tokens, 1, 2)


def test_ngrams_may_span_gaps_when_explicitly_asked() -> None:
    got = generate_ngrams(["king", GAP, "pop"], 1, 2, cross_gaps=True)
    assert got == ["king", "pop", f"king{JOINER}pop"]
    assert GAP not in got, "the sentinel is a boundary marker, never a feature"


def test_ngram_encoding_is_injective_and_reversible() -> None:
    """The joiner cannot occur inside a token, so n-grams decode unambiguously."""
    bg = generate_ngrams(["new", "york", "city"], 2, 2)
    assert bg == [f"new{JOINER}york", f"york{JOINER}city"]
    assert split_ngram(bg[0]) == ["new", "york"]
    assert ngram_order(bg[0]) == 2
    assert ngram_order("solo") == 1


@pytest.mark.property
@given(st.lists(st.text(alphabet="abcdef", min_size=1, max_size=4), max_size=8))
def test_ngram_round_trip(tokens: list[str]) -> None:
    for n in generate_ngrams(tokens, 1, 3):
        assert JOINER.join(split_ngram(n)) == n


def test_ngram_order_range_is_validated() -> None:
    with pytest.raises(ValueError, match="n_min"):
        generate_ngrams(["a"], 0, 2)
    with pytest.raises(ValueError, match="at least n_min"):
        generate_ngrams(["a"], 3, 2)


# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------
def test_stopword_asset_loads_and_is_byte_sorted() -> None:
    sw = load_stopwords()
    assert len(sw) > 100
    words = list(sw)
    assert words == sorted(words)
    assert "the" in sw
    assert "quick" not in sw
    assert len(sw.digest) == 64


def test_stopword_removal_collapses_runs_and_trims_edges() -> None:
    """Canonical output: two inputs differing only in stopword runs agree."""
    sw = load_stopwords()
    a = remove_stopwords(tokenise(normalise("the the the cat the the")), sw)
    b = remove_stopwords(tokenise(normalise("the cat the")), sw)
    assert a == b == ["cat"]


def test_stopword_removal_can_drop_instead_of_gapping() -> None:
    sw = load_stopwords()
    got = remove_stopwords(tokenise(normalise("king of pop")), sw, insert_gaps=False)
    assert got == ["king", "pop"]


# ---------------------------------------------------------------------------
# The whole map
# ---------------------------------------------------------------------------
def test_pipeline_is_repeatable(pipeline: PreprocessingPipeline) -> None:
    text = "The King of Pop was running through New York City."
    assert pipeline.preprocess(text) == pipeline.preprocess(text)


def test_pipeline_is_independent_of_document_order(
    pipeline: PreprocessingPipeline, mini_corpus
) -> None:  # type: ignore[no-untyped-def]
    docs = [(str(d["doc_id"]), str(d["text"])) for d in mini_corpus]
    forward = {d.doc_id: d.features for d in pipeline.preprocess_corpus(docs)}
    backward = {d.doc_id: d.features for d in pipeline.preprocess_corpus(list(reversed(docs)))}
    assert forward == backward


def test_pipeline_digest_changes_when_the_map_changes() -> None:
    base = PreprocessingPipeline(PreprocessingConfig())
    for change in (
        {"n_max": 3},
        {"lemmatiser": LemmatiserKind.NONE},
        {"insert_gaps": False},
        {"cross_gaps": True},
        {"stopword_asset": None},
    ):
        other = PreprocessingPipeline(PreprocessingConfig().with_(**change))
        assert other.digest() != base.digest(), f"digest unchanged after {change}"


def test_pipeline_digest_binds_the_stopword_contents_not_just_the_name() -> None:
    from tfidf_stability.preprocessing.stopwords import StopwordSet

    cfg = PreprocessingConfig()
    a = PreprocessingPipeline(cfg, stopwords=StopwordSet.from_iterable(["the", "a"]))
    b = PreprocessingPipeline(cfg, stopwords=StopwordSet.from_iterable(["the", "a", "of"]))
    assert a.digest() != b.digest()


def test_all_stopword_document_yields_no_features(pipeline: PreprocessingPipeline) -> None:
    """Section 2.2's zero-vector case, arising at the preprocessing stage."""
    assert pipeline.preprocess("the of and a") == []


def test_map_is_stable_across_processes_and_hash_seeds() -> None:
    """PYTHONHASHSEED must not be able to influence the output.

    Subprocesses, because ``PYTHONHASHSEED`` is fixed at interpreter start-up
    and cannot be changed from inside a running process. Dict and set iteration
    order depend on it, and nothing else in the suite would see that order
    leaking into the feature stream.
    """
    snippet = (
        "import sys; sys.path.insert(0, r'%s');"
        "from tfidf_stability.preprocessing.pipeline import PreprocessingPipeline as P;"
        "p = P();"
        "print(p.digest());"
        "print('|'.join(p.preprocess('The King of Pop ran through New York City twice')))"
    ) % (REPO / "src")

    outputs = set()
    for seed in ("0", "1", "12345", "random"):
        res = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": seed, "PATH": ""},
            check=True,
        )
        outputs.add(res.stdout)
    assert len(outputs) == 1, f"output varied with PYTHONHASHSEED: {outputs}"


def test_the_stopword_asset_is_verified_against_its_recorded_digest(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """ "Hash-verified" is asserted in three places; make it true in one.

    ``stopwords.py``'s header, the asset's own header and
    ``configs/default.yaml`` all say the list is verified at load, and nothing
    verified it: ``data/assets/MANIFEST.sha256`` did not exist, and
    ``scripts/check_vendored.py`` discovers work by ``rglob("MANIFEST.sha256")``
    so a directory without one is invisible to it. Editing a single word changed
    every df, idf and score; the reproducibility snapshot compares runs to each
    other rather than to a pinned value, so it could not catch that.
    """
    import shutil

    from tfidf_stability.preprocessing import stopwords as module
    from tfidf_stability.utils.validation import DataIntegrityError

    genuine = module.load_stopwords()
    assert genuine.digest == module._recorded_digest(module.DEFAULT_STOPWORD_ASSET)

    shutil.copy(module._MANIFEST, tmp_path / "MANIFEST.sha256")
    original = (module._ASSET_DIR / module.DEFAULT_STOPWORD_ASSET).read_text(encoding="utf-8")
    (tmp_path / module.DEFAULT_STOPWORD_ASSET).write_text(
        original.replace("\nnot\n", "\n"), encoding="utf-8"
    )

    asset_dir, manifest = module._ASSET_DIR, module._MANIFEST
    try:
        module._ASSET_DIR, module._MANIFEST = tmp_path, tmp_path / "MANIFEST.sha256"
        module.load_stopwords.cache_clear()
        with pytest.raises(DataIntegrityError, match="recorded digest"):
            module.load_stopwords()
    finally:
        module._ASSET_DIR, module._MANIFEST = asset_dir, manifest
        module.load_stopwords.cache_clear()

    assert module.load_stopwords().digest == genuine.digest, "the real asset still loads"


def test_an_injected_lemmatiser_reaches_the_digest() -> None:
    """Two pipelines that produce different features must not share an identity.

    ``digest()`` read the config alone, but the constructor takes a ready-made
    lemmatiser that bypasses ``config.lemmatiser``. ``PreprocessingPipeline(cfg)``
    and ``PreprocessingPipeline(cfg, lemmatiser=IdentityLemmatiser())`` turn
    "running cats" into ``run|cat`` and ``running|cats`` respectively, yet
    digested to the same string, so a run manifest could not tell them apart. The
    sibling ``stopwords=`` injection was always bound by content.
    """
    from tfidf_stability.preprocessing.lemmatise import LemmatiserKind, make_lemmatiser
    from tfidf_stability.preprocessing.pipeline import (
        PreprocessingConfig,
        PreprocessingPipeline,
    )

    config = PreprocessingConfig()
    plain = PreprocessingPipeline(config)
    injected = PreprocessingPipeline(config, lemmatiser=make_lemmatiser(LemmatiserKind.NONE))

    assert plain.preprocess("running cats") != injected.preprocess("running cats")
    assert plain.digest() != injected.digest(), "different features, same identity"

    # Injecting the lemmatiser the config already names must NOT move the
    # digest, or every recorded value in the repository would have churned.
    agreeing = PreprocessingPipeline(config, lemmatiser=make_lemmatiser(LemmatiserKind.PORTER2))
    assert agreeing.digest() == plain.digest()


# ---------------------------------------------------------------------------
# Offsets, accessors and the wrappers nothing had called
# ---------------------------------------------------------------------------
def test_tokens_carry_the_source_span_they_came_from() -> None:
    """Provenance: `tfidf inspect` shows where in the text a feature originated,
    so the span has to index back into the original string."""
    text = "alpha beta gamma"
    tokens = tokenise_with_offsets(text)
    assert [t.text for t in tokens] == ["alpha", "beta", "gamma"]
    for token in tokens:
        assert text[token.start : token.end] == token.text, "the span must recover the token"


def test_offsets_honour_the_same_length_filter_as_plain_tokenisation() -> None:
    """Two code paths that must agree, or a reported span would point at a token
    the pipeline discarded."""
    config = TokenisationConfig(min_token_length=4)
    text = "a bb cccc ddddd"
    assert [t.text for t in tokenise_with_offsets(text, config)] == tokenise(text, config)


def test_offsets_over_empty_text_yield_nothing() -> None:
    assert tokenise_with_offsets("") == []


def test_a_preprocessed_document_reports_how_many_features_it_produced() -> None:
    pipeline = PreprocessingPipeline(PreprocessingConfig())
    document = pipeline.preprocess_document("d0", "alpha beta gamma")
    assert document.n_features == len(document.features)
    assert document.n_features > 0


def test_a_pipeline_exposes_the_two_assets_that_decide_its_identity() -> None:
    """Both feed digest(), so a report naming one map must be able to show which
    stopword list and which lemmatiser produced it."""
    pipeline = PreprocessingPipeline(PreprocessingConfig())
    assert pipeline.stopwords.name
    assert pipeline.lemmatiser.name
    assert pipeline.digest()


def test_the_repr_names_the_parts_a_reader_needs_to_tell_two_pipelines_apart() -> None:
    """Asserted by content rather than by exact text, so adding a field later
    does not break this."""
    text = repr(PreprocessingPipeline(PreprocessingConfig()))
    assert "PreprocessingPipeline(" in text
    assert "lemmatiser=" in text
    assert "stopwords=" in text
    assert "digest=" in text


def test_the_stopword_set_repr_names_its_asset_and_size() -> None:
    pipeline = PreprocessingPipeline(PreprocessingConfig())
    text = repr(pipeline.stopwords)
    assert "StopwordSet(" in text
    assert "digest=" in text


def test_the_convenience_wrapper_agrees_with_building_a_pipeline_by_hand() -> None:
    """A second entry point that must not drift from the first, or two callers
    would preprocess the same corpus differently."""
    texts = ["alpha beta", "gamma the delta"]
    config = PreprocessingConfig()
    assert preprocess_all(texts, config) == [
        PreprocessingPipeline(config).preprocess(t) for t in texts
    ]


def test_preprocessing_no_texts_at_all_yields_no_streams() -> None:
    assert preprocess_all([], PreprocessingConfig()) == []


def test_gap_sentinels_are_dropped_when_a_caller_asks_for_a_flat_stream() -> None:
    """Used where the gaps have served their purpose and only the tokens matter."""
    assert list(iter_gap_free(["a", GAP, "b", GAP])) == ["a", "b"]
    assert list(iter_gap_free([])) == []


def test_a_stopword_asset_missing_from_the_manifest_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An asset that exists but is unrecorded must not load.

    The list decides the vocabulary, so an unverifiable one changes every
    published number. This is distinct from the digest-mismatch case: here there
    is nothing to compare against at all, which is the weaker and easier failure
    to overlook. The asset directory is redirected rather than the loader
    patched, so the loader's own logic runs.
    """
    from tfidf_stability.preprocessing import stopwords as stopwords_module

    (tmp_path / "unrecorded_v1.txt").write_text("the\nof\n", encoding="utf-8")
    (tmp_path / "MANIFEST.sha256").write_text("", encoding="utf-8")
    monkeypatch.setattr(stopwords_module, "_ASSET_DIR", tmp_path)
    monkeypatch.setattr(stopwords_module, "_MANIFEST", tmp_path / "MANIFEST.sha256")
    load_stopwords.cache_clear()

    try:
        with pytest.raises(DataIntegrityError, match="no recorded digest"):
            load_stopwords("unrecorded_v1.txt")
    finally:
        load_stopwords.cache_clear()


def test_a_missing_stopword_asset_raises_before_any_digest_is_computed() -> None:
    """A different failure from an unrecorded one, and the two must stay
    distinct: "the file is not there" and "the file is not vouched for" need
    different fixes."""
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        load_stopwords("no_such_asset_v9.txt")


def test_the_manifest_parser_skips_blank_lines_and_comments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sha256sum files carry comments, and a parser that treated one as an entry
    would either fail to find a real asset or vouch for a comment."""
    import hashlib

    from tfidf_stability.preprocessing import stopwords as stopwords_module

    # write_bytes, not write_text: on Windows the text form translates LF to
    # CRLF, so the digest would be taken over bytes that never reach the file.
    body = b"the\nof\n"
    (tmp_path / "commented_v1.txt").write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()
    (tmp_path / "MANIFEST.sha256").write_bytes(
        b"# a comment line\n\n   \n"
        # A real entry for a different asset, so the name comparison has to
        # reject one and keep looking rather than taking the first line it parses.
        + b"0" * 64
        + b"  some_other_asset.txt\n"
        + f"{digest}  commented_v1.txt\n".encode()
    )
    monkeypatch.setattr(stopwords_module, "_ASSET_DIR", tmp_path)
    monkeypatch.setattr(stopwords_module, "_MANIFEST", tmp_path / "MANIFEST.sha256")
    load_stopwords.cache_clear()

    try:
        loaded = load_stopwords("commented_v1.txt")
        assert "the" in loaded
    finally:
        load_stopwords.cache_clear()


def test_consecutive_gap_sentinels_open_one_boundary_rather_than_an_empty_segment() -> None:
    """Two stopwords in a row leave two sentinels side by side, and a leading
    stopword leaves one at the front. Either would produce an empty segment if
    the splitter closed a run unconditionally, and an empty segment contributes
    nothing but changes how many there are.
    """
    assert generate_ngrams(["a", GAP, GAP, "b"], 1, 2) == generate_ngrams(["a", GAP, "b"], 1, 2)
    assert generate_ngrams([GAP, "a", "b"], 1, 2) == generate_ngrams(["a", "b"], 1, 2)
    assert generate_ngrams([GAP], 1, 2) == []
    assert generate_ngrams([GAP, GAP], 1, 2) == []


def test_every_normalisation_stage_can_be_turned_off_independently() -> None:
    """The normative configuration has all of them on, which leaves the off
    paths carrying published numbers on no evidence. They exist so an ablation
    can attribute a change to one stage, and that only works if each is really
    separable.
    """
    text = "  A​B   c  "

    everything_off = NormalisationConfig(
        lowercase=False, strip_control=False, collapse_whitespace=False
    )
    assert normalise(text, everything_off) == unicodedata.normalize("NFKC", text)

    kept_control = normalise(text, NormalisationConfig(strip_control=False))
    assert "​" in kept_control, "the zero-width space survives"
    assert kept_control == "a​b c", "the other two stages still ran"

    kept_spacing = normalise(text, NormalisationConfig(collapse_whitespace=False))
    assert kept_spacing == "  ab   c  ", "the control character went, the spacing stayed"


def test_a_lemmatiser_override_changes_the_config_digest_on_its_own() -> None:
    """The pipeline accepts a ready-made lemmatiser that bypasses the configured
    one, so without this two pipelines producing different features would share
    an identity -- and a cached result computed under one would be served for the
    other."""
    config = PreprocessingConfig()
    plain = config.digest()

    assert config.digest(lemmatiser_override="lookup:v3") != plain
    assert config.digest(lemmatiser_override="lookup:v3") != config.digest(
        lemmatiser_override="lookup:v4"
    )
    assert config.digest(lemmatiser_override=None) == plain, "absent means digest as before"

    # The two bindings are independent: neither one masks the other.
    assert config.digest(stopword_digest="abc") != config.digest(
        stopword_digest="abc", lemmatiser_override="lookup:v3"
    )


def test_the_default_ngram_orders_are_unigrams_and_bigrams() -> None:
    """Every call site in the package passes the orders explicitly, so the
    defaults are only what an examples script or a notebook gets. They are part
    of the preprocessing map's identity: the config digest covers `n_min` and
    `n_max`, so a caller relying on the defaults and one passing (1, 2) must be
    getting the same features.
    """
    tokens = ["new", "york", "city"]
    assert generate_ngrams(tokens) == generate_ngrams(tokens, 1, 2)
    assert generate_ngrams(tokens) == [
        "new",
        "york",
        "city",
        f"new{JOINER}york",
        f"york{JOINER}city",
    ]


# ---------------------------------------------------------------------------
# Token length: the bounds are part of the hashed contract
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("length", [1, 2, 63, 64])
def test_a_token_at_or_inside_the_length_bounds_survives(length: int) -> None:
    """64 is the documented ceiling and it is inclusive. The bounds are applied
    during tokenisation rather than downstream precisely so they are hashed with
    the configuration -- a run that changed them would change its own digest."""
    assert tokenise("x" * length) == ["x" * length]


@pytest.mark.parametrize("length", [65, 200])
def test_a_token_past_the_ceiling_is_dropped_rather_than_truncated(length: int) -> None:
    """Truncating would manufacture a feature the document does not contain and
    could collide two distinct long tokens onto one. The guard exists against
    pathological input from fuzzing, where a megabyte-long 'word' is ordinary.
    """
    assert tokenise("x" * length) == []


def test_the_two_length_bounds_are_independent() -> None:
    """Each filters its own end, so a configuration can raise the floor without
    touching the ceiling."""
    text = "a bb ccc"
    assert tokenise(text, TokenisationConfig(min_token_length=2)) == ["bb", "ccc"]
    assert tokenise(text, TokenisationConfig(max_token_length=2)) == ["a", "bb"]


def test_an_inverted_length_range_admits_nothing_rather_than_everything() -> None:
    """`min > max` is an empty interval. It filters the corpus to nothing, which
    the vocabulary builder then refuses -- a loud failure rather than a silent
    inversion of the filter."""
    assert tokenise("a bb ccc", TokenisationConfig(min_token_length=5, max_token_length=2)) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("don't", ["don", "t"]),
        ("well-known", ["well", "known"]),
        ("a_b", ["a", "b"]),
        ("abc 123", ["abc", "123"]),
    ],
)
def test_the_pattern_splits_on_everything_that_is_not_a_letter_or_digit(
    text: str, expected: list[str]
) -> None:
    """Apostrophes, hyphens and underscores are all separators. Pinned because
    the pattern is hashed into the manifest: a run that treated "don't" as one
    token would have a different vocabulary and a different every-number."""
    assert tokenise(text) == expected


def test_the_gap_sentinel_cannot_survive_tokenisation() -> None:
    """It is a control character, so the word pattern cannot match it. That is
    what lets the sentinel be inserted afterwards without any risk of a document
    having contained one."""
    assert GAP not in tokenise("a" + GAP + "b")


# ---------------------------------------------------------------------------
# Stopword removal: where a gap goes, and where it does not
# ---------------------------------------------------------------------------
def _stopwords(*words: str) -> StopwordSet:
    """A set built here rather than loaded, so these tests do not depend on the
    frozen asset's contents. Local by house convention."""
    return StopwordSet(words, name="test", digest="0" * 64)


def test_an_interior_stopword_leaves_a_gap_behind() -> None:
    """The whole point of G7's sentinel: without it, removing "of" from
    "king of pop" yields the bigram "king pop", a feature that appears in no
    document and is manufactured by the preprocessing order."""
    assert remove_stopwords(["king", "of", "pop"], _stopwords("of")) == ["king", GAP, "pop"]


def test_consecutive_stopwords_collapse_to_a_single_gap() -> None:
    """One boundary, however many tokens were removed at it. A gap per stopword
    would leave empty segments between them, which changes how many segments the
    n-gram splitter sees without changing where the boundaries are."""
    removed = remove_stopwords(["a", "of", "the", "b"], _stopwords("of", "the"))
    assert removed == ["a", GAP, "b"]


def test_a_leading_or_trailing_stopword_leaves_no_gap() -> None:
    """There is nothing on the far side for an n-gram to span to, so a boundary
    marker there would be a segment separator with one empty side."""
    assert remove_stopwords(["of", "a", "the"], _stopwords("of", "the")) == ["a"]


def test_a_document_of_nothing_but_stopwords_becomes_empty() -> None:
    """Not a list of gaps. This is the all-stopword document that embeds to the
    zero vector -- the degenerate case G3 names and the corpus fixture plants."""
    assert remove_stopwords(["of", "the"], _stopwords("of", "the")) == []


def test_removal_can_be_asked_not_to_mark_the_boundary() -> None:
    """The ablation G7 offers. Kept so the seam-bigram effect can be measured
    rather than only asserted."""
    kept = remove_stopwords(["king", "of", "pop"], _stopwords("of"), insert_gaps=False)
    assert kept == ["king", "pop"]


def test_an_empty_stopword_set_removes_nothing_at_all() -> None:
    """The configuration that disables stopword removal, which must be a
    pass-through rather than a special case downstream."""
    assert remove_stopwords(["a", "b"], StopwordSet.empty()) == ["a", "b"]
    assert len(StopwordSet.empty()) == 0
    assert StopwordSet.empty().name == "none"


def test_membership_is_exact_and_expects_an_already_normalised_token() -> None:
    """No case folding here -- normalisation has already run. A set that
    lowercased on lookup would silently work on un-normalised input and then
    disagree with the vocabulary."""
    stopwords = _stopwords("of")
    assert stopwords.is_stopword("of")
    assert not stopwords.is_stopword("OF")


def test_iterating_a_stopword_set_is_sorted_so_a_manifest_is_stable() -> None:
    """The set is a frozenset internally, whose iteration order is not promised
    stable across runs. Anything writing the list out needs the sort."""
    assert list(_stopwords("z", "a", "m")) == ["a", "m", "z"]


# ---------------------------------------------------------------------------
# The pipeline end to end
# ---------------------------------------------------------------------------
def test_the_gap_is_what_stops_a_seam_bigram_being_manufactured() -> None:
    """The two configurations side by side, which is the only way to see that
    the sentinel does anything.

    With gaps on, "king of pop" yields no bigram at all -- the two survivors sit
    either side of a boundary. With gaps off it yields `king<JOINER>pop`, a
    feature no document contains.
    """
    with_gaps = PreprocessingPipeline().preprocess("King of Pop")
    without = PreprocessingPipeline(PreprocessingConfig(insert_gaps=False)).preprocess(
        "King of Pop"
    )

    assert with_gaps == ["king", "pop"]
    assert f"king{JOINER}pop" in without, "the seam bigram the sentinel exists to prevent"


@pytest.mark.parametrize("text", ["", "   ", "the of a"])
def test_a_document_with_no_surviving_tokens_preprocesses_to_nothing(text: str) -> None:
    """Empty, whitespace, and all-stopword all reach the same place. The
    vocabulary builder counts such a document towards the corpus size while it
    contributes no features."""
    assert PreprocessingPipeline().preprocess(text) == []


def test_an_inverted_ngram_range_is_refused_by_the_pipeline() -> None:
    """The guard lives in `generate_ngrams`, so the pipeline surfaces it rather
    than silently producing no features."""
    pipeline = PreprocessingPipeline(PreprocessingConfig(n_min=3, n_max=1))
    with pytest.raises(ValueError, match=r"n_max \(1\) must be at least n_min \(3\)"):
        pipeline.preprocess("a b c")
