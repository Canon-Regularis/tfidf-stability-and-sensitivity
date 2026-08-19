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
from tfidf_stability.preprocessing.normalise import normalise
from tfidf_stability.preprocessing.pipeline import (
    PreprocessingConfig,
    PreprocessingPipeline,
    preprocess_all,
)
from tfidf_stability.preprocessing.stopwords import load_stopwords, remove_stopwords
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
@given(st.text(max_size=200))
def test_normalisation_is_idempotent(text: str) -> None:
    """Otherwise the map depends on how many times it has been applied, which is
    an irreproducibility that is hard to locate."""
    once = normalise(text)
    assert normalise(once) == once


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
    with pytest.raises(FileNotFoundError):
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
