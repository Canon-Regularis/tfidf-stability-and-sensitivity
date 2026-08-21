"""Shared fixtures and test configuration.

Hypothesis profiles, selected by ``HYPOTHESIS_PROFILE`` and defaulting to
``dev``: ``dev`` 50 examples, ``ci`` 1,000 (the per-PR gate), ``nightly``
100,000 (adversarial search for violations of the README section 4
inequalities).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

# src/ on sys.path, but only when the package cannot be imported without it: the
# suite runs with no install, and before the native backend has ever been built.
#
# The guard matters more than the insertion. Unconditionally prepending `src/`
# means the suite always tests the source tree, even when it was pointed at an
# installed distribution on purpose. `release.yml`'s CIBW_TEST_COMMAND runs
# `pytest {project}/tests` against a freshly built wheel for exactly that
# purpose, and got the source tree instead -- so nothing in the wheel was ever
# under test, and a wheel missing its data files would pass.
#
# Checked by import rather than by looking for a marker file, because that is
# the question being asked: can this interpreter import the package as it
# stands? An editable install resolves back to `src/` anyway, so a developer
# checkout behaves exactly as before.
if importlib.util.find_spec("tfidf_stability") is None:  # pragma: no cover
    _SRC = Path(__file__).resolve().parents[1] / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

from tfidf_stability.preprocessing.pipeline import (
    PreprocessingConfig,
    PreprocessingPipeline,
)
from tfidf_stability.ranking.attributes import AttributeTable
from tfidf_stability.vectorisation.tfidf import TfidfModel, TfidfVectoriser

FIXTURES = Path(__file__).resolve().parent / "fixtures"


# ---------------------------------------------------------------------------
# Hypothesis profiles
# ---------------------------------------------------------------------------
def pytest_configure(config: pytest.Config) -> None:
    try:
        from hypothesis import HealthCheck, Verbosity, settings
    except ImportError:  # pragma: no cover - hypothesis is a dev dependency
        return

    settings.register_profile("dev", max_examples=50, deadline=None)
    settings.register_profile(
        "ci",
        max_examples=1_000,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    settings.register_profile(
        "nightly",
        max_examples=100_000,
        deadline=None,
        verbosity=Verbosity.normal,
        suppress_health_check=[HealthCheck.too_slow],
    )
    import os

    settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


# ---------------------------------------------------------------------------
# Corpus fixtures
# ---------------------------------------------------------------------------
def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.fixture(scope="session")
def mini_corpus() -> list[dict[str, object]]:
    """Six hand-written documents covering the degenerate cases.

    Exact-duplicate pair d3/d4 (they tie exactly), near-duplicate pair d1/d2,
    all-stopword document d5 (embeds to the zero vector). Most hand-derived
    expectations in the suite are computed from this corpus.
    """
    return _read_jsonl(FIXTURES / "mini_corpus.jsonl")


@pytest.fixture(scope="session")
def mini_queries() -> list[dict[str, object]]:
    """Five queries, including two that embed to the zero vector by different routes."""
    return _read_jsonl(FIXTURES / "mini_queries.jsonl")


@pytest.fixture(scope="session")
def pipeline() -> PreprocessingPipeline:
    """The normative preprocessing map."""
    return PreprocessingPipeline(PreprocessingConfig())


@pytest.fixture(scope="session")
def mini_features(
    mini_corpus: list[dict[str, object]], pipeline: PreprocessingPipeline
) -> list[list[str]]:
    """Preprocessed feature streams for the mini corpus."""
    return [pipeline.preprocess(str(d["text"])) for d in mini_corpus]


@pytest.fixture(scope="session")
def mini_model(
    mini_features: Sequence[Sequence[str]], mini_corpus: list[dict[str, object]]
) -> TfidfModel:
    """A fitted model over the mini corpus, under the normative configuration."""
    return TfidfVectoriser().fit(
        list(mini_features), doc_ids=[str(d["doc_id"]) for d in mini_corpus]
    )


@pytest.fixture(scope="session")
def mini_attributes(mini_corpus: list[dict[str, object]]) -> AttributeTable:
    """Tie-break attributes for the mini corpus.

    The fixture file already carries ``popularity``, ``rating_sum2``,
    ``rating_count`` and ``engagement`` in G8's exact-pair representation, so
    the loader takes them as written rather than reshaping them.
    """
    return AttributeTable.from_records(mini_corpus)


@pytest.fixture(scope="session")
def snowball_vectors() -> tuple[list[str], list[str]]:
    """The official Snowball English test vectors, vendored for offline runs."""
    d = FIXTURES / "snowball"
    voc = d.joinpath("voc.txt").read_bytes().decode("utf-8").splitlines()
    out = d.joinpath("output.txt").read_bytes().decode("utf-8").splitlines()
    return voc, out
