"""The noise floor between runs produced outside the package.

The model-based floor and the run-based floor share one comparison, so feeding the
second the cosine vectors the first computes must reproduce it field for field.
"""

from __future__ import annotations

import math
import random

import pytest

from tfidf_stability.analysis import measure_noise_floor, noise_floor_from_runs, tau_band
from tfidf_stability.similarity.cosine import cosine_against_corpus
from tfidf_stability.utils.numerics import Reduction
from tfidf_stability.vectorisation.sparse import SparseVector
from tfidf_stability.vectorisation.tfidf import TfidfModel, TfidfVectoriser

_REFERENCE = [[0.5, 0.25, 0.125], [1.0, 0.0, 0.75]]


def _corpus() -> tuple[TfidfModel, list[SparseVector]]:
    rng = random.Random(11)
    vocab = [f"t{i}" for i in range(150)]
    documents = [[rng.choice(vocab) for _ in range(80)] for _ in range(50)]
    model = TfidfVectoriser().fit(documents, [f"d{i}" for i in range(50)])
    queries = [
        TfidfVectoriser.transform_query([rng.choice(vocab) for _ in range(40)], model)
        for _ in range(5)
    ]
    return model, queries


def test_the_run_floor_reproduces_the_model_floor() -> None:
    model, queries = _corpus()
    documents = [model.document(i) for i in range(model.n_documents)]

    def scores(policy: Reduction) -> list[list[float]]:
        norms = model.matrix.row_norms(policy)
        return [cosine_against_corpus(q, documents, norms, policy) for q in queries]

    from_model = measure_noise_floor(model, queries)
    from_runs = noise_floor_from_runs(
        scores(Reduction.EXACT),
        {str(p): scores(p) for p in (Reduction.PAIRWISE, Reduction.NAIVE, Reduction.NEUMAIER)},
    )

    assert from_model.eta > 0.0, "the corpus must make some policy stray from exact"
    assert from_runs.as_dict() == from_model.as_dict()


def test_a_run_against_itself_has_no_noise() -> None:
    floor = noise_floor_from_runs(_REFERENCE, {"repeat": [list(v) for v in _REFERENCE]})

    assert (floor.eta, floor.tau_floor) == (0.0, 0.0)
    assert floor.per_policy[0].as_dict()["n_compared"] == 6
    assert (floor.n_queries, floor.n_documents) == (2, 3)


def test_one_ulp_is_measured_exactly() -> None:
    bumped = [list(v) for v in _REFERENCE]
    bumped[1][2] = math.nextafter(0.75, 0.0)

    floor = noise_floor_from_runs(_REFERENCE, {"gpu_batch32": bumped})
    (error,) = floor.per_policy

    assert (error.policy, error.n_differing, error.max_ulps) == ("gpu_batch32", 1, 1.0)
    assert floor.eta == 0.75 - math.nextafter(0.75, 0.0)


def test_labels_are_compared_in_utf8_order() -> None:
    same = [list(v) for v in _REFERENCE]
    floor = noise_floor_from_runs(_REFERENCE, {"b": same, "B": same, "a": same})

    assert [p.policy for p in floor.per_policy] == ["B", "a", "b"]


def test_the_band_derives_from_a_run_floor() -> None:
    bumped = [[0.5, 0.25, math.nextafter(0.125, 1.0)], [1.0, 0.0, 0.75]]
    floor = noise_floor_from_runs(_REFERENCE, {"cpu_threads4": bumped})

    band = tau_band(floor, [sorted(v, reverse=True) for v in _REFERENCE])

    assert band.tau_floor == 2.0 * (math.nextafter(0.125, 1.0) - 0.125)
    assert band.g_min == 0.125


@pytest.mark.parametrize(
    ("variants", "message"),
    [
        ({}, "at least one variant run is needed"),
        ({"short": [[0.5, 0.25, 0.125]]}, "run 'short' has 1 queries; the reference has 2"),
        (
            {"narrow": [[0.5, 0.25, 0.125], [1.0, 0.0]]},
            "run 'narrow' query 1 scores 2 documents; the reference scores 3",
        ),
    ],
)
def test_a_run_shaped_unlike_the_reference_is_refused(
    variants: dict[str, list[list[float]]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        noise_floor_from_runs(_REFERENCE, variants)


@pytest.mark.parametrize("which", ["variant", "reference"])
def test_a_non_finite_score_on_either_side_is_refused(which: str) -> None:
    poisoned = [[0.5, math.nan, 0.125], [1.0, 0.0, 0.75]]
    reference, variant = (_REFERENCE, poisoned) if which == "variant" else (poisoned, _REFERENCE)

    with pytest.raises(ValueError, match="or the reference holds a non-finite score at query 0"):
        noise_floor_from_runs(reference, {"gpu": variant})


def test_an_empty_reference_has_no_documents() -> None:
    floor = noise_floor_from_runs([], {"none": []})

    assert (floor.n_queries, floor.n_documents, floor.eta) == (0, 0, 0.0)
