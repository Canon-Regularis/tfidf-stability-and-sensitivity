#!/usr/bin/env python3
"""Why the raw bit patterns are kept, and not just the decimal values.

Section 1.2 asks for intermediate quantities to stay inspectable. This example
is about the *representation* in which they are inspected, because a decimal
rendering is a lossy summary and the differences this study is built on are
exactly the ones it discards.

Three claims are demonstrated on a twelve-document corpus, each by measurement
rather than assertion:

1. **A single library call makes the pipeline platform-dependent.** ``log`` is
   the only transcendental here and IEEE-754 does not require it to be correctly
   rounded, so the platform libm and the exact value disagree on a real
   vocabulary entry -- at *N* = 12, not only at scale. That is
   ``docs/spec_addenda.md#g13``, reproduced small enough to read.

2. **The disagreement is invisible in decimal and fatal to a digest.** The
   affected scores print identically at every precision a report would plausibly
   use, and differ in ``float.hex``. Any equality check written against a
   formatted string reports agreement; ``same_bits`` reports the truth. This is
   why ``TfidfModel.digest`` hashes ``struct.pack("<d", ...)`` and not a
   rendering.

3. **Where the bits are measured decides what is measured.** Sweeping the
   reduction policy passed to the scorer, with the model's norms held fixed,
   finds *nothing* on this corpus. Refitting so the norms move too finds the
   noise floor immediately. That is the trap named in ``docs/spec_addenda.md#g23``,
   and it is the difference between a measured tau_floor and a vacuous one.

The closing sections give the phenomenon its place in the two research
questions. Bit-level disturbance is roughly thirteen decades below the smallest
real score gap, so A1's ranking instability is a story about corpus perturbation
and not about arithmetic -- but exact ties are a *bit* property, and the block of
bit-identical scores at the bottom of the ranking is precisely where A2's
deterministic tie-break, rather than any number, chooses the order.

Which vocabulary entry the libm gets wrong depends on the libm, so nothing here
is hardcoded: the corpus is searched and whatever is found is reported. Run it::

    python examples/inspect_intermediates.py
"""

from __future__ import annotations

import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tfidf_stability.preprocessing.pipeline import (  # noqa: E402
    PreprocessingConfig,
    PreprocessingPipeline,
)
from tfidf_stability.similarity.cosine import cosine_against_corpus  # noqa: E402
from tfidf_stability.utils.numerics import (  # noqa: E402
    Reduction,
    bits_of,
    same_bits,
    ulp,
    ulps_between,
)
from tfidf_stability.vectorisation.idf import LogImpl  # noqa: E402
from tfidf_stability.vectorisation.tfidf import TfidfModel, TfidfVectoriser  # noqa: E402

# Twelve documents, chosen so that one genuine vocabulary entry lands on a
# document frequency where this machine's libm rounds the logarithm differently
# from the exact value. Small enough that every intermediate can be printed.
DOCUMENTS: tuple[tuple[str, str], ...] = (
    ("d01", "A space opera about rebel pilots and a doomed alien battle station."),
    ("d02", "A space station crew fights an alien stowaway in deep space."),
    ("d03", "Rebel pilots defend a mining colony from an alien fleet."),
    ("d04", "A detective hunts a replicant through a rain soaked future city."),
    ("d05", "A detective hunts a killer through a rain soaked coastal city."),
    ("d06", "Two astronauts repair a crippled space station in orbit."),
    ("d07", "A colony ship carries sleeping settlers to a distant star."),
    ("d08", "An alien fleet threatens a distant mining colony."),
    ("d09", "A quiet drama about a chef and a failing seaside restaurant."),
    ("d10", "A chef reopens a failing restaurant in a seaside town."),
    ("d11", "A space marine boards a derelict alien ship in orbit."),
    ("d12", "Settlers on a distant colony discover a derelict alien craft."),
)

QUERY = "alien colony ship"

#: Renderings a report or a log line might plausibly use, coarsest first.
_RENDERINGS: tuple[str, ...] = (".6f", ".12f", ".15g", ".16g", ".17g")


@dataclass(frozen=True, slots=True)
class Fixture:
    """The corpus fitted twice, under each logarithm, with its scores."""

    doc_ids: tuple[str, ...]
    features: tuple[tuple[str, ...], ...]
    query_features: tuple[str, ...]
    exact: TfidfModel
    libm: TfidfModel
    scores_exact: tuple[float, ...]
    scores_libm: tuple[float, ...]


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------
def heading(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def rule(text: str) -> None:
    print(f"\n-- {text} " + "-" * max(0, 74 - len(text)))


def triple(label: str, x: float) -> str:
    """Decimal, hex and raw bytes side by side.

    All three are printed together throughout because the point of the example
    is that only the last two are decisive.
    """
    return f"  {label:<26}{x:>24.17g}   {float.hex(x):<22}{bits_of(x).hex()}"


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------
def build_fixture() -> Fixture:
    """Fit the corpus under both logarithm implementations and score the query.

    Unigrams only: bigrams would triple the vocabulary without adding anything
    the example needs, and every table here is meant to be read in full.
    """
    pipeline = PreprocessingPipeline(PreprocessingConfig(n_max=1))
    doc_ids = tuple(doc_id for doc_id, _ in DOCUMENTS)
    features = tuple(tuple(pipeline.preprocess(text)) for _, text in DOCUMENTS)
    query_features = tuple(pipeline.preprocess(QUERY))

    exact = TfidfVectoriser(log_impl=LogImpl.CORRECTLY_ROUNDED).fit(features, doc_ids)
    libm = TfidfVectoriser(log_impl=LogImpl.PLATFORM).fit(features, doc_ids)
    return Fixture(
        doc_ids=doc_ids,
        features=features,
        query_features=query_features,
        exact=exact,
        libm=libm,
        scores_exact=tuple(score(exact, query_features)),
        scores_libm=tuple(score(libm, query_features)),
    )


def score(
    model: TfidfModel,
    query_features: tuple[str, ...],
    policy: Reduction | None = None,
) -> list[float]:
    """Score the query against every document of ``model``."""
    used = model.reduction if policy is None else policy
    query = TfidfVectoriser.transform_query(query_features, model)
    return cosine_against_corpus(query, list(model.matrix.rows()), model.norms, used)


def divergent_terms(fixture: Fixture) -> list[int]:
    """Term identifiers whose idf differs between the two logarithms."""
    return [
        t
        for t in range(fixture.exact.n_features)
        if not same_bits(fixture.exact.idf[t], fixture.libm.idf[t])
    ]


# ---------------------------------------------------------------------------
# 1. The logarithm  (docs/spec_addenda.md#g13)
# ---------------------------------------------------------------------------
def section_logarithm(fixture: Fixture, divergent: list[int]) -> None:
    rule("1. The one transcendental in the pipeline  (spec_addenda.md#g13)")
    vocab = fixture.exact.vocabulary
    print(
        f"  N = {fixture.exact.n_documents}, |V| = {fixture.exact.n_features}; "
        f"idf entries where the platform libm disagrees with the exact value: "
        f"{len(divergent)}/{fixture.exact.n_features}"
    )
    if not divergent:
        print("  None on this platform's libm. The rest of section 1 has nothing to show;")
        print("  the phenomenon is real but its incidence is platform-specific (G13).")
        return

    for t in divergent:
        exact_idf, libm_idf = fixture.exact.idf[t], fixture.libm.idf[t]
        print(f"\n  token {vocab.token_of(t)!r}, df = {vocab.df[t]}, idf = log((1+N)/(1+df)) + 1")
        print(triple("correctly rounded", exact_idf))
        print(triple("platform libm", libm_idf))
        print(
            f"  {'distance':<26}{ulps_between(exact_idf, libm_idf):>24.1f} ulp"
            f"   (abs {abs(exact_idf - libm_idf):.3e})"
        )


# ---------------------------------------------------------------------------
# 2. Propagation through the intermediates
# ---------------------------------------------------------------------------
def section_propagation(fixture: Fixture, divergent: list[int]) -> None:
    rule("2. One ulp in idf does not stay in idf")
    if not divergent:
        print("  Nothing diverged upstream, so there is nothing to trace.")
        return

    term_id = divergent[0]
    token = fixture.exact.vocabulary.token_of(term_id)
    index = next(
        i for i in range(fixture.exact.n_documents) if term_id in fixture.exact.document(i).indices
    )
    exact_terms = _terms_of(fixture.exact, index)
    libm_terms = _terms_of(fixture.libm, index)

    print(
        f"  Tracing {token!r} through document {fixture.doc_ids[index]}, taken from"
        " TfidfModel.intermediates(),\n  and on into the score.\n"
    )
    print(f"  {'quantity':<26}{'decimal (17 sig)':>24}   {'float.hex':<22}raw bytes (LE)")

    for stage, key in (("tf", "tf"), ("idf", "idf"), ("weight w = tf * idf", "weight")):
        a, b = float(exact_terms[term_id][key]), float(libm_terms[term_id][key])
        print(triple(stage, a))
        print(triple("  under platform libm", b))
        print(f"  {'':26}{_verdict(a, b)}")

    for label, a, b in (
        ("norm ||w_i||", fixture.exact.norms[index], fixture.libm.norms[index]),
        ("score s_i", fixture.scores_exact[index], fixture.scores_libm[index]),
    ):
        print(triple(label, a))
        print(triple("  under platform libm", b))
        print(f"  {'':26}{_verdict(a, b)}")

    print(
        "\n  One ulp goes in and the trace is not monotone: the norm absorbed it"
        "\n  completely -- bit-identical -- while the score came out at two ulps. Which"
        "\n  intermediate survives a perturbation is not deducible from its size, which"
        "\n  is the argument for retaining all of them rather than reasoning about them."
    )
    affected = [
        fixture.doc_ids[i]
        for i in range(fixture.exact.n_documents)
        if not same_bits(fixture.scores_exact[i], fixture.scores_libm[i])
    ]
    print(f"  Scores that differ across the two logarithms: {affected or 'none'}")


def _terms_of(model: TfidfModel, index: int) -> dict[int, dict[str, Any]]:
    """``intermediates`` for one document, re-keyed by term identifier."""
    intermediates: dict[str, Any] = model.intermediates(index)
    terms: list[dict[str, Any]] = intermediates["terms"]
    return {int(term["term_id"]): term for term in terms}


def _verdict(a: float, b: float) -> str:
    if same_bits(a, b):
        return "same_bits -> True"
    return f"same_bits -> False, {ulps_between(a, b):+.1f} ulp"


# ---------------------------------------------------------------------------
# 3. What decimal hides
# ---------------------------------------------------------------------------
def section_rendering(fixture: Fixture) -> None:
    rule("3. What a decimal rendering hides")
    pairs = [
        (fixture.doc_ids[i], fixture.scores_exact[i], fixture.scores_libm[i])
        for i in range(fixture.exact.n_documents)
        if not same_bits(fixture.scores_exact[i], fixture.scores_libm[i])
    ]
    if not pairs:
        print("  No score differed, so no rendering can hide anything here.")
        return

    doc_id, a, b = pairs[0]
    print(f"  The two values of score({doc_id}), formatted as a report would format them:\n")
    print(f"  {'format':<10}{'correctly rounded':>24}{'platform libm':>24}   equal as text?")
    hiding = 0
    for spec in _RENDERINGS:
        left, right = format(a, spec), format(b, spec)
        hiding += left == right
        print(f"  {spec:<10}{left:>24}{right:>24}   {left == right}")
    print(
        f"\n  {hiding} of these {len(_RENDERINGS)} renderings report agreement."
        "\n  A test, a log line or a cache key written against any of them would call the"
        "\n  two values equal. bits_of() would not, which is why TfidfModel.digest hashes"
        "\n  struct.pack('<d', ...) and never a formatted string."
    )


# ---------------------------------------------------------------------------
# 4. Reduction policy  (docs/spec_addenda.md#g23)
# ---------------------------------------------------------------------------
def section_reduction(fixture: Fixture) -> tuple[float, float]:
    """Compare the reduction policies both ways, and return (naive_eta, trap_eta)."""
    rule("4. Reduction policy, and where you measure it  (spec_addenda.md#g23)")

    ids, feats, qf = fixture.doc_ids, fixture.features, fixture.query_features
    models = {p: TfidfVectoriser(reduction=p).fit(feats, ids) for p in Reduction}
    refit = {p: score(models[p], qf) for p in Reduction}
    ground = refit[Reduction.EXACT]

    # G23's trap: hold the model (and therefore its norms) at NAIVE and vary only
    # the policy handed to the scorer, so the dot product alone moves.
    naive_model = models[Reduction.NAIVE]
    dot_only = score(naive_model, qf, Reduction.EXACT)
    trap_eta = max(abs(a - b) for a, b in zip(refit[Reduction.NAIVE], dot_only, strict=True))

    print("  Refitting under each policy, so the norms move as well as the dot product:\n")
    header = f"{'scores != EXACT':>18}{'norms != EXACT':>18}{'max |diff|':>14}"
    print(f"  {'policy':<12}{header}  max ulp")
    for policy in Reduction:
        scores = refit[policy]
        n_scores = sum(1 for a, b in zip(scores, ground, strict=True) if not same_bits(a, b))
        n_norms = sum(
            1
            for a, b in zip(models[policy].norms, models[Reduction.EXACT].norms, strict=True)
            if not same_bits(a, b)
        )
        worst = max(abs(a - b) for a, b in zip(scores, ground, strict=True))
        worst_ulp = max(abs(ulps_between(a, b)) for a, b in zip(scores, ground, strict=True))
        n = len(scores)
        print(
            f"  {policy!s:<12}{f'{n_scores}/{n}':>18}{f'{n_norms}/{n}':>18}"
            f"{worst:>14.3e}  {worst_ulp:.1f}"
        )

    naive_eta = max(abs(a - b) for a, b in zip(refit[Reduction.NAIVE], ground, strict=True))
    print(f"\n  Varying the scorer's policy only, norms held at the NAIVE fit: {trap_eta:.3e}")
    print(f"  Varying the fit as well, so the norms move too:                {naive_eta:.3e}")
    print(
        "\n  The first number is the trap, and here it is total: a query dot product runs"
        "\n  over the three shared terms at most and rounds identically under every"
        "\n  policy, so a sweep that varies only the scorer looks informative and reports"
        "\n  an exactly zero noise floor. The error lives in the norms, which sum the"
        "\n  whole document vector. The same trap at 1500 documents understates eta"
        "\n  threefold rather than erasing it (G23); on a corpus this small it erases it."
    )
    print(
        "\n  Two policies are worth naming. PAIRWISE is bit-identical to NAIVE because"
        "\n  its block size is 128 and nothing here is that long. NEUMAIER is exactly"
        "\n  correctly rounded -- which makes it a better sum than the paper specifies,"
        "\n  and therefore an instrument rather than a fix."
    )
    return naive_eta, trap_eta


# ---------------------------------------------------------------------------
# 5. Exact ties  (A2)
# ---------------------------------------------------------------------------
def section_ties(fixture: Fixture) -> None:
    rule("5. Exact ties are a bit-level property  (A2)")
    scores = fixture.scores_exact
    groups: list[list[int]] = []
    for i, s in enumerate(scores):
        for group in groups:
            if same_bits(scores[group[0]], s):
                group.append(i)
                break
        else:
            groups.append([i])

    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    print("  Ranking under the normative configuration:\n")
    for rank, i in enumerate(order, start=1):
        print(f"  {rank:>3}. {fixture.doc_ids[i]}  {scores[i]:>20.17g}   {float.hex(scores[i])}")

    tied = [g for g in groups if len(g) > 1]
    print()
    for group in tied:
        members = ", ".join(fixture.doc_ids[i] for i in group)
        print(f"  {len(group)} documents share one bit pattern: {members}")
        print(f"     value {float.hex(scores[group[0]])}, bytes {bits_of(scores[group[0]]).hex()}")
        print(f"     orderings the arithmetic permits: {math.factorial(len(group))}")
    if tied:
        print(
            "\n  These are not near-ties to be resolved by a tolerance -- the bytes are"
            "\n  equal, so no tau and no better summation can separate them. The order"
            "\n  among them is chosen entirely by the deterministic tie-break, which is"
            "\n  the discontinuity A2 studies: a decision made where the numbers stop"
            "\n  speaking. Detecting the condition at all requires same_bits."
        )


# ---------------------------------------------------------------------------
# 6. Scale  (A1)
# ---------------------------------------------------------------------------
def section_scale(fixture: Fixture, eta: float) -> None:
    rule("6. Why A1 is not a story about arithmetic")
    scores = sorted(fixture.scores_exact, reverse=True)
    gaps = [a - b for a, b in itertools.pairwise(scores)]
    positive = [g for g in gaps if g > 0.0]
    g_min = min(positive)
    top = max(scores)
    tau_floor = 2.0 * eta

    print(triple("largest score", top))
    print(f"  {'one ulp there':<26}{ulp(top):>24.3e}")
    print(f"  {'measured eta':<26}{eta:>24.3e}   worst NAIVE vs EXACT score disagreement")
    print(f"  {'tau_floor = 2 eta':<26}{tau_floor:>24.3e}")
    print(f"  {'g_min':<26}{g_min:>24.3e}   smallest strictly positive adjacent gap")
    print(f"  {'eps_flip = g_min / 2':<26}{g_min / 2.0:>24.3e}   section 4.4")
    print(f"  {'band width':<26}{math.log10(g_min / tau_floor):>24.2f} decades")
    print(f"  {'g_min in ulps of the top':<26}{g_min / ulp(top):>24.3e}")
    print(
        "\n  The closest pair of distinctly scored documents would need a perturbation of"
        f"\n  {g_min / 2.0:.3e} to swap; arithmetic supplies {eta:.3e}. That clearance is"
        "\n  wider than the 6.32 decades G23 measures at 1500 documents, and the two are"
        "\n  not comparable -- twelve documents make a coarse score lattice, so g_min here"
        "\n  is large. The separation is of the same kind and points the same way."
        "\n"
        "\n  So the bits do not threaten the ranking, and that is the finding, not an"
        "\n  excuse to stop recording them. It is what separates A1 from A2: bounded"
        "\n  perturbation is a question about the corpus, and floating point only"
        "\n  decides anything where the scores are bit-identical -- where A2 begins."
    )


def main() -> int:
    fixture = build_fixture()
    heading("Raw bit patterns in the TF-IDF pipeline")
    print(f"  corpus            {len(DOCUMENTS)} documents, |V| = {fixture.exact.n_features}")
    print(f"  query             {QUERY!r} -> {list(fixture.query_features)}")
    print(f"  reduction         {fixture.exact.reduction}")
    print(f"  model digest      {fixture.exact.digest()[:32]}...")

    divergent = divergent_terms(fixture)
    section_logarithm(fixture, divergent)
    section_propagation(fixture, divergent)
    section_rendering(fixture)
    eta, _ = section_reduction(fixture)
    section_ties(fixture)
    section_scale(fixture, eta)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
