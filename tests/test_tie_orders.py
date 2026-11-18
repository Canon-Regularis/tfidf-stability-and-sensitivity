"""Identifier tie orders: bijections, reversal, seeded shuffles and their effect."""

from __future__ import annotations

import pytest

from tfidf_stability.evaluation.tie_orders import TieOrder, tie_order_label, with_tie_order
from tfidf_stability.ranking.attributes import AttributeTable
from tfidf_stability.ranking.ranker import rank_top_k
from tfidf_stability.ranking.sort_keys import PI_SCORE

#: UTF-8 order is Z, a10, a9, b, c, é.
_IDS = ["b", "a10", "a9", "c", "é", "Z"]


def _table(ids: list[str] = _IDS) -> AttributeTable:
    return AttributeTable.from_records([{"doc_id": d} for d in ids])


def test_ascending_rebuilds_the_order_from_records() -> None:
    table = _table()

    assert table.id_ranks == (3, 1, 2, 4, 5, 0)
    assert with_tie_order(table, TieOrder.ID_ASC) == table


def test_descending_reverses_it() -> None:
    table = _table()

    assert with_tie_order(table, TieOrder.ID_DESC).id_ranks == (2, 4, 3, 1, 0, 5)


def test_a_shuffle_is_pinned_by_its_seed() -> None:
    """Literals: a change of hash input or tie key shows here, on every platform."""
    table = _table()

    assert with_tie_order(table, TieOrder.SHUFFLE, seed=0).id_ranks == (5, 4, 2, 1, 3, 0)
    assert with_tie_order(table, TieOrder.SHUFFLE, seed=1).id_ranks == (2, 5, 1, 3, 0, 4)


@pytest.mark.parametrize(
    ("order", "seed"),
    [(TieOrder.ID_ASC, None), (TieOrder.ID_DESC, None), (TieOrder.SHUFFLE, 7)],
)
def test_every_order_is_a_bijection_and_keeps_the_columns(
    order: TieOrder, seed: int | None
) -> None:
    table = AttributeTable.from_records(
        [{"doc_id": f"d{i}", "popularity": i % 3} for i in range(40)]
    )

    reordered = with_tie_order(table, order, seed=seed)

    assert sorted(reordered.id_ranks) == list(range(40))
    assert reordered.columns is table.columns
    assert reordered.doc_ids == table.doc_ids


def test_seeds_give_distinct_orders_and_a_seed_repeats_its_own() -> None:
    table = _table([f"d{i}" for i in range(30)])
    orders = {with_tie_order(table, TieOrder.SHUFFLE, seed=s).id_ranks for s in range(10)}

    assert len(orders) == 10
    assert with_tie_order(table, "shuffle", seed=3) == with_tie_order(
        table, TieOrder.SHUFFLE, seed=3
    ), "a string spelling names the same order"


@pytest.mark.parametrize(
    ("order", "seed"),
    [(TieOrder.SHUFFLE, None), (TieOrder.ID_ASC, 0), (TieOrder.ID_DESC, 1)],
)
def test_a_seed_is_required_for_a_shuffle_and_only_for_it(
    order: TieOrder, seed: int | None
) -> None:
    with pytest.raises(ValueError, match="a seed is required for shuffle and only for it"):
        with_tie_order(_table(), order, seed=seed)


def test_the_order_decides_which_tied_document_ranks_first() -> None:
    table = _table(["a", "b", "c"])
    scores = [0.5, 0.5, 0.25]

    asc = rank_top_k(scores, with_tie_order(table, TieOrder.ID_ASC), PI_SCORE, k=2)
    desc = rank_top_k(scores, with_tie_order(table, TieOrder.ID_DESC), PI_SCORE, k=2)

    assert asc.top_k(2) == (0, 1)
    assert desc.top_k(2) == (1, 0), "only the tie flips; the lower score stays behind"


def test_labels_name_each_variant() -> None:
    assert tie_order_label(TieOrder.ID_ASC) == "id_asc"
    assert tie_order_label(TieOrder.ID_DESC) == "id_desc"
    assert tie_order_label(TieOrder.SHUFFLE, 4) == "shuffle_4"
    assert str(TieOrder.SHUFFLE) == "shuffle"
