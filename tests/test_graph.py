import pandas as pd
import pytest

from aml_triage.graph import build_edges, build_graph, structural_signals


def signals(tx):
    G = build_graph(build_edges(tx))
    return G, structural_signals(tx, G)


def test_self_transfer_is_not_an_edge(tx_factory):
    G, _ = signals(tx_factory([("A", "A"), ("A", "B")]))
    assert not G.has_edge("A", "A") and G.number_of_edges() == 1


def test_account_paying_itself_is_not_a_cycle(tx_factory):
    _, s = signals(tx_factory([("A", "A"), ("A", "B")]))
    assert s.loc["A", "two_cycle"] == 0 and s.loc["A", "three_cycle"] == 0


def test_two_cycle_detected(tx_factory):
    _, s = signals(tx_factory([("A", "B"), ("B", "A")]))
    assert (s.two_cycle > 0).all() and (s.three_cycle == 0).all()


def test_three_cycle_detected(tx_factory):
    _, s = signals(tx_factory([("A", "B"), ("B", "C"), ("C", "A")]))
    assert (s.three_cycle > 0).all() and (s.two_cycle == 0).all()


@pytest.mark.xfail(reason="the given signals cover 2- and 3-cycles only; a 4-node cycle is invisible to them (see C4)")
def test_four_node_cycle_detected(tx_factory):
    _, s = signals(tx_factory([("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]))
    assert ((s.two_cycle > 0) | (s.three_cycle > 0)).all()


def test_fan_out_counts_distinct_receivers_per_day(tx_factory):
    _, s = signals(tx_factory([("A", "B"), ("A", "C"), ("A", "D"), ("A", "B")]))
    assert s.loc["A", "max_fan_out_day"] == 3


def test_gather_scatter_in_on_day_d_out_on_d_plus_1(tx_factory):
    gather = tx_factory([(f"S{i}", "H") for i in range(4)], day="2022-09-02")
    scatter = tx_factory([("H", f"R{i}") for i in range(4)], day="2022-09-03")
    _, s = signals(pd.concat([gather, scatter], ignore_index=True))
    assert s.loc["H", "gather_scatter"] == 1 and s.gather_scatter.sum() == 1