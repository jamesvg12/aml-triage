"""Notebook Sections 4-5: the account graph (no self-loops) and the structural signals."""
import networkx as nx
import numpy as np
import pandas as pd


def build_edges(tx: pd.DataFrame) -> pd.DataFrame:
    """One row per (sender, receiver) pair; self-transfers are not edges (Trap 4)."""
    return (tx[~tx.is_self]
            .groupby(["from_acct", "to_acct"], observed=True)
            .agg(n_tx=("ts", "size"), usd=("amt_usd", "sum"), first_ts=("ts", "min"), last_ts=("ts", "max"))
            .reset_index())


def build_graph(edges: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_edges_from(((r.from_acct, r.to_acct, {"n_tx": r.n_tx, "usd": r.usd}) for r in edges.itertuples(index=False)))
    return G


def laundering_accounts(tx: pd.DataFrame) -> set:
    """Account label: involved in >= 1 laundering transaction in the slice."""
    return set(tx.loc[tx.is_laundering == 1, "from_acct"]) | set(tx.loc[tx.is_laundering == 1, "to_acct"])


def structural_signals(tx: pd.DataFrame, G: nx.DiGraph, k: int = 4) -> pd.DataFrame:
    """2-/3-cycles, daily fan-out/fan-in peaks and gather-scatter, one row per graph node."""
    nodes = list(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr", weight=None)
    two_cycle = np.asarray(A.multiply(A.T).sum(axis=1)).ravel()
    three_cycle = np.asarray((A @ A).multiply(A.T).sum(axis=1)).ravel()
    struct = pd.DataFrame({"two_cycle": two_cycle, "three_cycle": three_cycle}, index=pd.Index(nodes, name="acct"))

    fan_out_day = tx[~tx.is_self].groupby(["from_acct", "day"], observed=True).to_acct.nunique()
    fan_in_day = tx[~tx.is_self].groupby(["to_acct", "day"], observed=True).from_acct.nunique()
    struct["max_fan_out_day"] = fan_out_day.groupby(level=0).max().reindex(nodes).fillna(0).values
    struct["max_fan_in_day"] = fan_in_day.groupby(level=0).max().reindex(nodes).fillna(0).values

    # gather-scatter: fan-in >= k on day d and fan-out >= k on day d or d+1
    fi = fan_in_day[fan_in_day >= k].reset_index(); fo = fan_out_day[fan_out_day >= k].reset_index()
    fi.columns = ["acct", "day", "fi"]; fo.columns = ["acct", "day", "fo"]
    fo_next = fo.assign(day=fo.day - pd.Timedelta(days=1))
    gs = pd.concat([fi.merge(fo, on=["acct", "day"]), fi.merge(fo_next, on=["acct", "day"])]).acct.unique()
    struct["gather_scatter"] = struct.index.isin(gs).astype(int)
    return struct


def rule_lift(struct: pd.DataFrame, laundering: set, base_rate: float) -> pd.DataFrame:
    """Precision, recall and lift of each single structural rule (Section 5 table)."""
    y = struct.index.isin(laundering).astype(int)
    rules = {
        "2-cycle (A↔B)": struct.two_cycle > 0,
        "3-cycle (A→B→C→A)": struct.three_cycle > 0,
        "daily fan-out ≥ 8": struct.max_fan_out_day >= 8,
        "daily fan-in ≥ 8": struct.max_fan_in_day >= 8,
        "gather-scatter (k=4)": struct.gather_scatter == 1,
    }
    lift = pd.DataFrame([{"rule": name, "accounts_flagged": int(m.sum()),
                          "precision": y[m.values].mean(),
                          "recall": y[m.values].sum() / len(laundering)}
                         for name, m in rules.items()]).set_index("rule")
    lift["lift_vs_base"] = lift.precision / base_rate
    return lift