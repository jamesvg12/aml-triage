"""Notebook Sections 7 and 9: account-level profile and structural features."""
import logging
import time

import networkx as nx
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def profile_table(d: pd.DataFrame) -> pd.DataFrame:
    """Section 7's profile features on a slice of transactions."""
    fs = d.fmt.astype(str)
    f = d[["ts", "from_acct", "amt_usd", "pay_cur", "fmt", "to_bank", "day", "is_self", "amt_paid"]].assign(
        is_cash=(fs == "Cash"), is_crypto=(fs == "Bitcoin"), is_wire=(fs == "Wire"), is_round=(d.amt_paid % 100 == 0))
    o = f.groupby("from_acct", observed=True).agg(
        n_out=("ts", "size"), usd_out=("amt_usd", "sum"), usd_out_mean=("amt_usd", "mean"), usd_out_max=("amt_usd", "max"),
        n_cur_out=("pay_cur", "nunique"), n_fmt_out=("fmt", "nunique"), n_banks_out=("to_bank", "nunique"),
        active_days=("day", "nunique"), cash_share=("is_cash", "mean"), crypto_share=("is_crypto", "mean"),
        wire_share=("is_wire", "mean"), round_share=("is_round", "mean"), self_share=("is_self", "mean"))
    del f
    i = d.groupby("to_acct", observed=True).agg(
        n_in=("ts", "size"), usd_in=("amt_usd", "sum"), usd_in_mean=("amt_usd", "mean"), n_banks_in=("from_bank", "nunique"))
    pr = o.join(i, how="outer").fillna(0)
    pr["flow_ratio"] = np.log1p(pr.usd_out) - np.log1p(pr.usd_in)
    pr["pass_through"] = np.minimum(pr.usd_in, pr.usd_out) / (np.maximum(pr.usd_in, pr.usd_out) + 1)
    return pr


def structural_table(G: nx.DiGraph, struct: pd.DataFrame, seed: int, run_louvain: bool = True) -> pd.DataFrame:
    """Section 7's structural features: Section 5's signals plus degrees, PageRank, k-core, component/community size."""
    nodes = list(G.nodes())
    st = struct.copy()
    st["in_deg"] = pd.Series(dict(G.in_degree())); st["out_deg"] = pd.Series(dict(G.out_degree()))
    st["pagerank"] = pd.Series(nx.pagerank(G, weight="usd"))
    Gu = G.to_undirected()
    st["core"] = pd.Series(nx.core_number(Gu))
    comp = {n: i for i, c in enumerate(nx.weakly_connected_components(G)) for n in c}
    comp_size = pd.Series(comp).map(pd.Series(comp).value_counts())
    st["log_component_size"] = np.log10(comp_size)
    if run_louvain:
        t1 = time.time()
        comms = nx.community.louvain_communities(Gu, seed=seed)
        cid = {n: i for i, c in enumerate(comms) for n in c}
        st["log_community_size"] = np.log10(pd.Series(cid).map(pd.Series(cid).value_counts()))
        log.info("louvain: %s communities in %.0fs", f"{len(comms):,}", time.time() - t1)
    # neighbour aggregates: how 'hub-like' are the accounts you send to?
    st["nbr_mean_in_deg"] = pd.Series(
        {n: np.mean([G.in_degree(m) for m in G.successors(n)]) if G.out_degree(n) else 0 for n in nodes})
    return st


def account_features(tx: pd.DataFrame, G: nx.DiGraph, struct: pd.DataFrame, laundering: set,
                     seed: int, run_louvain: bool = True):
    """Section 7's account table. Returns (acct, PROFILE, STRUCT)."""
    prof = profile_table(tx)
    st = structural_table(G, struct, seed, run_louvain)
    acct = prof.join(st, how="left").fillna(0)
    acct["y"] = acct.index.isin(laundering).astype(int)
    return acct, list(prof.columns), list(st.columns)


def account_table(d: pd.DataFrame) -> pd.DataFrame:
    """Section 9: Section 7's profile and structural features on a slice of transactions (no Louvain); label from the same slice."""
    pr = profile_table(d)
    ns = d[~d.is_self]
    e = ns.groupby(["from_acct", "to_acct"], observed=True).agg(n_tx=("ts", "size"), usd=("amt_usd", "sum")).reset_index()
    g = nx.DiGraph()
    g.add_edges_from(((r.from_acct, r.to_acct, {"usd": r.usd}) for r in e.itertuples(index=False)))
    nl = list(g.nodes())
    a = nx.to_scipy_sparse_array(g, nodelist=nl, format="csr", weight=None)
    s = pd.DataFrame({"two_cycle": np.asarray(a.multiply(a.T).sum(axis=1)).ravel(),
                      "three_cycle": np.asarray((a @ a).multiply(a.T).sum(axis=1)).ravel()}, index=nl)
    fo_d = ns.groupby(["from_acct", "day"], observed=True).to_acct.nunique()
    fi_d = ns.groupby(["to_acct", "day"], observed=True).from_acct.nunique()
    s["max_fan_out_day"] = fo_d.groupby(level=0).max().reindex(nl).fillna(0).values
    s["max_fan_in_day"] = fi_d.groupby(level=0).max().reindex(nl).fillna(0).values
    fi_ = fi_d[fi_d >= 4].reset_index(); fo_ = fo_d[fo_d >= 4].reset_index()   # k = 4, as in Section 5
    fi_.columns = ["acct", "day", "fi"]; fo_.columns = ["acct", "day", "fo"]
    gs_ = pd.concat([fi_.merge(fo_, on=["acct", "day"]),
                     fi_.merge(fo_.assign(day=fo_.day - pd.Timedelta(days=1)), on=["acct", "day"])]).acct.unique()
    s["gather_scatter"] = s.index.isin(gs_).astype(int)
    s["in_deg"] = pd.Series(dict(g.in_degree())); s["out_deg"] = pd.Series(dict(g.out_degree()))
    s["pagerank"] = pd.Series(nx.pagerank(g, weight="usd"))
    s["core"] = pd.Series(nx.core_number(g.to_undirected()))
    cm = {n: j for j, c in enumerate(nx.weakly_connected_components(g)) for n in c}
    s["log_component_size"] = np.log10(pd.Series(cm).map(pd.Series(cm).value_counts()))
    s["nbr_mean_in_deg"] = pd.Series({n: np.mean([g.in_degree(m) for m in g.successors(n)]) if g.out_degree(n) else 0 for n in nl})
    t = pr.join(s, how="left").fillna(0)
    bad = set(d.loc[d.is_laundering == 1, "from_acct"]) | set(d.loc[d.is_laundering == 1, "to_acct"])
    t["y"] = t.index.isin(bad).astype(int)
    return t