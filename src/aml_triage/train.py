"""Notebook Sections 7, 9, 10: the given LightGBM account model, its evaluation and the analyst queue."""
import hashlib
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .features import account_table

KS = [100, 200, 500, 1000, 2000, 3000]
# The given model's learner and settings (identical in the notebook's fit_eval and fit_score).
LGBM_PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=63, min_child_samples=40,
                   scale_pos_weight=20, subsample=0.8, colsample_bytree=0.8, verbose=-1)


def make_model(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(**LGBM_PARAMS, random_state=seed)


def precision_at_k(y_true, score, k):
    top = np.argsort(-np.asarray(score))[:k]
    y = np.asarray(y_true)
    return y[top].mean(), y[top].sum() / y.sum()


def split_accounts(acct: pd.DataFrame, seed: int):
    """Section 7: 70/30 random split of accounts, stratified on the label (transductive)."""
    return train_test_split(np.arange(len(acct)), test_size=0.3, random_state=seed, stratify=acct.y)


def fit_eval(acct, idx_tr, idx_te, cols, name, seed, ks=KS):
    t1 = time.time()
    y_tr, y_te = acct.y.values[idx_tr], acct.y.values[idx_te]
    m = make_model(seed)
    m.fit(acct.iloc[idx_tr][cols], y_tr)
    p = m.predict_proba(acct.iloc[idx_te][cols])[:, 1]
    res = {"model": name, "AUPRC": average_precision_score(y_te, p), "AUC": roc_auc_score(y_te, p)}
    res.update({f"p@{k}": precision_at_k(y_te, p, k)[0] for k in ks})
    res["fit_s"] = time.time() - t1
    return m, p, res


def fit_score(X_tr, y_tr_, X_te, seed):
    """The given model's learner and settings (as in fit_eval), on any train/test pair."""
    m = make_model(seed)
    m.fit(X_tr, y_tr_)
    return m.predict_proba(X_te)[:, 1]


def summary(y, p, name, capacity):
    return {"evaluation": name, "accounts": len(y), "laundering": int(np.sum(y)), "base rate": np.mean(y),
            "AUPRC": average_precision_score(y, p), "lift": average_precision_score(y, p) / np.mean(y),
            "p@200": precision_at_k(y, p, capacity)[0], "p@1000": precision_at_k(y, p, 1000)[0]}


def motif(r):
    tags = []
    if r.three_cycle > 0: tags.append("3-cycle")
    if r.two_cycle > 0: tags.append("2-cycle")
    if r.gather_scatter: tags.append("gather-scatter")
    if r.max_fan_out_day >= 8: tags.append(f"fan-out {int(r.max_fan_out_day)}/day")
    if r.max_fan_in_day >= 8: tags.append(f"fan-in {int(r.max_fan_in_day)}/day")
    return ", ".join(tags) or "no simple motif (model-only)"


def build_queue(acct, idx_te, p, capacity):
    """Section 7: the top-`capacity` held-out accounts, with the motif that explains each."""
    queue = acct.iloc[idx_te][["y", "two_cycle", "three_cycle", "max_fan_out_day", "max_fan_in_day", "gather_scatter",
                               "in_deg", "out_deg", "usd_in", "usd_out"]].copy()
    queue["score"] = p
    queue = queue.sort_values("score", ascending=False).head(capacity)
    queue["why"] = queue.apply(motif, axis=1)
    return queue


def forward_eval(tx, cols, forward_split, seed):
    """Section 9, Trap 1: learn on the accounts of days before forward_split, score the accounts after it."""
    tr_tab = account_table(tx[tx.ts < forward_split])
    te_tab = account_table(tx[tx.ts >= forward_split])
    p_fwd = fit_score(tr_tab[cols], tr_tab.y, te_tab[cols], seed)
    seen = te_tab.index.isin(tr_tab.index)
    return tr_tab, te_tab, p_fwd, seen


def holdout_fingerprint(acct, idx_te) -> str:
    """Section 10: SHA-256 of the frozen evaluation split."""
    holdout = acct.iloc[idx_te][["y"]].reset_index()
    return hashlib.sha256(pd.util.hash_pandas_object(holdout, index=False).values.tobytes()).hexdigest()