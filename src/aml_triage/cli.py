"""aml-train: run the given model end to end; write metrics, model and the analyst queue."""
import argparse
import gc
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

from .config import load_config
from .features import account_features
from .graph import build_edges, build_graph, laundering_accounts, structural_signals
from .load import load_transactions, prepare
from .train import build_queue, fit_eval, forward_eval, holdout_fingerprint, split_accounts, summary

log = logging.getLogger("aml_triage")


def _git(*args) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def run(config_path: str, skip_forward: bool = False) -> dict:
    t_start, timings = time.time(), {}
    cfg = load_config(config_path)
    root = Path(config_path).resolve().parent.parent
    reports, models = root / "reports", root / "models"
    reports.mkdir(exist_ok=True); models.mkdir(exist_ok=True)

    t = time.time()
    tx = prepare(load_transactions(cfg), cfg)
    timings["load_s"] = time.time() - t
    log.info("%s transactions in window", f"{len(tx):,}")

    t = time.time()
    G = build_graph(build_edges(tx))
    struct = structural_signals(tx, G)
    laundering = laundering_accounts(tx)
    n_acc = pd.concat([tx.from_acct, tx.to_acct]).nunique()
    timings["graph_s"] = time.time() - t
    log.info("graph: %s nodes, %s edges", f"{G.number_of_nodes():,}", f"{G.number_of_edges():,}")

    t = time.time()
    acct, PROFILE, STRUCT = account_features(tx, G, struct, laundering, cfg.seed, cfg.run_louvain)
    timings["features_s"] = time.time() - t
    log.info("%s accounts, %d profile + %d structural features", f"{len(acct):,}", len(PROFILE), len(STRUCT))

    idx_tr, idx_te = split_accounts(acct, cfg.seed)
    results = {}
    for cols, name in [(PROFILE, "profile only"), (STRUCT, "structural only"), (PROFILE + STRUCT, "profile + structural")]:
        m, p, res = fit_eval(acct, idx_tr, idx_te, cols, name, cfg.seed)
        results[name] = res
        log.info("%s: AUPRC %.3f, p@200 %.3f, fit %.1fs", name, res["AUPRC"], res["p@200"], res["fit_s"])
    m_all, p_all = m, p                       # the last fit is the given model (profile + structural)
    timings["train_s"] = results["profile + structural"]["fit_s"]

    queue = build_queue(acct, idx_te, p_all, cfg.analyst_capacity)
    y_te = acct.y.values[idx_te]
    fingerprint = holdout_fingerprint(acct, idx_te)

    forward = {}
    if not skip_forward:
        del G, struct; gc.collect()           # as in the notebook: free Sections 5-7's working copies
        t = time.time()
        fwd_cols = [c for c in PROFILE + STRUCT if c != "log_community_size"]   # Louvain left out on both sides
        _, p_naive, _ = fit_eval(acct, idx_tr, idx_te, fwd_cols, "transductive, no community size", cfg.seed)
        tr_tab, te_tab, p_fwd, seen = forward_eval(tx, fwd_cols, cfg.forward_split, cfg.seed)
        forward = {
            "transductive_no_community": summary(y_te, p_naive, "transductive, no community size", cfg.analyst_capacity),
            "forward": summary(te_tab.y.values, p_fwd, "forward", cfg.analyst_capacity),
            "forward_new_accounts": summary(te_tab.y.values[~seen], p_fwd[~seen], "forward, new accounts", cfg.analyst_capacity),
            "train_window_accounts": len(tr_tab), "test_window_new_accounts": int((~seen).sum()),
        }
        timings["forward_s"] = time.time() - t
        log.info("forward: AUPRC %.3f, lift %.1fx; new accounts lift %.1fx",
                 forward["forward"]["AUPRC"], forward["forward"]["lift"], forward["forward_new_accounts"]["lift"])

    timings["total_s"] = time.time() - t_start
    metrics = {
        "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "config": {k: str(v) for k, v in vars(cfg).items()},
        "accounts": len(acct), "n_profile": len(PROFILE), "n_struct": len(STRUCT),
        "laundering_accounts": len(laundering), "base_rate": len(laundering) / n_acc,
        "holdout": {"accounts": len(idx_te), "laundering": int(y_te.sum()), "sha256": fingerprint},
        "results": results,
        "headline": summary(y_te, p_all, "transductive (headline)", cfg.analyst_capacity),
        "queue": {"size": len(queue), "true_laundering": int(queue.y.sum()),
                  "no_motif_share": float((queue.why == "no simple motif (model-only)").mean())},
        "forward": forward, "timings": timings,
    }
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    joblib.dump({"model": m_all, "features": PROFILE + STRUCT, "git_commit": metrics["git_commit"]}, models / "model.joblib")
    queue.to_csv(reports / "queue_top200.csv")
    log.info("done in %.0fs: AUPRC %.3f, top-%d %d/%d, sha256 %s", timings["total_s"], metrics["headline"]["AUPRC"],
             cfg.analyst_capacity, queue.y.sum(), len(queue), fingerprint[:16])
    return metrics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aml-train", description="Train and evaluate the given AML triage model.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--skip-forward", action="store_true", help="skip the forward-in-time evaluation (saves ~75 s)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        run(args.config, args.skip_forward)
        return 0
    except (FileNotFoundError, ValueError) as e:      # missing data / config, unmapped currency
        log.error("%s", e)
        return 1
    except Exception:
        log.exception("unexpected failure")
        return 2


if __name__ == "__main__":
    sys.exit(main())