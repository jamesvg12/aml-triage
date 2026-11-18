"""aml-score: nightly batch scoring with the trained model, through the same preprocessing code as training."""
import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

from .config import load_config
from .features import account_features
from .graph import build_edges, build_graph, laundering_accounts, structural_signals
from .load import load_transactions, prepare
from .train import motif

log = logging.getLogger("aml_triage")


def score_window(cfg, model_path, start=None, end=None):
    """Score every account active in [start, end) (default: the configured window)."""
    timings = {}
    bundle = joblib.load(model_path)
    model, cols = bundle["model"], bundle["features"]
    if start or end:
        cfg = dataclasses.replace(cfg, window_start=pd.Timestamp(start) if start else cfg.window_start,
                                  window_end=pd.Timestamp(end) if end else cfg.window_end)
    t = time.time()
    tx = prepare(load_transactions(cfg), cfg)                       # same input checks as training
    timings["load_s"] = time.time() - t
    t = time.time()
    G = build_graph(build_edges(tx))
    struct = structural_signals(tx, G)
    timings["graph_s"] = time.time() - t
    t = time.time()
    acct, PROFILE, STRUCT = account_features(tx, G, struct, laundering_accounts(tx), cfg.seed, cfg.run_louvain)
    timings["features_s"] = time.time() - t
    if PROFILE + STRUCT != cols:
        raise ValueError(f"feature mismatch: model expects {cols}, scoring built {PROFILE + STRUCT}")
    t = time.time()
    acct["score"] = model.predict_proba(acct[cols])[:, 1]
    timings["predict_s"] = time.time() - t
    out = acct.sort_values("score", ascending=False)
    out["rank"] = range(1, len(out) + 1)
    return out, bundle, timings, len(tx), cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aml-score", description="Batch-score accounts and write the analyst queue.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--model", default=None, help="default: <repo>/models/model.joblib")
    ap.add_argument("--start", default=None, help="window start (YYYY-MM-DD), default from config")
    ap.add_argument("--end", default=None, help="window end, exclusive, default from config")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        t0 = time.time()
        root = Path(args.config).resolve().parent.parent
        reports = root / "reports"; reports.mkdir(exist_ok=True)
        cfg = load_config(args.config)
        scored, bundle, timings, n_tx, cfg = score_window(cfg, args.model or root / "models" / "model.joblib",
                                                         args.start, args.end)
        timings["total_s"] = time.time() - t0
        queue = scored.head(cfg.analyst_capacity).copy()
        queue["why"] = queue.apply(motif, axis=1)
        scored[["score", "rank", "y"]].to_parquet(reports / "scores.parquet")
        queue[["rank", "score", "y", "why", "in_deg", "out_deg", "usd_in", "usd_out"]].to_csv(reports / "queue_scored.csv")
        run = {"model_git_commit": bundle["git_commit"], "window": [str(cfg.window_start.date()), str(cfg.window_end.date())],
               "transactions": n_tx, "accounts_scored": len(scored), "timings": timings,
               "throughput_accounts_per_s": len(scored) / timings["total_s"],
               "throughput_tx_per_s": n_tx / timings["total_s"],
               "queue_true_laundering": int(queue.y.sum())}
        (reports / "score_run.json").write_text(json.dumps(run, indent=2, default=str))
        log.info("scored %s accounts (%s tx) in %.0fs: %.0f accounts/s; top-%d holds %d laundering",
                 f"{len(scored):,}", f"{n_tx:,}", timings["total_s"], run["throughput_accounts_per_s"],
                 cfg.analyst_capacity, run["queue_true_laundering"])
        return 0
    except (FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 1
    except Exception:
        log.exception("unexpected failure")
        return 2


if __name__ == "__main__":
    sys.exit(main())