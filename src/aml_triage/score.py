"""aml-score: nightly batch scoring with the trained model, through the same preprocessing code as training."""
import argparse
import dataclasses
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
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
    tx_per_day = tx.groupby("day").size()
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
    return out, bundle, timings, len(tx), tx_per_day, cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aml-score", description="Batch-score accounts and write the analyst queue.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--model", default=None, help="default: <repo>/models/model.joblib")
    ap.add_argument("--start", default=None, help="window start (YYYY-MM-DD), default from config")
    ap.add_argument("--end", default=None, help="window end, exclusive, default from config")
    ap.add_argument("--db-url", default=os.environ.get("DATABASE_URL"), help="Postgres DSN for the prediction log")
    ap.add_argument("--pushgateway", default=os.environ.get("PUSHGATEWAY_URL"), help="host:port of the Pushgateway")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        t0 = time.time()
        root = Path(args.config).resolve().parent.parent
        reports = root / "reports"; reports.mkdir(exist_ok=True)
        model_path = Path(args.model) if args.model else root / "models" / "model.joblib"
        cfg = load_config(args.config)
        scored, bundle, timings, n_tx, tx_per_day, cfg = score_window(cfg, model_path, args.start, args.end)
        queue = scored.head(cfg.analyst_capacity).copy()
        queue["why"] = queue.apply(motif, axis=1)
        scored[["score", "rank", "y"]].to_parquet(reports / "scores.parquet")
        queue[["rank", "score", "y", "why", "in_deg", "out_deg", "usd_in", "usd_out"]].to_csv(reports / "queue_scored.csv")
        model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]
        run = {"run_id": str(uuid.uuid4()), "scored_at": datetime.now(timezone.utc).isoformat(),
               "model_version": f"{bundle['git_commit'][:8]}-{model_sha}",
               "window": [str(cfg.window_start.date()), str(cfg.window_end.date())],
               "transactions": n_tx, "accounts_scored": len(scored),
               "queue_true_laundering": int(queue.y.sum()), "timings": timings}
        if args.db_url:
            t = time.time()
            from .monitor import log_predictions
            timings["total_s"] = time.time() - t0               # provisional, for the score_runs row
            run["throughput_accounts_per_s"] = len(scored) / timings["total_s"]
            log_predictions(scored, run, args.db_url)
            timings["log_s"] = time.time() - t
        timings["total_s"] = time.time() - t0
        run["throughput_accounts_per_s"] = len(scored) / timings["total_s"]
        if args.pushgateway:
            from .monitor import push_metrics
            push_metrics(run, scored, tx_per_day, cfg.analyst_capacity, args.pushgateway)
        (reports / "score_run.json").write_text(json.dumps(run, indent=2, default=str))
        log.info("run %s · model %s · %s accounts (%s tx) in %.0fs: %.0f accounts/s; top-%d holds %d laundering",
                 run["run_id"][:8], run["model_version"], f"{len(scored):,}", f"{n_tx:,}", timings["total_s"],
                 run["throughput_accounts_per_s"], cfg.analyst_capacity, run["queue_true_laundering"])
        return 0
    except (FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 1
    except Exception:
        log.exception("unexpected failure")
        return 2


if __name__ == "__main__":
    sys.exit(main())