"""Step 4: log every prediction to Postgres; push batch-run metrics to a Prometheus Pushgateway."""
import io
import logging

import pandas as pd

log = logging.getLogger(__name__)


def log_predictions(scored: pd.DataFrame, run: dict, dsn: str) -> None:
    """One row per scored account, bulk-loaded with COPY, plus one summary row per run."""
    import psycopg
    rows = pd.DataFrame({"run_id": run["run_id"], "scored_at": run["scored_at"], "model_version": run["model_version"],
                         "window_start": run["window"][0], "window_end": run["window"][1],
                         "acct": scored.index.astype(str), "score": scored.score.values, "rank": scored["rank"].values})
    buf = io.StringIO()
    rows.to_csv(buf, index=False, header=False)
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        with cur.copy("COPY predictions (run_id, scored_at, model_version, window_start, window_end, acct, score, rank) "
                      "FROM STDIN WITH (FORMAT csv)") as cp:
            cp.write(buf.getvalue())
        cur.execute("INSERT INTO score_runs VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (run["run_id"], run["scored_at"], run["model_version"], run["window"][0], run["window"][1],
                     run["transactions"], run["accounts_scored"], run["timings"]["total_s"],
                     run["throughput_accounts_per_s"], run["queue_true_laundering"]))
    log.info("logged %s predictions for run %s", f"{len(rows):,}", run["run_id"])


def push_metrics(run: dict, scored: pd.DataFrame, tx_per_day: pd.Series, capacity: int, gateway: str) -> None:
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
    reg, mv = CollectorRegistry(), run["model_version"]

    def g(name, doc, value):
        Gauge(name, doc, ["model_version"], registry=reg).labels(mv).set(value)

    g("aml_accounts_scored", "Accounts scored in the last batch run (request count)", run["accounts_scored"])
    g("aml_transactions_in_window", "Transactions in the scored window", run["transactions"])
    g("aml_throughput_accounts_per_second", "Accounts scored per second, end to end", run["throughput_accounts_per_s"])
    g("aml_input_transfers_last_day", "Transfers on the last day of the window (normal 208k-654k; tail 396)",
      int(tx_per_day.iloc[-1]))
    g("aml_input_usd_out_median", "Median outgoing USD per account in the window", float(scored.usd_out.median()))
    g("aml_queue_cutoff_score", "Score of the last account in the analyst queue", float(scored.score.iloc[capacity - 1]))
    d = Gauge("aml_run_duration_seconds", "Batch run duration by stage (latency)", ["model_version", "stage"], registry=reg)
    for stage, secs in run["timings"].items():
        d.labels(mv, stage.removesuffix("_s")).set(secs)
    Gauge("aml_last_run_timestamp_seconds", "When the last batch run finished", registry=reg).set_to_current_time()
    push_to_gateway(gateway, job="aml_score", registry=reg)
    log.info("pushed metrics to %s", gateway)