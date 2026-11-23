CREATE TABLE IF NOT EXISTS predictions (
    run_id        uuid             NOT NULL,
    scored_at     timestamptz      NOT NULL,
    model_version text             NOT NULL,
    window_start  date             NOT NULL,
    window_end    date             NOT NULL,
    acct          text             NOT NULL,
    score         double precision NOT NULL,
    rank          integer          NOT NULL,
    PRIMARY KEY (run_id, acct)
);
CREATE TABLE IF NOT EXISTS score_runs (
    run_id                uuid PRIMARY KEY,
    scored_at             timestamptz,
    model_version         text,
    window_start          date,
    window_end            date,
    transactions          integer,
    accounts              integer,
    total_s               double precision,
    accounts_per_s        double precision,
    queue_true_laundering integer
);