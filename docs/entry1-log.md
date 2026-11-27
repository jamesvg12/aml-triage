# Entry 1 log

## Reference run (original notebook, unchanged)
- Date: 2026-11-18
- Env: WSL2 Ubuntu 26.04, Python 3.14.4, pandas 3.1.1, networkx 3.7, numpy 2.5.5, 6 cores / 10 GB
- Total wall time: 354 s
- Load + parquet cache: 24.3 s (download 454 MB in 13 s)
- Window: 5,077,237 transactions, 515,070 accounts, 4,522 laundering tx (0.09%), self-transfers 11.6%
- Graph: 422,648 nodes, 647,316 edges, 21,817 WCCs (largest 88%)
- Features: 19 profile + 12 structural, built in 100 s (Louvain 79 s)
- Held-out: 154,521 accounts, 1,759 laundering, SHA-256 d4d1fbf90a2856e4...

| Model | AUPRC | AUC | p@200 | p@1000 | fit_s |
|---|---|---|---|---|---|
| profile only | 0.265 | 0.885 | 98% | 38% | 7 |
| structural only | 0.100 | 0.843 | 41% | 19% | 6 |
| profile + structural | 0.292 | 0.909 | 99% | 40% | 10 |

- Top-200 queue: 198 true laundering; 92% no single-motif explanation
- Forward (train days 1-5, test 6-10): 372,954 accounts, 3,357 laundering, AUPRC 0.247, lift 27.5x, p@200 100%, p@1000 62%
- New accounts in days 6-10: 2,766 (76 laundering), lift 8.6x, p@200 17%

## Quirks seen in reference run
- Tail: 11 Sep has 396 tx, 58.6% laundering
- ACH: 3,828 of 4,522 laundering tx, rate 0.638%
- Self-loops as edges: 2-cycle flags 368,771 accounts vs 6,680
- FX: no conversion -> 18/200 top accounts change, AUPRC unchanged

## Notes
- Notebook first ran from repo root: parquet cache landed in ~/data (DATA_DIR = cwd.parent.parent). Fixed by setting DATASETS_TASKS_DATA in .env.

## Packaged pipeline (aml-train), clean run
- Commit 30d4c31, git_dirty False
- AUPRC 0.292 (0.265 / 0.100 / 0.292 by feature family), top-200 198/200, no-motif 92%
- Forward AUPRC 0.247, lift 27.5x; new-account lift 8.6x
- Holdout 154,521 / 1,759, SHA-256 d4d1fbf90a2856e4... (identical to notebook)
- Timings: load 3 s, graph 14 s, features 106 s, train 10 s, forward 67 s, total 217 s
- graph_s varied 120 / 49 / 14 s across runs: timings must be medians
- Tests: 16 passed, 1 xfailed (4-node cycle not detected by the given signals)
- prepare() on an empty window crashes with a TypeError (pandas 3): candidate input check
- Notebook paths depended on the launch directory twice (~/data, ../config): fixed by resolving from the repo root

## Step 2: data check (commit ba2449c, reports/data_check.json)
- Tail: last day >10k tx = 10 Sep; days 11-18 = 1,108 tx, 655 laundering (59.1%) vs 0.089% in window
- Day 1: 1,114,921 rows (2.3x median day 482,650), 43% Reinvestment
- Self-transfers: 590,819 (11.6%), only 8 laundering
- Currencies: 15/15 mapped, 71,779 cross-currency rows
- Patterns: medians 71.9/84.9/76.7/150.8 h; 80% > 24 h; 61 of 234 attempts straddle the forward split (6 Sep)
- Typed: 1,854 of 5,177
- NOT ON CARD: Wire = 171,855 transfers, 0 laundering (also: 9 duplicate rows, 8 multi-bank account IDs, 147,791 amounts < 1)
- Guard (commit 372e321): prepare() raises if any day > 5% laundering. With window_end 2022-09-19:
  "ERROR window reaches the laundering-only tail: 2022-09-11 is 59% laundering (limit 5%)", exit 1
- Guard trade-off: share threshold also trips on tiny fixtures (test needed 100 background rows); real feed would need a min-volume rule
- Tests: 20 passed, 1 xfailed

## Step 3: reproducible + container (commits a95ca81, 1cd6b7f)
- aml-score local: 515,070 accounts / 5,077,237 tx in 107 s = 4,820 accounts/s
- Image aml-triage:0.1.0 = 949 MB on disk (219 MB content); python:3.14-slim + libgomp1 + requirements-serve.txt
- Container: 156 s = 3,291 accounts/s (about 45% slower than venv; to confirm with 3-run medians)
- Parity train vs serve (container): 154,521 held-out, 0 missing, max |diff| 0.00e+00, top-200 identical
- Scoring all accounts puts 200/200 laundering in the top 200: training accounts are memorised; honest numbers are held-out 198/200 and forward
- Container logs in UTC, host in CET: store UTC timestamps in the prediction log
- Fresh clone: see Step 3a (identical numbers, commit 886d653)

## Step 4: monitoring (compose: postgres, pushgateway, prometheus, grafana)
- Every prediction logged: predictions(run_id, scored_at UTC, model_version, window, acct, score, rank); 3 runs x 515,070 rows
- model_version = training commit + model-file hash: 372e321f-729f45c5b275
- Metrics pushed: aml_accounts_scored, aml_run_duration_seconds{stage}, aml_throughput_accounts_per_second,
  aml_input_transfers_last_day (input feature), aml_input_usd_out_median, aml_queue_cutoff_score
- Load: 3 full-window runs (5,077,237 tx each), caches dropped between runs, grafana+prometheus paused
- Runs: 136 / 151 / 132 s -> MEDIAN 136 s, 3,786 accounts/s (~37k tx/s); prediction-log write ~6.5 s (log stage); DB median 130 s excludes it, end-to-end log-line median 136 s
- First attempt: prometheus_client missing from image (freeze writes prometheus_client, grep looked for prometheus-client);
  predictions were logged but metrics not pushed, exit 2 -> partial failure invisible from either side alone
- Memory: Windows at 92% with vmmem 5.9 GB before runs; had to close apps + drop caches to get clean timings

## Step 3a: fresh-copy reproduction (commit 886d653)
- git clone -> new venv -> pip install -r requirements.txt -> pytest (20 passed, 1 xfailed) -> aml-train
- Separate KAGGLEHUB_CACHE: data re-downloaded (454 MB in 13 s), parsed + cached in 18.7 s; data/processed 126 MB
- Result identical: AUPRC 0.292, top-200 198/200, forward 0.247, new-account lift 8.6x, SHA-256 d4d1fbf90a2856e4; 198 s
- Tolerance stated: exact match on AUPRC (4 dp), top-200 count and holdout SHA-256
- FIRST ATTEMPT WAS INVALID: it "passed" without downloading anything. Cause: (1) VS Code terminal injected
  DATASETS_TASKS_DATA from .env into the shell, so the fresh clone read the project's data; (2) .env had a missing
  newline, gluing DATASETS_TASKS_DATA onto the token line, so copying "only the token" copied the path too.
  Caught only by checking that .kagglehub existed. Fix: rewrote .env one key per line; reran from a clean shell.
