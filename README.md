# aml-triage — ARI3217 Task 01: graph-based AML triage

The given account-level triage model from `notebooks/reference/01_aml_graph_triage.ipynb`,
packaged unchanged: nightly batch scoring that ranks accounts for an AML analyst queue (top 200).

## Data
- IBM Transactions for Anti-Money Laundering, `HI-Small` split (Kaggle: `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`).
- Licence: CDLA-Sharing-1.0 (derived data carries the same licence). Synthetic, no personal data.
- **Sampling rule:** transactions with `WINDOW_START = 2022-09-01` ≤ ts < `WINDOW_END = 2022-09-11`
  (after 10 Sep the data is laundering-only). Forward split at `2022-09-06`. See `config/config.yaml`.
- The data is downloaded on first run with `kagglehub`. Put `KAGGLE_API_TOKEN=...` in `.env`
  (or `~/.kaggle/kaggle.json`). `data/` is never committed.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```
Tested on Python 3.14.4 (WSL2 Ubuntu 26.04, 6 cores / 10 GB). Peak run about 4 min.

## Run
```bash
aml-train            # writes reports/metrics.json, reports/queue_top200.csv, models/model.joblib
pytest -v            # unit tests on hand-built data (no dataset needed)
```

## Reproduced numbers (commit 30d4c31)
| | Notebook | Package |
|---|---|---|
| Accounts / features | 515,070 · 19 + 12 | 515,070 · 19 + 12 |
| Held-out | 154,521 (1,759 laundering) | same, SHA-256 `d4d1fbf90a2856e4…` |
| AUPRC (transductive) | 0.292 | 0.292 |
| Top-200 true laundering | 198 | 198 |
| Forward AUPRC (days 6–10) | 0.247 | 0.247 |
| New-account lift | 8.6× | 8.6× |

## Layout
`src/aml_triage/` — `config.py`, `load.py` (data, FX table), `graph.py` (edges without self-loops, structural signals),
`features.py`, `train.py` (the given LightGBM model, evaluation, queue), `cli.py` (`aml-train`).
`notebooks/01_aml_triage.ipynb` is a thin client; `notebooks/reference/` holds the original.