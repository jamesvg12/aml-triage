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
