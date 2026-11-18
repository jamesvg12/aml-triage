"""Training vs scoring parity: held-out scores from aml-train must equal aml-score's scores for the same accounts."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
train = pd.read_csv(ROOT / "reports" / "holdout_scores.csv", index_col=0)
serve = pd.read_parquet(ROOT / "reports" / "scores.parquet")
train.index, serve.index = train.index.astype(str), serve.index.astype(str)
j = train.join(serve["score"].rename("served"), how="left")
diff = (j.score - j.served).abs()
same_top = set(j.score.nlargest(200).index) == set(j.served.nlargest(200).index)
print(f"held-out accounts {len(j):,} · missing at scoring {int(j.served.isna().sum())} · "
      f"max |train - serve| {diff.max():.2e} · top-200 identical {same_top}")