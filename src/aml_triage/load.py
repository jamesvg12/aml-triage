"""Notebook Section 1 (+ load_patterns from Section 3): raw files, transactions, FX table."""
import logging
import re
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pc

from .config import Config

log = logging.getLogger(__name__)

# Fixed FX approximation to USD (September 2022). An assumption, not a fact: Trap 2 measures what it moves.
FX = {"US Dollar": 1.0, "Euro": 1.0, "UK Pound": 1.15, "Swiss Franc": 1.03, "Canadian Dollar": 0.76,
      "Australian Dollar": 0.68, "Yen": 0.0070, "Yuan": 0.145, "Rupee": 0.0125, "Shekel": 0.29,
      "Saudi Riyal": 0.27, "Mexican Peso": 0.050, "Brazil Real": 0.19, "Ruble": 0.0165, "Bitcoin": 20000.0}


def get_raw(filename: str, cfg: Config) -> Path:
    """Return a local path to a raw file, downloading from Kaggle on first use.
    Kaggle credentials: set KAGGLE_API_TOKEN (access token) or place kaggle.json in ~/.kaggle/.
    """
    local = cfg.raw / filename
    if local.exists():
        return local
    import kagglehub
    path = Path(kagglehub.dataset_download(cfg.kaggle_handle, path=filename))
    return path


def load_transactions(cfg: Config) -> pd.DataFrame:
    """Load HI-Small_Trans.csv into a compact frame (categoricals via Arrow dictionary encoding), cached as parquet."""
    cache = cfg.processed / "aml_hi_small.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    t = time.time()
    convert = pc.ConvertOptions(column_types={"From Bank": pa.string(), "To Bank": pa.string()}, auto_dict_encode=True)
    df = pc.read_csv(get_raw(cfg.files["transactions"], cfg), convert_options=convert).to_pandas()
    df.columns = ["ts", "from_bank", "from_acct", "to_bank", "to_acct", "amt_recv", "recv_cur",
                  "amt_paid", "pay_cur", "fmt", "is_laundering"]
    df["ts"] = pd.to_datetime(df["ts"].astype(str), format="%Y/%m/%d %H:%M")
    for c in ["from_bank", "from_acct", "to_bank", "to_acct"]:
        df[c] = df[c].astype(str)
    df.to_parquet(cache)
    log.info("loaded + cached in %.1fs", time.time() - t)
    return df


def prepare(tx: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Notebook cells 5-6: keep the window, convert to USD, add day and self-transfer flag."""
    tx = tx[(tx.ts >= cfg.window_start) & (tx.ts < cfg.window_end)].reset_index(drop=True)
    unmapped = set(tx.pay_cur.astype(str).unique()) - set(FX)
    if unmapped:
        raise ValueError(f"unmapped currency: {sorted(unmapped)}")
    tx["amt_usd"] = tx.amt_paid * tx.pay_cur.astype(str).map(FX)
    tx["day"] = tx.ts.dt.floor("D")
    tx["is_self"] = tx.from_acct == tx.to_acct
    return tx


def load_patterns(cfg: Config) -> pd.DataFrame:
    """The injected laundering attempts with their typology (notebook Section 3)."""
    rows, cur, attempt = [], None, -1
    for line in open(get_raw(cfg.files["patterns"], cfg)):
        m = re.match(r"BEGIN LAUNDERING ATTEMPT - ([A-Z\- ]+?):", line)
        if m:
            cur, attempt = m.group(1).strip(), attempt + 1
            continue
        if line.startswith("END"):
            cur = None
            continue
        if cur and line.strip():
            p = line.strip().split(",")
            rows.append((attempt, cur, p[0], p[1], p[2], p[3], p[4], float(p[7]), p[8], p[9]))
    pt = pd.DataFrame(rows, columns=["attempt", "pattern", "ts", "from_bank", "from_acct", "to_bank", "to_acct",
                                     "amt_paid", "pay_cur", "fmt"])
    pt["ts"] = pd.to_datetime(pt.ts, format="%Y/%m/%d %H:%M")
    return pt