"""Entry 1, task 2: check the data against its card. Writes reports/data_check.json."""
import json
from pathlib import Path

import pandas as pd

from aml_triage.config import load_config
from aml_triage.load import FX, load_patterns, load_transactions, prepare

ROOT = Path(__file__).resolve().parent.parent
cfg = load_config(ROOT / "config" / "config.yaml")
tx = prepare(load_transactions(cfg), cfg)                 # the 10-day window the model uses
full = load_transactions(cfg)                             # all 18 days
full["day"] = full.ts.dt.floor("D")
daily = full.groupby("day").agg(n=("ts", "size"), laundering=("is_laundering", "sum"))
pt = load_patterns(cfg)
out = {}

# ---- quirks the card lists ----
tail = full[full.ts >= cfg.window_end]
out["card_1_tail"] = {"last_day_over_10k_tx": str(daily[daily.n > 10_000].index.max().date()),
                      "tail_rows": len(tail), "tail_laundering": int(tail.is_laundering.sum()),
                      "tail_laundering_share": round(float(tail.is_laundering.mean()), 3),
                      "window_laundering_share": round(float(tx.is_laundering.mean()), 5)}

d1 = full[full.day == pd.Timestamp("2022-09-01")]
out["card_2_day1_burst"] = {"day1_rows": len(d1),
                            "day1_reinvestment_share": round(float((d1.fmt.astype(str) == "Reinvestment").mean()), 3),
                            "median_rows_days_2_10": int(daily.loc["2022-09-02":"2022-09-10", "n"].median())}

selfs = tx[tx.is_self]
out["card_3_self_transfers"] = {"rows": len(selfs), "share": round(float(tx.is_self.mean()), 4),
                                "laundering_among_them": int(selfs.is_laundering.sum()),
                                "top_formats": selfs.fmt.astype(str).value_counts().head(3).to_dict()}

pay, recv = tx.pay_cur.astype(str), tx.recv_cur.astype(str)
out["card_4_currencies"] = {"pay_currencies": int(pay.nunique()), "recv_currencies": int(recv.nunique()),
                            "unmapped_in_FX": sorted(set(pay) - set(FX)),
                            "cross_currency_rows": int((pay != recv).sum())}

att = pt.groupby(["attempt", "pattern"]).ts.agg(["min", "max"]).reset_index()
att["hours"] = (att["max"] - att["min"]).dt.total_seconds() / 3600
out["card_5_patterns_span"] = {"attempts": len(att),
                               "median_hours_by_pattern": att.groupby("pattern").hours.median().round(1).to_dict(),
                               "share_over_24h": round(float((att.hours > 24).mean()), 3),
                               "straddle_forward_split": int(((att["min"] < cfg.forward_split) & (att["max"] >= cfg.forward_split)).sum())}

lt = full[full.is_laundering == 1]
keys = pt[["ts", "from_acct", "to_acct"]].astype({"from_acct": str, "to_acct": str}).drop_duplicates()
out["card_6_typed_share"] = {"laundering_tx_all_days": len(lt),
                             "matched_to_typed_attempt": len(lt.merge(keys, on=["ts", "from_acct", "to_acct"]))}

# ---- candidates the card does not list ----
ab = pd.concat([full[["from_acct", "from_bank"]].set_axis(["acct", "bank"], axis=1),
                full[["to_acct", "to_bank"]].set_axis(["acct", "bank"], axis=1)]).drop_duplicates()
banks_per_acct = ab.groupby("acct").bank.nunique()
out["new_a_account_id_at_several_banks"] = {"accounts": int((banks_per_acct > 1).sum()),
                                            "max_banks": int(banks_per_acct.max())}
out["new_b_exact_duplicate_rows"] = int(full.duplicated().sum())
out["new_c_same_currency_amount_mismatch"] = int(((pay == recv) & (tx.amt_paid != tx.amt_recv)).sum())
out["new_d_amount_under_1"] = {"rows": int((tx.amt_paid < 1).sum()),
                               "laundering": int(tx.loc[tx.amt_paid < 1, "is_laundering"].sum())}
fmt_l = tx.groupby(tx.fmt.astype(str)).is_laundering.sum()
out["new_e_formats_with_zero_laundering"] = {f: int(n) for f, n in tx.fmt.astype(str).value_counts().items() if fmt_l[f] == 0}

(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / "data_check.json").write_text(json.dumps(out, indent=2, default=str))
print(json.dumps(out, indent=2, default=str))