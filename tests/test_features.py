import numpy as np
import pandas as pd

from aml_triage.features import account_table, profile_table
from conftest import make_tx


def test_profile_counts_amounts_and_shares():
    tx = pd.concat([make_tx([("A", "B"), ("A", "B")]), make_tx([("B", "A")], amt=50.0, fmt="Cash")], ignore_index=True)
    p = profile_table(tx)
    assert p.loc["A", "n_out"] == 2 and p.loc["A", "usd_out"] == 200.0
    assert p.loc["B", "n_in"] == 2 and p.loc["B", "cash_share"] == 1.0
    assert np.isclose(p.loc["A", "flow_ratio"], np.log1p(200) - np.log1p(50))


def test_account_label_marks_both_ends_of_a_laundering_transfer():
    normal = [(f"N{i}", f"M{i}") for i in range(100)]          # background traffic keeps the day below the tail limit
    t = account_table(make_tx([("A", "B"), ("B", "C")] + normal, laundering={("A", "B")}))
    assert t.loc[["A", "B", "C"], "y"].to_dict() == {"A": 1, "B": 1, "C": 0}
    assert t.y.sum() == 2