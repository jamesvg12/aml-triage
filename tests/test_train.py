import numpy as np
import pandas as pd

from aml_triage.train import holdout_fingerprint, motif, precision_at_k


def test_precision_and_recall_at_k():
    assert precision_at_k([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], 2) == (0.5, 0.5)


def _row(**kw):
    base = dict(three_cycle=0, two_cycle=0, gather_scatter=0, max_fan_out_day=0, max_fan_in_day=0)
    return pd.Series({**base, **kw})


def test_motif_tags():
    assert motif(_row(three_cycle=1, max_fan_out_day=9)) == "3-cycle, fan-out 9/day"
    assert motif(_row()) == "no simple motif (model-only)"


def test_fingerprint_is_stable_and_label_sensitive():
    acct = pd.DataFrame({"y": [0, 1, 0, 1]}, index=list("ABCD"))
    idx = np.array([0, 1, 3])
    flipped = acct.assign(y=[0, 0, 0, 1])
    assert holdout_fingerprint(acct, idx) == holdout_fingerprint(acct, idx)
    assert holdout_fingerprint(acct, idx) != holdout_fingerprint(flipped, idx)