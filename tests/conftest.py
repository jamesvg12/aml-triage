import pandas as pd
import pytest
from types import SimpleNamespace

from aml_triage.load import prepare

WINDOW = SimpleNamespace(window_start=pd.Timestamp("2022-09-01"), window_end=pd.Timestamp("2022-09-11"))


def raw_tx(edges, day="2022-09-02", amt=100.0, cur="US Dollar", fmt="ACH", laundering=()):
    """Hand-built raw transactions (schema of load_transactions), one row per (sender, receiver), a minute apart."""
    return pd.DataFrame([{"ts": pd.Timestamp(day) + pd.Timedelta(minutes=i), "from_bank": "001", "from_acct": a,
                          "to_bank": "001", "to_acct": b, "amt_recv": amt, "recv_cur": cur, "amt_paid": amt,
                          "pay_cur": cur, "fmt": fmt, "is_laundering": int((a, b) in laundering)}
                         for i, (a, b) in enumerate(edges)])


def make_tx(edges, **kw):
    """raw_tx run through prepare()."""
    return prepare(raw_tx(edges, **kw), WINDOW)


@pytest.fixture
def tx_factory():
    return make_tx