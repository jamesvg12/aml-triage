import pandas as pd
import pytest

from aml_triage.load import FX, prepare
from conftest import WINDOW, make_tx, raw_tx

def test_fx_table_has_the_15_dataset_currencies():
    assert len(FX) == 15 and FX["US Dollar"] == 1.0


def test_amounts_converted_to_usd():
    tx = pd.concat([make_tx([("A", "B")], cur="Euro"), make_tx([("A", "B")], cur="UK Pound")])
    assert sorted(tx.amt_usd) == pytest.approx([100.0, 115.0])



def test_unmapped_currency_fails_loudly():
    with pytest.raises(ValueError, match="unmapped currency"):
        make_tx([("A", "B")], cur="Klingon Darsek")


def test_window_end_is_exclusive():
    raw = pd.concat([raw_tx([("A", "B")], day="2022-09-10"), raw_tx([("A", "B")], day="2022-09-11")], ignore_index=True)
    out = prepare(raw, WINDOW)
    assert len(out) == 1 and out.ts.max() < WINDOW.window_end


def test_self_transfer_flag(tx_factory):
    tx = tx_factory([("A", "A"), ("A", "B")])
    assert tx.is_self.tolist() == [True, False]