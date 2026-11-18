import pandas as pd

from aml_triage.config import load_config


def test_configured_window_stops_before_the_tail():
    cfg = load_config("config/config.yaml")
    assert cfg.window_end <= pd.Timestamp("2022-09-11"), "window_end reaches the laundering-only tail (Trap 3)"


def test_forward_split_inside_window():
    cfg = load_config("config/config.yaml")
    assert cfg.window_start < cfg.forward_split < cfg.window_end