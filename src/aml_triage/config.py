"""Load config/config.yaml, with .env / environment overrides."""
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    seed: int
    kaggle_handle: str
    files: dict
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    train_end: pd.Timestamp
    forward_split: pd.Timestamp
    analyst_capacity: int
    run_louvain: bool
    data_dir: Path

    @property
    def raw(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed(self) -> Path:
        return self.data_dir / "processed"


def load_config(path: str | Path = "config/config.yaml") -> Config:
    """Precedence: environment (incl. .env) > YAML. Relative data_dir resolves against the repo root, not the cwd."""
    path = Path(path).resolve()
    repo_root = path.parent.parent
    load_dotenv(repo_root / ".env")
    c = yaml.safe_load(path.read_text())

    data_dir = Path(os.environ.get("DATASETS_TASKS_DATA", c["data_dir"])).expanduser()
    if not data_dir.is_absolute():
        data_dir = repo_root / data_dir

    cfg = Config(
        seed=int(c["seed"]),
        kaggle_handle=c["kaggle_handle"],
        files=c["files"],
        window_start=pd.Timestamp(c["window_start"]),
        window_end=pd.Timestamp(c["window_end"]),
        train_end=pd.Timestamp(c["train_end"]),
        forward_split=pd.Timestamp(c["forward_split"]),
        analyst_capacity=int(c["analyst_capacity"]),
        run_louvain=bool(c["run_louvain"]),
        data_dir=data_dir,
    )
    cfg.raw.mkdir(parents=True, exist_ok=True)        # as in the notebook's Section 0
    cfg.processed.mkdir(parents=True, exist_ok=True)
    return cfg