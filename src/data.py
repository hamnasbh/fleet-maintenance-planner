"""Loading and preparing the NASA C-MAPSS data."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

COLUMNS = ["unit", "cycle", "set1", "set2", "set3"] + [f"s{i}" for i in range(1, 22)]
MAX_RUL = 125


def load_split(split: str, subset: str = "FD001") -> pd.DataFrame:
    """Load the 'train' or 'test' file of a subset."""
    path = DATA_DIR / f"{split}_{subset}.txt"
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def load_test_rul(subset: str = "FD001") -> pd.Series:
    """True RUL of each test engine at its last recorded cycle (row i = engine i + 1)."""
    path = DATA_DIR / f"RUL_{subset}.txt"
    return pd.read_csv(path, header=None, names=["RUL"])["RUL"]


def add_rul(df: pd.DataFrame, max_rul: int = MAX_RUL) -> pd.DataFrame:
    """Add the RUL target to run-to-failure data, capped at max_rul."""
    df = df.copy()
    df["RUL"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=max_rul)
    return df


def find_constant(df: pd.DataFrame, tol: float = 1e-4) -> list[str]:
    """Columns that never vary."""
    return [c for c in df.columns if df[c].std() < tol]


def sensor_columns(df: pd.DataFrame) -> list[str]:
    """Sensor columns (s1, s2, ...), excluding operational settings."""
    return [c for c in df.columns if c[0] == "s" and c[1:].isdigit()]
