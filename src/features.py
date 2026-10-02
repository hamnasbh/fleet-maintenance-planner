"""Feature selection and engineering."""
import pandas as pd


def select_features(df: pd.DataFrame, sensors: list[str], threshold: float = 0.5) -> list[str]:
    """Keep sensors whose absolute correlation with RUL is above the threshold."""
    corr = df[sensors + ["RUL"]].corr()["RUL"]
    return [s for s in sensors if abs(corr[s]) > threshold]


def add_rolling(df: pd.DataFrame, cols: list[str], window: int) -> pd.DataFrame:
    """Add rolling mean and standard deviation of each column, per engine."""
    df = df.copy()
    g = df.groupby("unit")
    for c in cols:
        df[f"{c}_mean"] = g[c].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f"{c}_std"] = g[c].transform(lambda x: x.rolling(window, min_periods=1).std()).fillna(0)
    return df


def feature_names(cols: list[str]) -> list[str]:
    """Names of all model inputs: raw sensors, rolling means and rolling stds."""
    return cols + [f"{c}_mean" for c in cols] + [f"{c}_std" for c in cols]
