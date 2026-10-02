"""Evaluation metrics."""
import numpy as np
import pandas as pd


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def nasa_score(y_true, y_pred) -> float:
    """Asymmetric score: late predictions (overestimating RUL) are penalized more."""
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def last_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """Last recorded cycle of each engine."""
    return df.groupby("unit").last().reset_index()
