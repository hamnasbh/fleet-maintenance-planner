"""Saving, loading and using the trained model."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import MAX_RUL, ROOT
from .features import add_rolling

MODEL_PATH = ROOT / "models" / "model.joblib"


def save_bundle(bundle: dict, path: Path = MODEL_PATH) -> None:
    """Save the model with everything needed to reproduce its inputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path = MODEL_PATH) -> dict:
    return joblib.load(path)


def predict(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Predict the RUL at every cycle of raw engine data (same format as the C-MAPSS files)."""
    df = df.drop(columns=bundle["constant"], errors="ignore")
    df = add_rolling(df, bundle["sensors"], bundle["window"])
    df["predicted_RUL"] = np.clip(bundle["model"].predict(df[bundle["features"]]), 0, MAX_RUL)
    return df
