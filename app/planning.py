"""Turning RUL predictions into maintenance decisions."""
import pandas as pd

CRITICAL = "🔴 Critical"
WARNING = "🟡 Plan soon"
OK = "🟢 OK"


def smooth_predictions(pred: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Average each engine's predictions over its last `window` cycles to reduce noise."""
    pred = pred.copy()
    pred["smoothed_RUL"] = pred.groupby("unit")["predicted_RUL"].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    return pred


def status(rul: float, critical: int, warning: int) -> str:
    if rul <= critical:
        return CRITICAL
    if rul <= warning:
        return WARNING
    return OK


def fleet_status(pred: pd.DataFrame, margin: int, critical: int, warning: int) -> pd.DataFrame:
    """One row per engine: its latest smoothed prediction, the planned RUL after the safety margin, and a status."""
    last = pred.groupby("unit").last().reset_index()
    fleet = pd.DataFrame({
        "Engine": last["unit"],
        "Cycles flown": last["cycle"],
        "Predicted RUL": last["smoothed_RUL"].round().astype(int),
    })
    fleet["Planned RUL"] = (fleet["Predicted RUL"] - margin).clip(lower=0)
    fleet["Status"] = fleet["Planned RUL"].apply(status, args=(critical, warning))
    return fleet.sort_values("Planned RUL").reset_index(drop=True)
