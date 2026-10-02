"""Train the final model on FD001, evaluate it on the test set and save it.

Run from the project root:  python -m src.train
"""
from sklearn.ensemble import HistGradientBoostingRegressor

from .data import MAX_RUL, add_rul, find_constant, load_split, load_test_rul, sensor_columns
from .evaluation import last_cycles, nasa_score, rmse
from .features import add_rolling, feature_names, select_features
from .model import MODEL_PATH, predict, save_bundle

WINDOW = 80


def main() -> None:
    train = add_rul(load_split("train"))
    constant = find_constant(train)
    train = train.drop(columns=constant)

    sensors = select_features(train, sensor_columns(train))
    features = feature_names(sensors)

    train_r = add_rolling(train, sensors, WINDOW)
    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(train_r[features], train_r["RUL"])

    bundle = {
        "model": model,
        "constant": constant,
        "sensors": sensors,
        "features": features,
        "window": WINDOW,
    }

    test_pred = last_cycles(predict(bundle, load_split("test")))
    y_true = load_test_rul().clip(upper=MAX_RUL).values
    y_pred = test_pred["predicted_RUL"].values
    print(f"Test RMSE: {rmse(y_true, y_pred):.1f} cycles | NASA score: {nasa_score(y_true, y_pred):.0f}")

    save_bundle(bundle)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
