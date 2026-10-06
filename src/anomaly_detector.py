from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest

from preprocessing import (
    load_raw_dataset,
    split_for_anomaly_detection,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"
PREDICTIONS_PATH = PREDICTIONS_DIR / "isolation_forest_predictions.csv"


def train_isolation_forest(
    X_train: pd.DataFrame,
    contamination: float = 0.10,
    random_state: int = 42,
) -> IsolationForest:
    """
    Train Isolation Forest on normal traffic only.
    """
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train)

    return model


def create_predictions(
    model: IsolationForest,
    X_test: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """
    Generate anomaly predictions and anomaly scores.

    Isolation Forest raw prediction:
        1  = normal
       -1  = anomaly
    """

    raw_predictions = model.predict(X_test)

    # تبدیل خروجی مدل به قالب پروژه:
    # 0 = Normal
    # 1 = Anomaly
    predictions = pd.Series(
        (raw_predictions == -1).astype(int),
        index=X_test.index,
        name="Predicted_Label",
    )

    # هرچه score پایین‌تر باشد، نمونه مشکوک‌تر است.
    anomaly_scores = pd.Series(
        model.score_samples(X_test),
        index=X_test.index,
        name="Anomaly_Score",
    )

    return predictions, anomaly_scores


def save_predictions(
    predictions: pd.Series,
    anomaly_scores: pd.Series,
) -> None:
    """Save model predictions and scores to the results directory."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.DataFrame(
        {
            "Predicted_Label": predictions,
            "Anomaly_Score": anomaly_scores,
        }
    )

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print(f"Predictions saved to: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    dataset = load_raw_dataset()

    X_train, X_test, y_test = split_for_anomaly_detection(
        dataset
    )

    model = train_isolation_forest(
        X_train=X_train,
        contamination=0.10,
        random_state=42,
    )

    predictions, anomaly_scores = create_predictions(
        model=model,
        X_test=X_test,
    )

    print("Model training completed.")
    print("\nPrediction distribution:")
    print(predictions.value_counts().sort_index())

    print("\nAnomaly score summary:")
    print(anomaly_scores.describe())

    save_predictions(
        predictions=predictions,
        anomaly_scores=anomaly_scores,
    )
