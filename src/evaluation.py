from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from preprocessing import (
    load_raw_dataset,
    split_for_anomaly_detection,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "results"
    / "predictions"
    / "isolation_forest_predictions.csv"
)
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


def load_predictions(path: Path = PREDICTIONS_PATH) -> pd.DataFrame:
    """Load saved model predictions."""
    if not path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {path}"
        )

    predictions_df = pd.read_csv(path)

    required_columns = {
        "Predicted_Label",
        "Anomaly_Score",
    }

    missing_columns = required_columns - set(predictions_df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing prediction columns: {sorted(missing_columns)}"
        )

    return predictions_df


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float]:
    """Calculate classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }


def save_metrics(metrics: dict[str, float]) -> None:
    """Save evaluation metrics as a CSV table."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(
        {
            "Metric": list(metrics.keys()),
            "Value": list(metrics.values()),
        }
    )

    output_path = TABLES_DIR / "evaluation_metrics.csv"
    metrics_df.to_csv(output_path, index=False)

    print(f"Metrics saved to: {output_path}")


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> None:
    """Save the confusion matrix as a CSV table."""
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual_Normal", "Actual_Anomaly"],
        columns=["Predicted_Normal", "Predicted_Anomaly"],
    )

    output_path = TABLES_DIR / "confusion_matrix.csv"
    matrix_df.to_csv(output_path)

    print(f"Confusion matrix saved to: {output_path}")


if __name__ == "__main__":
    dataset = load_raw_dataset()

    _, _, y_test = split_for_anomaly_detection(dataset)

    predictions_df = load_predictions()
    y_pred = predictions_df["Predicted_Label"].astype(int)

    if len(y_test) != len(y_pred):
        raise ValueError(
            "The number of true labels and predictions does not match: "
            f"{len(y_test)} != {len(y_pred)}"
        )

    metrics = evaluate_predictions(
        y_true=y_test,
        y_pred=y_pred,
    )

    print("Evaluation results:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Normal", "Anomaly"],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1],
        )
    )

    save_metrics(metrics)
    save_confusion_matrix(y_test, y_pred)
