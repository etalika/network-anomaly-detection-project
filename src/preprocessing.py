from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "synthetic_network_flows.csv"

# این ستون‌ها نباید به‌عنوان ویژگی وارد مدل شوند.
# Record_ID فقط شناسه است.
# Label پاسخ واقعی است.
# Anomaly_Type توضیح نوع ناهنجاری است.
EXCLUDED_COLUMNS = {
    "Record_ID",
    "Label",
    "Anomaly_Type",
}


def load_raw_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return columns used as model input."""
    missing_columns = {"Label", "Anomaly_Type"} - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {sorted(missing_columns)}"
        )

    feature_columns = [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    if not feature_columns:
        raise ValueError("No feature columns were found.")

    return feature_columns

def split_for_anomaly_detection(
    df: pd.DataFrame,
    test_size: float = 0.30,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Create training and testing data for unsupervised anomaly detection.

    The model is trained only on normal records.
    The test set contains normal and anomalous records.
    """
    feature_columns = get_feature_columns(df)

    # مقادیر واقعی ستون Label در CSV متنی هستند.
    normal_df = df[df["Label"] == "Normal"].copy()
    anomaly_df = df[df["Label"] == "Anomaly"].copy()

    if normal_df.empty:
        raise ValueError(
            "No normal records were found. "
            f"Available labels: {df['Label'].value_counts().to_dict()}"
        )

    if anomaly_df.empty:
        raise ValueError(
            "No anomalous records were found. "
            f"Available labels: {df['Label'].value_counts().to_dict()}"
        )

    train_normal_df, test_normal_df = train_test_split(
        normal_df,
        test_size=test_size,
        random_state=random_state,
    )

    test_df = pd.concat(
        [test_normal_df, anomaly_df],
        ignore_index=True,
    )

    test_df = test_df.sample(
        frac=1.0,
        random_state=random_state,
    ).reset_index(drop=True)

    X_train = train_normal_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()

    # تبدیل برچسب متنی به برچسب عددی برای ارزیابی مدل
    y_test = (
        test_df["Label"]
        .map({"Normal": 0, "Anomaly": 1})
        .astype(int)
    )

    return X_train, X_test, y_test


if __name__ == "__main__":
    dataset = load_raw_dataset()

    feature_columns = get_feature_columns(dataset)

    X_train, X_test, y_test = split_for_anomaly_detection(dataset)

    print("Feature columns:")
    print(feature_columns)

    print("\nTraining shape:")
    print(X_train.shape)

    print("\nTesting shape:")
    print(X_test.shape)

    print("\nTest label distribution:")
    print(y_test.value_counts().sort_index())
