"""Generate traceable predictions aligned with the exact test split.

Reads raw records (Record_ID, Label, Anomaly_Type are taken directly from the
same records used in the test split - no guessing or fragile joins), trains
IsolationForest with the project settings, and saves a traceable output where
row order equals the real test-set order.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "synthetic_network_flows.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "predictions" / "traceable_predictions.csv"

EXCLUDED_COLUMNS = {"Record_ID", "Label", "Anomaly_Type"}

TEST_SIZE = 0.30
RANDOM_STATE = 42


def load_raw_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def build_exact_test_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the exact preprocessing split (same code path/order logic)."""
    feature_columns = [
        c for c in df.columns if c not in EXCLUDED_COLUMNS
    ]

    normal_df = df[df["Label"] == "Normal"].copy()
    anomaly_df = df[df["Label"] == "Anomaly"].copy()

    if normal_df.empty or anomaly_df.empty:
        raise ValueError("Both Normal and Anomaly records are required.")

    _, test_normal_df = train_test_split(
        normal_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    test_df = pd.concat([test_normal_df, anomaly_df], ignore_index=True)
    test_df = test_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return test_df, feature_columns


def main() -> None:
    df = load_raw_dataset()
    print(f"Input records: {len(df)}")

    test_df, feature_columns = build_exact_test_frame(df)

    train_normal_df, _ = train_test_split(
        df[df["Label"] == "Normal"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    X_train = train_normal_df.drop(columns=EXCLUDED_COLUMNS)
    X_test = test_df[feature_columns]

    model = IsolationForest(
        n_estimators=200,
        contamination=0.10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)

    raw_predictions = model.predict(X_test)
    scores = model.score_samples(X_test)

    out = pd.DataFrame(
        {
            "Record_ID": test_df["Record_ID"].values,
            "Label": test_df["Label"].values,
            "Anomaly_Type": test_df["Anomaly_Type"].values,
            "Predicted_Label": (raw_predictions == -1).astype(int),
            "Anomaly_Score": scores,
        }
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Traceable predictions saved to: {OUTPUT_PATH}")
    print(out["Predicted_Label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
