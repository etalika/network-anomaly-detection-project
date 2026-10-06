from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "synthetic_network_flows.csv"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "results"
    / "predictions"
    / "isolation_forest_predictions.csv"
)

TABLES_DIR = PROJECT_ROOT / "results" / "tables"


def build_test_metadata(
    df: pd.DataFrame,
    test_size: float = 0.30,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Recreate the same test-set order used during preprocessing.
    """
    normal_df = df[df["Label"] == "Normal"].copy()
    anomaly_df = df[df["Label"] == "Anomaly"].copy()

    _, test_normal_df = train_test_split(
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

    return test_df[
        [
            "Record_ID",
            "Label",
            "Anomaly_Type",
        ]
    ]


def main() -> None:
    dataset = pd.read_csv(DATA_PATH)
    predictions_df = pd.read_csv(PREDICTIONS_PATH)

    test_metadata = build_test_metadata(dataset)

    if len(test_metadata) != len(predictions_df):
        raise ValueError(
            "Test metadata and predictions have different lengths: "
            f"{len(test_metadata)} != {len(predictions_df)}"
        )

    analysis_df = pd.concat(
        [
            test_metadata.reset_index(drop=True),
            predictions_df.reset_index(drop=True),
        ],
        axis=1,
    )

    analysis_df["Correct"] = (
        (
            (analysis_df["Label"] == "Normal")
            & (analysis_df["Predicted_Label"] == 0)
        )
        |
        (
            (analysis_df["Label"] == "Anomaly")
            & (analysis_df["Predicted_Label"] == 1)
        )
    )

    anomaly_df = analysis_df[
        analysis_df["Label"] == "Anomaly"
    ].copy()

    type_results = (
        anomaly_df
        .groupby("Anomaly_Type")
        .agg(
            Total=("Correct", "size"),
            Correct=("Correct", "sum"),
        )
        .reset_index()
    )

    type_results["Missed"] = (
        type_results["Total"] - type_results["Correct"]
    )

    type_results["Recall"] = (
        type_results["Correct"] / type_results["Total"]
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = TABLES_DIR / "anomaly_type_analysis.csv"
    type_results.to_csv(output_path, index=False)

    print("Anomaly type analysis:")
    print(type_results.to_string(index=False))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
