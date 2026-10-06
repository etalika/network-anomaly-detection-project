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

FALSE_POSITIVES_PATH = (
    TABLES_DIR / "false_positive_records.csv"
)

SUMMARY_PATH = (
    TABLES_DIR / "false_positive_summary.csv"
)

FEATURE_COMPARISON_PATH = (
    TABLES_DIR / "false_positive_feature_comparison.csv"
)


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataframe_name: str,
) -> None:
    """Validate that required columns exist."""
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"{dataframe_name} is missing columns: "
            f"{sorted(missing_columns)}\n"
            f"Available columns: {list(dataframe.columns)}"
        )


def load_dataset() -> pd.DataFrame:
    """Load and validate the raw dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset file not found:\n{DATA_PATH}"
        )

    dataset = pd.read_csv(DATA_PATH)

    require_columns(
        dataset,
        {
            "Record_ID",
            "Label",
            "Anomaly_Type",
        },
        "Dataset",
    )

    return dataset


def load_predictions() -> pd.DataFrame:
    """Load and validate saved predictions."""
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predictions file not found:\n{PREDICTIONS_PATH}"
        )

    predictions = pd.read_csv(PREDICTIONS_PATH)

    require_columns(
        predictions,
        {
            "Predicted_Label",
            "Anomaly_Score",
        },
        "Predictions",
    )

    return predictions


def recreate_test_metadata(
    dataset: pd.DataFrame,
    test_size: float = 0.30,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Recreate the test-set order used by the existing pipeline
    while preserving all original dataset features.
    """
    dataset = dataset.copy()

    dataset["Label"] = (
        dataset["Label"]
        .astype(str)
        .str.strip()
    )

    normal_df = dataset[
        dataset["Label"].str.lower() == "normal"
    ].copy()

    anomaly_df = dataset[
        dataset["Label"].str.lower() == "anomaly"
    ].copy()

    if normal_df.empty:
        raise ValueError("No Normal records were found.")

    if anomaly_df.empty:
        raise ValueError("No Anomaly records were found.")

    _, test_normal_df = train_test_split(
        normal_df,
        test_size=test_size,
        random_state=random_state,
    )

    test_df = pd.concat(
        [
            test_normal_df,
            anomaly_df,
        ],
        ignore_index=True,
    )

    test_df = test_df.sample(
        frac=1.0,
        random_state=random_state,
    ).reset_index(drop=True)

    return test_df


def build_analysis_dataframe(
    test_metadata: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine test-set records with their corresponding predictions.
    """
    if len(test_metadata) != len(predictions):
        raise ValueError(
            "Test metadata and predictions have different lengths: "
            f"{len(test_metadata)} != {len(predictions)}"
        )

    analysis_df = pd.concat(
        [
            test_metadata.reset_index(drop=True),
            predictions.reset_index(drop=True),
        ],
        axis=1,
    )

    analysis_df["Actual_Label"] = (
        analysis_df["Label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "normal": 0,
                "anomaly": 1,
            }
        )
    )

    if analysis_df["Actual_Label"].isna().any():
        unknown_labels = (
            analysis_df.loc[
                analysis_df["Actual_Label"].isna(),
                "Label",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Unknown label values found: {unknown_labels}"
        )

    analysis_df["Predicted_Label"] = (
        pd.to_numeric(
            analysis_df["Predicted_Label"],
            errors="raise",
        )
        .astype(int)
    )

    valid_predictions = {0, 1}

    invalid_predictions = set(
        analysis_df["Predicted_Label"].unique()
    ) - valid_predictions

    if invalid_predictions:
        raise ValueError(
            "Predicted_Label must contain only 0 or 1. "
            f"Invalid values: {sorted(invalid_predictions)}"
        )

    analysis_df["Prediction_Status"] = "Unknown"

    analysis_df.loc[
        (
            (analysis_df["Actual_Label"] == 1)
            & (analysis_df["Predicted_Label"] == 1)
        ),
        "Prediction_Status",
    ] = "True Positive"

    analysis_df.loc[
        (
            (analysis_df["Actual_Label"] == 0)
            & (analysis_df["Predicted_Label"] == 0)
        ),
        "Prediction_Status",
    ] = "True Negative"

    analysis_df.loc[
        (
            (analysis_df["Actual_Label"] == 0)
            & (analysis_df["Predicted_Label"] == 1)
        ),
        "Prediction_Status",
    ] = "False Positive"

    analysis_df.loc[
        (
            (analysis_df["Actual_Label"] == 1)
            & (analysis_df["Predicted_Label"] == 0)
        ),
        "Prediction_Status",
    ] = "False Negative"

    if (analysis_df["Prediction_Status"] == "Unknown").any():
        raise ValueError(
            "Some records could not be assigned a prediction status."
        )

    return analysis_df


def create_false_positive_summary(
    analysis_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create a summary of false positives."""
    total_records = len(analysis_df)

    normal_count = (
        analysis_df["Actual_Label"] == 0
    ).sum()

    false_positive_count = (
        analysis_df["Prediction_Status"]
        == "False Positive"
    ).sum()

    false_positive_rate = (
        false_positive_count / normal_count
        if normal_count > 0
        else 0.0
    )

    return pd.DataFrame(
        {
            "Metric": [
                "Total test records",
                "Actual normal records",
                "False positive records",
                "False positive rate among normal records",
            ],
            "Value": [
                total_records,
                normal_count,
                false_positive_count,
                false_positive_rate,
            ],
        }
    )


def create_feature_comparison(
    analysis_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare numeric features between false positives
    and correctly classified normal records.
    """
    normal_df = analysis_df[
        analysis_df["Actual_Label"] == 0
    ].copy()

    false_positive_df = normal_df[
        normal_df["Prediction_Status"] == "False Positive"
    ].copy()

    true_negative_df = normal_df[
        normal_df["Prediction_Status"] == "True Negative"
    ].copy()

    excluded_columns = {
        "Record_ID",
        "Label",
        "Anomaly_Type",
        "Predicted_Label",
        "Actual_Label",
        "Anomaly_Score",
    }

    numeric_columns = normal_df.select_dtypes(
        include="number"
    ).columns

    feature_columns = [
        column
        for column in numeric_columns
        if column not in excluded_columns
    ]

    rows = []

    for feature in feature_columns:
        false_positive_mean = (
            false_positive_df[feature].mean()
            if not false_positive_df.empty
            else float("nan")
        )

        true_negative_mean = (
            true_negative_df[feature].mean()
            if not true_negative_df.empty
            else float("nan")
        )

        rows.append(
            {
                "Feature": feature,
                "False_Positive_Mean": false_positive_mean,
                "True_Negative_Mean": true_negative_mean,
                "Mean_Difference": (
                    false_positive_mean
                    - true_negative_mean
                ),
            }
        )

    comparison_df = pd.DataFrame(rows)

    if not comparison_df.empty:
        comparison_df["Absolute_Mean_Difference"] = (
            comparison_df["Mean_Difference"].abs()
        )

        comparison_df = comparison_df.sort_values(
            by="Absolute_Mean_Difference",
            ascending=False,
        ).reset_index(drop=True)

    return comparison_df


def save_outputs(
    analysis_df: pd.DataFrame,
) -> None:
    """Save false-positive records and analysis tables."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    false_positive_df = analysis_df[
        analysis_df["Prediction_Status"] == "False Positive"
    ].copy()

    summary_df = create_false_positive_summary(
        analysis_df
    )

    comparison_df = create_feature_comparison(
        analysis_df
    )

    false_positive_df.to_csv(
        FALSE_POSITIVES_PATH,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    comparison_df.to_csv(
        FEATURE_COMPARISON_PATH,
        index=False,
    )

    print("\nFalse-positive records saved to:")
    print(FALSE_POSITIVES_PATH)

    print("\nFalse-positive summary saved to:")
    print(SUMMARY_PATH)

    print("\nFeature comparison saved to:")
    print(FEATURE_COMPARISON_PATH)


def main() -> None:
    dataset = load_dataset()
    predictions = load_predictions()

    test_metadata = recreate_test_metadata(
        dataset=dataset,
        test_size=0.30,
        random_state=42,
    )

    analysis_df = build_analysis_dataframe(
        test_metadata=test_metadata,
        predictions=predictions,
    )

    true_positive_count = (
        analysis_df["Prediction_Status"]
        == "True Positive"
    ).sum()

    true_negative_count = (
        analysis_df["Prediction_Status"]
        == "True Negative"
    ).sum()

    false_positive_count = (
        analysis_df["Prediction_Status"]
        == "False Positive"
    ).sum()

    false_negative_count = (
        analysis_df["Prediction_Status"]
        == "False Negative"
    ).sum()

    print("Prediction status counts:")
    print(f"True Positive: {true_positive_count}")
    print(f"True Negative: {true_negative_count}")
    print(f"False Positive: {false_positive_count}")
    print(f"False Negative: {false_negative_count}")

    summary_df = create_false_positive_summary(
        analysis_df
    )

    print("\nFalse-positive summary:")
    print(summary_df.to_string(index=False))

    comparison_df = create_feature_comparison(
        analysis_df
    )

    if comparison_df.empty:
        print(
            "\nNo numeric feature columns were found "
            "for feature comparison."
        )
    else:
        print("\nTop features associated with false positives:")
        print(
            comparison_df.head(10).to_string(
                index=False
            )
        )

    save_outputs(analysis_df)


if __name__ == "__main__":
    main()
