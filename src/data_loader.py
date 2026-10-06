from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "synthetic_network_flows.csv"


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw network-flow dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def inspect_dataset(df: pd.DataFrame) -> None:
    """Print basic information about the dataset."""
    print("=" * 60)
    print("Dataset shape")
    print("=" * 60)
    print(df.shape)

    print("\n" + "=" * 60)
    print("Column names")
    print("=" * 60)
    print(df.columns.tolist())

    print("\n" + "=" * 60)
    print("First five rows")
    print("=" * 60)
    print(df.head())

    print("\n" + "=" * 60)
    print("Data types")
    print("=" * 60)
    print(df.dtypes)

    print("\n" + "=" * 60)
    print("Missing values")
    print("=" * 60)
    print(df.isna().sum())

    if "Label" in df.columns:
        print("\n" + "=" * 60)
        print("Label distribution")
        print("=" * 60)
        print(df["Label"].value_counts().sort_index())

    if "Anomaly_Type" in df.columns:
        print("\n" + "=" * 60)
        print("Anomaly type distribution")
        print("=" * 60)
        print(df["Anomaly_Type"].value_counts())


if __name__ == "__main__":
    dataset = load_dataset()
    inspect_dataset(dataset)
