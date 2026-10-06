"""
visualization.py
================

Publication-ready figure generation for the network anomaly detection project.

Run from the project root:

    python src/visualization.py

Inputs (all resolved relative to the project root, i.e. the parent of src/):
    data/raw/synthetic_network_flows.csv
    results/predictions/traceable_predictions.csv   (fallback: isolation_forest_predictions.csv)
    results/tables/confusion_matrix.csv
    results/tables/traceable_evaluation_metrics.csv (fallback: evaluation_metrics.csv)
    results/tables/false_positive_feature_comparison.csv
    results/tables/anomaly_type_analysis.csv

Outputs (PNG files written to results/figures/):
    01_label_distribution.png
    02_anomaly_type_distribution.png
    03_confusion_matrix.png
    04_metrics_bar_chart.png
    05_anomaly_score_distribution.png
    06_false_positive_feature_comparison.png
    07_recall_by_anomaly_type.png

Each figure is generated independently: if its inputs are unavailable the
script prints a warning and continues rather than crashing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: no interactive display required

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths and style
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """Return the project root as the parent directory of src/."""
    script_dir = Path(__file__).resolve().parent
    if script_dir.name == "src":
        root = script_dir.parent
    else:  # e.g. executed from an unusual location; fall back to cwd
        root = Path.cwd()
        if (root / "src").exists():
            root = root
        elif (script_dir.parent / "data").exists():
            root = script_dir.parent
    return root


ROOT: Path = get_project_root()
RAW_DIR: Path = ROOT / "data" / "raw"
PRED_DIR: Path = ROOT / "results" / "predictions"
TABLE_DIR: Path = ROOT / "results" / "tables"
FIG_DIR: Path = ROOT / "results" / "figures"

RAW_DATA: Path = RAW_DIR / "synthetic_network_flows.csv"
TRACEABLE_PRED: Path = PRED_DIR / "traceable_predictions.csv"
BASE_PRED: Path = PRED_DIR / "isolation_forest_predictions.csv"
CONFUSION_TABLE: Path = TABLE_DIR / "confusion_matrix.csv"
TRACEABLE_METRICS: Path = TABLE_DIR / "traceable_evaluation_metrics.csv"
BASE_METRICS: Path = TABLE_DIR / "evaluation_metrics.csv"
FP_FEATURE_TABLE: Path = TABLE_DIR / "false_positive_feature_comparison.csv"
ANOMALY_TYPE_TABLE: Path = TABLE_DIR / "anomaly_type_analysis.csv"

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_saved(path: Path) -> None:
    print(f"  Saved: {path}")


def load_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    """Read a CSV file, returning None (with a warning) if unavailable/invalid."""
    if not path.is_file():
        print(f"  [WARNING] Input not found, skipping dependent figure: {path}")
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            print(f"  [WARNING] Input is empty, skipping dependent figure: {path}")
            return None
        return df
    except Exception as exc:  # noqa: BLE001 - defensive: any read failure
        print(f"  [WARNING] Could not read {path} ({exc}); figure skipped.")
        return None


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Safely locate a column by trying candidate (case-insensitive) names."""
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        hit = lowered.get(cand.lower())
        if hit is not None:
            return hit
    return None


def normalize_binary(series: pd.Series) -> pd.Series:
    """Map Normal/Anomaly (any case) or 0/1 to the strings 'Normal'/'Anomaly'."""
    mapping = {"normal": "Normal", "anomaly": "Anomaly",
               "0": "Normal", "1": "Anomaly", 0: "Normal", 1: "Anomaly"}
    return series.map(lambda v: mapping.get(v, mapping.get(str(v).strip().lower(), v)))


def load_raw_data() -> Optional[pd.DataFrame]:
    return load_csv_safe(RAW_DATA)


def load_predictions() -> Optional[pd.DataFrame]:
    """Prefer traceable_predictions.csv; fall back to isolation_forest_predictions.csv."""
    for path in (TRACEABLE_PRED, BASE_PRED):
        df = load_csv_safe(path)
        if df is not None:
            print(f"  Using predictions file: {path.name}")
            return df
    return None


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_label_distribution(raw: Optional[pd.DataFrame]) -> None:
    """Figure 1: distribution of Label in the raw dataset."""
    print("[1/7] Label distribution")
    if raw is None:
        print("  [WARNING] Raw data unavailable; skipped.")
        return
    col = find_column(raw, ["Label"])
    if col is None:
        print("  [WARNING] No 'Label' column in raw data; skipped.")
        return
    counts = normalize_binary(raw[col]).value_counts()

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.index.astype(str), counts.values,
                  color=["#4C9F70", "#C0504D"][: len(counts)], edgecolor="black")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:,}",
                ha="center", va="bottom", fontsize=12)
    total = counts.sum()
    ax.set_title("Label Distribution in Raw Dataset")
    ax.set_ylabel("Number of Flows")
    ax.set_xlabel("Label")
    sec = ax.secondary_yaxis("right")
    sec.set_ylabel("Share (%)")
    sec.set_ticks([])
    for tick, val in zip(ax.get_xticklabels(), counts.values):
        tick.set_text(tick.get_text())
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val / 2, f"{val / total:.1%}",
                ha="center", va="center", color="white", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "01_label_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    _print_saved(out)


def plot_anomaly_type_distribution(raw: Optional[pd.DataFrame]) -> None:
    """Figure 2: distribution of Anomaly_Type in the raw dataset."""
    print("[2/7] Anomaly type distribution")
    if raw is None:
        print("  [WARNING] Raw data unavailable; skipped.")
        return
    col = find_column(raw, ["Anomaly_Type", "AnomalyType", "Attack_Type"])
    if col is None:
        print("  [WARNING] No 'Anomaly_Type' column in raw data; skipped.")
        return
    counts = raw[col].astype(str).value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=counts.values, y=counts.index, hue=counts.index, palette="viridis", legend=False, ax=ax)
    for i, val in enumerate(counts.values):
        ax.text(val, i, f" {val:,} ({val / counts.sum():.1%})", va="center", fontsize=11)
    ax.set_title("Anomaly Type Distribution in Raw Dataset")
    ax.set_xlabel("Number of Flows")
    ax.set_ylabel("Anomaly Type")
    fig.tight_layout()
    out = FIG_DIR / "02_anomaly_type_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    _print_saved(out)


def plot_confusion_matrix(preds: Optional[pd.DataFrame]) -> None:
    """Figure 3: confusion matrix from traceable predictions, else the table CSV."""
    print("[3/7] Confusion matrix")

    actual_col = pred_col = None
    if preds is not None:
        actual_col = find_column(preds, ["Label", "Actual_Label", "True_Label", "Actual"])
        pred_col = find_column(preds, ["Predicted_Label", "Prediction", "Predicted"])
    if preds is not None and actual_col and pred_col:
        y_true = normalize_binary(preds[actual_col])
        y_pred = normalize_binary(preds[pred_col])
        labels = ["Normal", "Anomaly"]
        cm = pd.crosstab(y_true, y_pred).reindex(index=labels, columns=labels, fill_value=0)
        title = "Confusion Matrix (Traceable Predictions)"
    else:
        table = load_csv_safe(CONFUSION_TABLE)
        if table is None:
            return
        try:
            index_name = table.columns[0]
            table = table.set_index(index_name)
            cm = table.astype(float)
            title = "Confusion Matrix (results/tables/confusion_matrix.csv)"
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARNING] Could not parse confusion matrix table ({exc}); skipped.")
            return

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", cbar=True,
                linewidths=1, linecolor="white", ax=ax,
                annot_kws={"fontsize": 14})
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels([str(t).replace("_", " ") for t in cm.columns], rotation=0)
    ax.set_yticklabels([str(t).replace("_", " ") for t in cm.index], rotation=0)
    fig.tight_layout()
    out = FIG_DIR / "03_confusion_matrix.png"
    fig.savefig(out)
    plt.close(fig)
    _print_saved(out)


def plot_metrics() -> None:
    """Figure 4: metrics bar chart (traceable metrics preferred, else base)."""
    print("[4/7] Evaluation metrics bar chart")
    for path in (TRACEABLE_METRICS, BASE_METRICS):
        table = load_csv_safe(path)
        if table is not None:
            print(f"  Using metrics file: {path.name}")
            break
    else:
        print("  [WARNING] No metrics table available; skipped.")
        return

    try:
        if {"Metric", "Value"}.issubset(table.columns):
            metrics = dict(zip(table["Metric"].astype(str), pd.to_numeric(table["Value"], errors="coerce")))
        else:  # key-value style: metric names as columns
            row = table.iloc[0]
            metrics = {str(c): pd.to_numeric(row[c], errors="coerce") for c in table.columns}
        metrics = {k: v for k, v in metrics.items() if pd.notna(v)}
        if not metrics:
            raise ValueError("no numeric metric values found")
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARNING] Could not parse metrics ({exc}); skipped.")
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(metrics.keys(), metrics.values(), color=sns.color_palette("mako", len(metrics)),
                  edgecolor="black")
    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}",
                ha="center", va="bottom", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Evaluation Metrics")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    out = FIG_DIR / "04_metrics_bar_chart.png"
    fig.savefig(out)
    plt.close(fig)
    _print_saved(out)


def plot_score_distribution(preds: Optional[pd.DataFrame]) -> None:
    """Figure 5: anomaly score distributions by actual label."""
    print("[5/7] Anomaly score distribution by actual label")
    if preds is None:
        print("  [WARNING] Traceable predictions unavailable; skipped.")
        return
    score_col = find_column(preds, ["Anomaly_Score", "Score", "AnomalyScore"])
    actual_col = find_column(preds, ["Label", "Actual_Label", "True_Label", "Actual"])
    if score_col is None or actual_col is None:
        print("  [WARNING] Required columns (Anomaly_Score / Label) missing; skipped.")
        return
    data = preds[[score_col, actual_col]].copy()
    data[score_col] = pd.to_numeric(data[score_col], errors="coerce")
    data = data.dropna()
    if data.empty:
        print("  [WARNING] No numeric anomaly scores available; skipped.")
        return
    data["_label"] = normalize_binary(data[actual_col])

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.kdeplot(data=data, x=score_col, hue="_label", fill=True, common_norm=False,
                palette={"Normal": "#4C9F70", "Anomaly": "#C0504D"}, alpha=0.45, ax=ax)
    ax.set_title("Anomaly Score Distribution by Actual Label")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    fig.tight_layout()
    out = FIG_DIR / "05_anomaly_score_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    _print_saved(out)


def plot_false_positive_features(preds: Optional[pd.DataFrame], raw: Optional[pd.DataFrame]) -> None:
    """Figure 6: FP vs TN feature comparison (computed or from table)."""
    print("[6/7] False positive vs true negative feature comparison")
    plotted = False

    if preds is not None and raw is not None:
        actual_col = find_column(preds, ["Label", "Actual_Label", "True_Label", "Actual"])
        pred_col = find_column(preds, ["Predicted_Label", "Prediction", "Predicted"])
        id_col = find_column(preds, ["Record_ID", "ID"])
        raw_id = find_column(raw, ["Record_ID", "ID"])
        if all([actual_col, pred_col, id_col, raw_id]):
            merged = preds.merge(raw, left_on=id_col, right_on=raw_id,
                                 suffixes=("_pred", "_raw"), how="inner")
            if actual_col not in merged.columns:  # suffix applied due to collision
                actual_col = find_column(merged, ["Label_pred", "Actual_Label_pred", "Label_raw"])
            if pred_col not in merged.columns:
                pred_col = find_column(merged, ["Predicted_Label_pred", "Predicted_Label"])
            if actual_col is None or pred_col is None or actual_col not in merged.columns or pred_col not in merged.columns:
                print("  [WARNING] Label columns lost in merge; falling back to table.")
            else:
                actual = normalize_binary(merged[actual_col])
                predicted = normalize_binary(merged[pred_col])
            fp_mask = (actual == "Normal") & (predicted == "Anomaly")
            tn_mask = (actual == "Normal") & (predicted == "Normal")
            if fp_mask.sum() > 0 and tn_mask.sum() > 0:
                numeric = merged.select_dtypes(include="number")
                drop_cols = [c for c in numeric.columns
                             if c in {find_column(merged, ["Record_ID", "ID"]), id_col, raw_id}]
                numeric = numeric.drop(columns=drop_cols, errors="ignore")
                if not numeric.empty:
                    fp_mean = numeric[fp_mask.reindex(numeric.index, fill_value=False)].mean()
                    tn_mean = numeric[tn_mask.reindex(numeric.index, fill_value=False)].mean()
                    # normalize each feature to 0-1 by its combined max for comparability
                    denom = (pd.concat([fp_mean, tn_mean], axis=1)
                             .abs().max(axis=1).replace(0, 1))
                    comp = pd.DataFrame({
                        "False Positives": fp_mean / denom,
                        "True Negatives": tn_mean / denom,
                    }).sort_values("False Positives", ascending=False).head(15)

                    fig, ax = plt.subplots(figsize=(10, max(6, 0.45 * len(comp))))
                    comp.plot(kind="barh", ax=ax,
                              color=["#C0504D", "#4C9F70"], edgecolor="black")
                    ax.invert_yaxis()
                    ax.set_title("Feature Means: False Positives vs True Negatives (normalized)")
                    ax.set_xlabel("Normalized Mean Value")
                    ax.set_ylabel("Feature")
                    ax.legend(loc="lower right")
                    fig.tight_layout()
                    out = FIG_DIR / "06_false_positive_feature_comparison.png"
                    fig.savefig(out)
                    plt.close(fig)
                    _print_saved(out)
                    plotted = True
            else:
                print("  [WARNING] No FP or TN records present in predictions; falling back to table.")
        else:
            print("  [WARNING] Required join columns missing; falling back to table.")
    else:
        print("  [WARNING] Predictions/raw data unavailable; falling back to table.")

    if not plotted:
        table = load_csv_safe(FP_FEATURE_TABLE)
        if table is None:
            print("  [WARNING] Fallback table unavailable; skipped.")
            return
        feat_col = find_column(table, ["Feature"])
        fp_col = find_column(table, ["False_Positive_Mean", "False Positives"])
        tn_col = find_column(table, ["True_Negative_Mean", "True Negatives"])
        if not all([feat_col, fp_col, tn_col]):
            print("  [WARNING] Fallback table missing required columns; skipped.")
            return
        comp = table.set_index(feat_col)[[fp_col, tn_col]].head(15)
        comp.columns = ["False Positives", "True Negatives"]

        fig, ax = plt.subplots(figsize=(10, max(6, 0.45 * len(comp))))
        comp.plot(kind="barh", ax=ax, color=["#C0504D", "#4C9F70"], edgecolor="black")
        ax.invert_yaxis()
        ax.set_title("Feature Means: False Positives vs True Negatives (from table)")
        ax.set_xlabel("Mean Value")
        ax.set_ylabel("Feature")
        ax.legend(loc="lower right")
        fig.tight_layout()
        out = FIG_DIR / "06_false_positive_feature_comparison.png"
        fig.savefig(out)
        plt.close(fig)
        _print_saved(out)


def plot_recall_by_anomaly_type(preds: Optional[pd.DataFrame], raw: Optional[pd.DataFrame]) -> None:
    """Figure 7: recall per anomaly type from predictions, else the analysis table."""
    print("[7/7] Recall by anomaly type")
    recall: Optional[pd.Series] = None
    title = "Recall by Anomaly Type"

    if preds is not None:
        actual_col = find_column(preds, ["Label", "Actual_Label", "True_Label", "Actual"])
        pred_col = find_column(preds, ["Predicted_Label", "Prediction", "Predicted"])
        type_col = find_column(preds, ["Anomaly_Type", "AnomalyType", "Attack_Type"])
        if not all([actual_col, pred_col, type_col]) and raw is not None:
            id_col = find_column(preds, ["Record_ID", "ID"])
            raw_id = find_column(raw, ["Record_ID", "ID"])
            raw_type = find_column(raw, ["Anomaly_Type", "AnomalyType", "Attack_Type"])
            if all([id_col, raw_id, raw_type]):
                preds = preds.merge(raw[[raw_id, raw_type]], left_on=id_col, right_on=raw_id,
                                    how="left")
                type_col = raw_type
        if all([actual_col, pred_col, type_col]):
            actual = normalize_binary(preds[actual_col])
            predicted = normalize_binary(preds[pred_col])
            sub = pd.DataFrame({"type": preds[type_col].astype(str),
                                "actual": actual, "pred": predicted})
            sub = sub[sub["actual"] == "Anomaly"]
            if not sub.empty:
                recall = sub.groupby("type")["pred"].apply(lambda s: (s == "Anomaly").mean())

    if recall is None or recall.empty:
        table = load_csv_safe(ANOMALY_TYPE_TABLE)
        if table is None:
            print("  [WARNING] No recall source available; skipped.")
            return
        type_col = find_column(table, ["Anomaly_Type", "Type"])
        rec_col = find_column(table, ["Recall"])
        if type_col is None or rec_col is None:
            print("  [WARNING] Anomaly type table missing required columns; skipped.")
            return
        recall = table.set_index(type_col)[rec_col]
        title += " (from anomaly_type_analysis.csv)"

    recall = (recall.astype(float).sort_values(ascending=False) * 100).clip(upper=100)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(recall.index.astype(str), recall.values,
                  color=sns.color_palette("rocket", len(recall)), edgecolor="black")
    for bar, val in zip(bars, recall.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Recall (%)")
    ax.set_xlabel("Anomaly Type")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    out = FIG_DIR / "07_recall_by_anomaly_type.png"
    fig.savefig(out)
    plt.close(fig)
    _print_saved(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("Network Anomaly Detection - Visualization")
    print(f"Project root : {ROOT}")
    print(f"Output dir   : {FIG_DIR}")
    print("=" * 70)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data()
    preds = load_predictions()

    steps: Tuple[Tuple[str, Callable[[], None]], ...] = (
        ("01_label_distribution", lambda: plot_label_distribution(raw)),
        ("02_anomaly_type_distribution", lambda: plot_anomaly_type_distribution(raw)),
        ("03_confusion_matrix", lambda: plot_confusion_matrix(preds)),
        ("04_metrics_bar_chart", lambda: plot_metrics()),
        ("05_anomaly_score_distribution", lambda: plot_score_distribution(preds)),
        ("06_false_positive_feature_comparison", lambda: plot_false_positive_features(preds, raw)),
        ("07_recall_by_anomaly_type", lambda: plot_recall_by_anomaly_type(preds, raw)),
    )

    succeeded: List[str] = []
    for name, fn in steps:
        try:
            fn()
            succeeded.append(name)
        except Exception as exc:  # noqa: BLE001 - keep other figures alive
            print(f"  [WARNING] Figure '{name}' failed ({exc}); skipped.")

    generated = sorted(p.name for p in FIG_DIR.glob("*.png"))
    print("-" * 70)
    print(f"Summary: {len(generated)} figure(s) generated in {FIG_DIR}")
    for name in generated:
        print(f"  - {name}")
    print("Done.")
    return 0 if generated else 1


if __name__ == "__main__":
    sys.exit(main())
