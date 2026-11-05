"""
Customer Segmentation using RFM + K-Means Clustering

Run from any working directory with: python analysis.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "customer_rfm.csv"
FEATURES = ["Recency", "Frequency", "Monetary"]
SEGMENT_COLORS = {
    "Active/Engaged": "#2563eb",
    "At-Risk": "#dc2626",
    "High-Value/VIP": "#d97706",
    "Frequent Buyers": "#16a34a",
}


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"RFM dataset not found: {path}")
    df = pd.read_csv(path)
    missing = [column for column in FEATURES if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    df = df.copy()
    for column in FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[FEATURES].isna().any().any():
        raise ValueError("RFM features contain missing or non-numeric values")
    if (df[FEATURES] < 0).any().any():
        raise ValueError("RFM features cannot contain negative values")
    if df.empty:
        raise ValueError("RFM dataset contains no customers")
    return df


def find_optimal_k(scaled_features, max_k: int = 8) -> None:
    max_k = min(max_k, len(scaled_features))
    if max_k < 1:
        raise ValueError("At least one customer is required")
    inertias = []
    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(scaled_features)
        inertias.append(model.inertia_)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(range(1, max_k + 1), inertias, marker="o", color="#2563eb")
    ax.set(title="Elbow Method for Optimal K", xlabel="Number of Clusters (k)", ylabel="Inertia")
    fig.tight_layout()
    fig.savefig(ROOT / "elbow_method.png", dpi=130)
    plt.close(fig)


def run_clustering(df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    if not 1 <= k <= len(df):
        raise ValueError(f"k must be between 1 and the number of customers ({len(df)})")
    result = df.copy()
    scaled = StandardScaler().fit_transform(result[FEATURES])
    if len(result) > 1:
        find_optimal_k(scaled)
    result["Cluster"] = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)

    # Cluster IDs are arbitrary; assign stable business labels from each cluster's
    # RFM profile rather than assuming a particular K-Means label ordering.
    profile = result.groupby("Cluster")[FEATURES].mean()
    labels = {}
    if k == 4:
        ranked = profile.sort_values(["Monetary", "Frequency"], ascending=False).index.tolist()
        labels[ranked[0]] = "High-Value/VIP"
        labels[ranked[1]] = "Frequent Buyers"
        labels[ranked[2]] = "Active/Engaged"
        labels[ranked[3]] = "At-Risk"
    else:
        labels = {cluster: f"Segment {i + 1}" for i, cluster in enumerate(profile.index)}
    result["Segment"] = result["Cluster"].map(labels)
    return result


def plot_segments(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for segment, subset in df.groupby("Segment"):
        ax.scatter(
            subset["Recency"], subset["Monetary"],
            label=segment, color=SEGMENT_COLORS.get(segment, "#64748b"), alpha=0.6, s=30,
        )
    ax.set(title="Customer Segments: Recency vs Monetary Value",
           xlabel="Recency (days since last purchase)", ylabel="Monetary Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "segments_scatter.png", dpi=130)
    plt.close(fig)

    counts = df["Segment"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts.plot(kind="bar", color=[SEGMENT_COLORS.get(s, "#64748b") for s in counts.index], ax=ax)
    ax.set(title="Customer Count by Segment", ylabel="Number of Customers")
    fig.tight_layout()
    fig.savefig(ROOT / "segment_sizes.png", dpi=130)
    plt.close(fig)


def main() -> None:
    df = run_clustering(load_data(), k=4)
    plot_segments(df)
    summary = df.groupby("Segment")[FEATURES].mean().round(1)
    summary["Count"] = df["Segment"].value_counts()
    print("=== Segment Summary (Average RFM) ===")
    print(summary)
    df.to_csv(ROOT / "customer_rfm_segmented.csv", index=False)
    summary.to_csv(ROOT / "cluster_summary.csv")
    print("Saved segmented data, summary, and charts in", ROOT)


if __name__ == "__main__":
    main()
