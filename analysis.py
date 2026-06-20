"""
Customer Segmentation using RFM + K-Means Clustering
Author: Harsh Pandey

Goal:
    Segment customers into actionable groups based on Recency, Frequency,
    and Monetary (RFM) value using K-Means clustering, then recommend
    targeted marketing actions per segment.

Run:
    pip install pandas matplotlib scikit-learn
    python analysis.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

DATA_PATH = "data/customer_rfm.csv"

SEGMENT_LABELS = {
    0: "Active/Engaged",
    1: "At-Risk",
    2: "High-Value/VIP",
    3: "Frequent Buyers",
}

SEGMENT_COLORS = {
    "Active/Engaged": "#2563eb",
    "At-Risk": "#dc2626",
    "High-Value/VIP": "#d97706",
    "Frequent Buyers": "#16a34a",
}


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def find_optimal_k(scaled_features, max_k: int = 8) -> None:
    inertias = []
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaled_features)
        inertias.append(km.inertia_)

    plt.figure(figsize=(6, 4.5))
    plt.plot(range(1, max_k + 1), inertias, marker="o", color="#2563eb")
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig("elbow_method.png", dpi=130)
    plt.close()


def run_clustering(df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    features = df[["Recency", "Frequency", "Monetary"]]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    find_optimal_k(scaled)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(scaled)
    df["Segment"] = df["Cluster"].map(SEGMENT_LABELS)
    return df


def plot_segments(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7.5, 5))
    for seg, color in SEGMENT_COLORS.items():
        sub = df[df["Segment"] == seg]
        plt.scatter(sub["Recency"], sub["Monetary"], label=seg, color=color, alpha=0.6, s=30)
    plt.title("Customer Segments: Recency vs Monetary Value")
    plt.xlabel("Recency (days since last purchase)")
    plt.ylabel("Monetary Value ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("segments_scatter.png", dpi=130)
    plt.close()

    seg_counts = df["Segment"].value_counts()
    plt.figure(figsize=(7, 4.5))
    seg_counts.plot(kind="bar", color=[SEGMENT_COLORS[s] for s in seg_counts.index])
    plt.title("Customer Count by Segment")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig("segment_sizes.png", dpi=130)
    plt.close()


def main() -> None:
    df = load_data()
    df = run_clustering(df, k=4)
    plot_segments(df)

    summary = df.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().round(1)
    summary["Count"] = df["Segment"].value_counts()
    print("=== Segment Summary (Average RFM) ===")
    print(summary)

    df.to_csv("data/customer_rfm_segmented.csv", index=False)
    summary.to_csv("cluster_summary.csv")
    print("\nSaved: data/customer_rfm_segmented.csv, cluster_summary.csv")
    print("Charts saved: elbow_method.png, segments_scatter.png, segment_sizes.png")


if __name__ == "__main__":
    main()
