"""Step 8: Clustering Analysis (Task 3).

Groups the 2,596 stock-days using k-means (k=3, 4, 5) without labels.
Preprocessing:
  - Standardize all 6 clustering features (fit on the clustering data itself).

Outputs:
  data/processed/clustering_results.csv (ticker, feature_date, clusters for k=3,4,5, plus SSE)
  figures/fig9_kmeans_elbow.png         (SSE vs k)
  Console prints: cluster profiles and Adjusted Rand Index comparing k=3 to actual classes.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from common import FIGURES, PROCESSED, load_config, to_tier

FEATURES = ["realized_volatility", "range_pct", "return_pct", "log_volume", 
            "rel_volume", "liquidity_rank"]
SEED = 42

def main():
    config = load_config()
    cutoffs = config["volatility_cutoffs"]
    
    # 1. Load Data
    data = pd.read_csv(PROCESSED / "classification_data.csv")
    
    # Add today's tier to compare against k=3 later
    data["today_tier"] = to_tier(data["realized_volatility"], cutoffs)
    
    # 2. Preprocess: Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[FEATURES])
    
    # 3. K-Means for k=3, 4, 5
    results_df = data[["ticker", "feature_date", "today_tier"]].copy()
    sse_list = []
    k_values = [3, 4, 5]
    
    print("[Clustering] Fitting K-Means...")
    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=SEED)
        labels = kmeans.fit_predict(scaled_features)
        results_df[f"cluster_k{k}"] = labels
        sse_list.append(kmeans.inertia_)
        
    # 4. Evaluate and export SSE
    sse_df = pd.DataFrame({"k": k_values, "SSE (Inertia)": sse_list})
    print("\n[Clustering] Sum of Squared Errors (SSE):")
    print(sse_df.to_string(index=False))

    results_df.to_csv(PROCESSED / "clustering_results.csv", index=False)

    # 5. Elbow Chart
    FIGURES.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_values, sse_list, marker="o", linestyle="-", color="tab:purple")
    ax.set_xticks(k_values)
    ax.set_title("K-Means Elbow Chart")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Sum of Squared Errors (SSE)")
    fig.savefig(FIGURES / "fig9_kmeans_elbow.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 6. Cluster Profiles & Cross-tabulation (k=3)
    data["cluster_k3"] = results_df["cluster_k3"]
    profiles = data.groupby("cluster_k3")[FEATURES].mean()
    print("\n[Clustering] Cluster Profiles for k=3 (Average Feature Values):")
    print(profiles.round(4).to_string())

    # Crosstab vs Today's Volatility Tier
    crosstab = pd.crosstab(data["cluster_k3"], data["today_tier"], 
                           dropna=False)[["Low", "Medium", "High"]]
    print("\n[Clustering] Crosstab: k=3 Clusters vs Today's Volatility Tier:")
    print(crosstab.to_string())

    # Adjusted Rand Index
    ari = adjusted_rand_score(data["today_tier"], data["cluster_k3"])
    print(f"\n[Clustering] Adjusted Rand Index (k=3 vs Today's Tier): {ari:.4f}")
    print("(1.0 = identical grouping, 0.0 = random agreement)")
    print("Saved clustering outputs to data/processed/ and figures/")

if __name__ == "__main__":
    main()