"""Step 7: Principal Component Analysis (Task 1).

Reduces the 7 numeric columns from Intraday_Bars down to 3 principal components.
Preprocessing:
  - Log-transform volume and trade_count (heavily right-skewed).
  - Standardize all 7 columns (z-score) so volume doesn't swamp prices.

Outputs:
  data/processed/pca_scores.csv      (ticker, bar_start_utc, PC1, PC2, PC3)
  data/processed/pca_loadings.csv    (weights of original columns in each PC)
  figures/fig6_pca_variance.png      (explained variance chart)
  figures/fig7_pca_loadings.png      (heatmap of component loadings)
  figures/fig8_pca_scatter.png       (PC1 vs PC2 colored by session_stage)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common import FIGURES, PROCESSED, load_config

NUMERIC_COLS = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
N_COMPONENTS = 3
SEED = 42

def main():
    config = load_config()
    calib_days = config["calibration_days"]
    
    # 1. Load data and filter out calibration days (leaves ~207,000 rows for the study window)
    bars = pd.read_parquet(PROCESSED / "intraday_bars.parquet")
    bars = bars[~bars["session_date"].isin(calib_days)].copy()
    
    # 2. Preprocess: Log-transform skewed columns, then standardize all
    bars_numeric = bars[NUMERIC_COLS].copy()
    bars_numeric["volume"] = np.log1p(bars_numeric["volume"])
    bars_numeric["trade_count"] = np.log1p(bars_numeric["trade_count"])
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(bars_numeric)
    
    # 3. Run PCA
    pca = PCA(n_components=N_COMPONENTS, random_state=SEED)
    pca_scores = pca.fit_transform(scaled_data)
    
    # 4. Record Explained Variance & Loadings
    explained_var = pca.explained_variance_ratio_
    loadings = pd.DataFrame(
        pca.components_.T, 
        columns=[f"PC{i+1}" for i in range(N_COMPONENTS)], 
        index=NUMERIC_COLS
    )
    
    print(f"[PCA] Explained variance: PC1 {explained_var[0]:.1%}, "
          f"PC2 {explained_var[1]:.1%}, PC3 {explained_var[2]:.1%} "
          f"(Total {explained_var.sum():.1%})")
    print(f"[PCA] Loadings:\n{loadings.round(3)}")

    # Prepare export files
    scores_df = bars[["ticker", "bar_start_utc"]].copy()
    for i in range(N_COMPONENTS):
        scores_df[f"PC{i+1}"] = pca_scores[:, i]
    
    loadings_export = loadings.copy()
    loadings_export.loc["Explained_Variance"] = explained_var
    loadings_export = loadings_export.reset_index().rename(columns={"index": "Metric"})

    scores_df.to_csv(PROCESSED / "pca_scores.csv", index=False)
    loadings_export.to_csv(PROCESSED / "pca_loadings.csv", index=False)

    # 5. Charts
    FIGURES.mkdir(exist_ok=True)
    
    # Chart A: Explained Variance
    fig, ax = plt.subplots(figsize=(6, 4))
    x = [f"PC{i+1}" for i in range(N_COMPONENTS)]
    ax.bar(x, explained_var, color="skyblue", label="Individual variance")
    ax.plot(x, np.cumsum(explained_var), marker="o", color="red", label="Cumulative variance")
    ax.set_ylim(0, 1.05)
    ax.set_title("PCA Explained Variance")
    ax.set_ylabel("Proportion of Variance Explained")
    ax.legend(loc="right")
    fig.savefig(FIGURES / "fig6_pca_variance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Chart B: Loadings Heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(loadings.values, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, label="Weight")
    ax.set_xticks(range(N_COMPONENTS))
    ax.set_yticks(range(len(NUMERIC_COLS)))
    ax.set_xticklabels([f"PC{i+1}" for i in range(N_COMPONENTS)])
    ax.set_yticklabels(NUMERIC_COLS)
    for i in range(len(NUMERIC_COLS)):
        for j in range(N_COMPONENTS):
            ax.text(j, i, f"{loadings.values[i, j]:.2f}", ha="center", va="center", 
                    color="white" if abs(loadings.values[i, j]) > 0.4 else "black")
    ax.set_title("PCA Component Loadings")
    fig.savefig(FIGURES / "fig7_pca_loadings.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Chart C: Scatter PC1 vs PC2 colored by session_stage
    bars["PC1"] = pca_scores[:, 0]
    bars["PC2"] = pca_scores[:, 1]
    sample = bars.sample(n=5000, random_state=SEED)
    
    stages = sample["session_stage"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(stages)))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    for stage, color in zip(stages, colors):
        subset = sample[sample["session_stage"] == stage]
        ax.scatter(subset["PC1"], subset["PC2"], label=stage, color=color, alpha=0.6, s=15)
    ax.set_title("PC1 (Price Level) vs PC2 (Trading Activity) [5,000 pt sample]")
    ax.set_xlabel(f"PC1 ({explained_var[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({explained_var[1]:.1%} variance)")
    ax.legend(title="Session Stage")
    fig.savefig(FIGURES / "fig8_pca_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved PCA outputs to data/processed/ and figures/")

if __name__ == "__main__":
    main()