"""Step 4: export the course workbook, 221953930-219431865.xlsx.

One sheet per table. The main analysis days only (calibration days are used
for the cutoffs and stay in the CSV/Parquet files). Result tables from later
steps are added automatically when their file exists, so rerun this step last.
"""
import pandas as pd

from common import PROCESSED, WORKBOOK, load_config, write_workbook

# sheet name -> file in data/processed (added only if the file exists)
RESULT_SHEETS = {
    "PCA_Scores": "pca_scores.csv",
    "PCA_Loadings": "pca_loadings.csv",
    "Classification_Results": "classification_results.csv",
    "Clustering_Results": "clustering_results.csv",
}


def main():
    config = load_config()
    calib = config["calibration_days"]
    day = pd.read_csv(PROCESSED / "stock_day.csv")
    bars = pd.read_parquet(PROCESSED / "intraday_bars.parquet")
    sheets = {
        "Stock_Profile": pd.read_csv(PROCESSED / "stock_profile.csv", dtype={"cik": str}),
        "Stock_Day": day[~day["is_calibration_day"]].drop(columns=["is_calibration_day"]),
        "Intraday_Bars": bars[~bars["session_date"].isin(calib)],
        "Classification_Data": pd.read_csv(PROCESSED / "classification_data.csv"),
    }
    for name, file in RESULT_SHEETS.items():
        if (PROCESSED / file).exists():
            sheets[name] = pd.read_csv(PROCESSED / file)
    write_workbook(WORKBOOK, sheets)
    for name, df in sheets.items():
        print(f"{name}: {len(df):,} rows")
    print("Saved", WORKBOOK.name)


if __name__ == "__main__":
    main()
