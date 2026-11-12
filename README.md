# Intraday Patterns in U.S. Stocks

EECS 4412 Data Mining, Fall 2026. Omid Firouzinia (221953930) and Diego Romero (219431865).

This project studies intraday price and trading activity in 59 U.S. stocks. It collects
five-minute bars, joins SEC company and exchange metadata, builds analysis-ready tables, and
applies classification, principal component analysis (PCA), and clustering. The longer-term
goal is to use this research as a foundation for an automated trading system.

**Current scope:** this repository is a data-mining and modelling pipeline, not a live trading
bot. It predicts next-day volatility tiers for analysis; it does not generate or execute buy/sell
orders, backtest a trading strategy, or implement portfolio, risk, or broker-execution controls.
Model results are experimental and are not financial advice.

## Project Data

Price data comes from Alpaca's SIP feed and is adjusted for splits. Company name, listing
exchange, and CIK are obtained from SEC EDGAR. The configured universe contains 59 tickers;
JBHT was excluded after failing the coverage rule. The date windows and analysis settings live
in [`config.json`](config.json).

| Window | Dates | Trading days | Purpose |
|---|---|---:|---|
| Calibration | Aug 14-20, 2026 | 5 | Set and freeze volatility and liquidity tiers |
| Main study | Aug 21-Oct 23, 2026 | 45 | Descriptive analysis and modelling |
| Historical data | Aug 1, 2025-Aug 13, 2026 | 258 | Additional cleaned price history; not used by the current classifiers |

The main period yields 44 feature dates for next-day classification because the final day has no
following-day label. The configured split is chronological: 26 training, 7 validation, and 11
test feature dates. It is never shuffled.

The main analysis tables are:

| Table | Contents |
|---|---|
| `Stock_Profile` | One row per stock with SEC identifiers, calibration-period averages, and liquidity tier |
| `Stock_Day` | Daily prices, volume, direction, realized volatility, coverage, and volatility tier per stock |
| `Intraday_Bars` | Cleaned regular-session five-minute bars per stock |
| `Classification_Data` | Daily features and recent-history features paired with the next day's volatility tier |

## Repository Guide

```text
config.json                         Universe, dates, cutoffs, and chronological split
src/common.py                       Shared data cleaning, feature, tier, and workbook helpers
src/01_download_prices.py            Download Alpaca SIP bars
src/02_download_company_info.py      Download SEC company and exchange metadata
src/03_build_tables.py               Clean bars, check coverage, and build analysis tables
src/04_export_workbook.py             Export available tables and model results to Excel
src/05_figures.py                     Create exchange and intraday price charts
src/06_classification.py              Train and evaluate next-day volatility classifiers
src/07_pca.py                         Reduce intraday bar features and chart components
src/08_clustering.py                  Cluster stock-days and compare clusters with volatility tiers
notebooks/                            Earlier exploratory notebooks, numbered by work sequence
data/raw/                             Source downloads; preserve these as raw inputs
data/processed/                       Generated tables, model outputs, and build reports
data/archive/                          Early raw downloads and Phase 1 processed tables
figures/                              Generated analysis charts
archive/                              Submitted Phase 1 materials and archive notes
analysis/                             Reserved for analysis materials
```

The numbered notebooks document earlier exploration and may use an older data layout. For the
current reproducible pipeline, use the scripts in `src/`.

## Setup

Python 3 and the packages pinned in [`requirements.txt`](requirements.txt) are required. From
the repository root, create and activate a virtual environment, then install dependencies.

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Price downloads require Alpaca credentials with access to the SIP data feed. Create a local
`.env` file in the repository root:

```text
ALPACA_KEY=your_key
ALPACA_SECRET=your_secret
```

Keep `.env` private and never commit credentials. SEC metadata is downloaded from the public
SEC EDGAR company-ticker file.

## Run The Pipeline

Run commands from the repository root. Download scripts skip existing data; `--dry-run` on the
price downloader checks the trading-day calendar without making API requests. Each downstream
script expects the preceding data products to exist.

| Order | Command | Main outputs |
|---:|---|---|
| 1 | `python src/01_download_prices.py` | Raw Parquet bars under `data/raw/prices/` and a download manifest |
| 2 | `python src/02_download_company_info.py` | SEC source snapshot under `data/raw/sec/` and `data/processed/stock_metadata.csv` |
| 3 | `python src/03_build_tables.py` | Cleaned tables, coverage reports, classification features, and `build_summary.json` under `data/processed/` |
| 4 | `python src/05_figures.py` | `fig1_exchange_bar.png` and `fig2_aapl_price_line.png` |
| 5 | `python src/06_classification.py` | Classifier predictions, metrics, tuning results, and figures 3-5 |
| 6 | `python src/07_pca.py` | PCA scores, loadings, and figures 6-8 |
| 7 | `python src/08_clustering.py` | Cluster assignments, printed profiles and ARI, and `fig9_kmeans_elbow.png` |
| 8 | `python src/04_export_workbook.py` | `221953930-219431865.xlsx` with the core tables and any result tables already generated |

Run the workbook export last to include classification, PCA, and clustering outputs. It adds
each result sheet only when the corresponding CSV exists. To check the calendar before
downloading prices, run `python src/01_download_prices.py --dry-run`.

## Analysis Methods

### Data Preparation

The build step converts timestamps to New York time, keeps regular trading hours (09:30-16:00),
removes duplicate bars, and records checks for invalid prices, inconsistent high/low ranges, and
negative volume. A stock-day passes coverage with at least 75 of 78 expected five-minute bars
(95%). Volatility and liquidity cutoffs are derived from the five calibration days and stored in
`config.json` so the main study does not set its own labels.

Daily realized volatility is calculated from squared five-minute log returns, without bridging
missing intervals. Classification uses six same-day features in v1 and adds four recent-history
features in v2: five-day average volatility, the today-to-yesterday volatility ratio, and
market-wide volatility for today and its five-day average. Decision trees and Gaussian Naive
Bayes are compared with majority-class, persistence, and five-day-average baselines. Tree
settings are selected on validation dates; final models are refit on training plus validation
data and evaluated on the later test dates.

PCA log-transforms volume and trade count, standardizes seven numeric bar fields, and exports
three components. K-means clustering standardizes six daily features and tests k=3, 4, and 5;
the k=3 grouping is compared with same-day volatility tiers using adjusted Rand index. These
are exploratory analyses, not trading signals.

### Key Design Decisions

- **SIP feed:** the consolidated feed covers U.S. exchanges and avoids the coverage gaps seen in
  the initial IEX pilot.
- **Coverage filter:** a stock-day needs at least 75 of 78 regular-session bars. JBHT did not
  meet the project rule and was excluded, leaving 59 stocks.
- **No look-ahead in classification:** cutoffs are based on calibration data, history features
  use only current and earlier dates, and train/validation/test splits follow time order.
- **Split-adjusted prices:** adjustments prevent stock splits from appearing as artificial price
  jumps.
- **Versioned feature experiment:** both same-day-only v1 and history-enriched v2 results are
  retained, with simple baselines for context.
- **Raw data preservation:** original downloads are kept under `data/raw/`; cleaning and analysis
  products are written to `data/processed/`.

## Outputs

Generated outputs are written to `data/processed/` and `figures/`:

- `stock_metadata.csv`, `stock_profile.csv`, `stock_day.csv`, and `intraday_bars.parquet` contain
  company metadata and cleaned stock-level tables.
- `classification_data.csv`, `classification_results.csv`, `classification_metrics.json`, and
  `tree_tuning.csv` contain model inputs, test predictions, evaluation metrics, and tree search
  results.
- `pca_scores.csv` and `pca_loadings.csv` contain PCA coordinates and component weights.
- `clustering_results.csv` contains stock-day cluster assignments for k=3, 4, and 5.
- `coverage_study.csv`, `coverage_history.csv`, and `build_summary.json` report data completeness
  and build checks.
- `figures/fig1_*.png` through `figures/fig9_*.png` visualize the data and analyses.
- `221953930-219431865.xlsx` is the course workbook export; rerun the export after analyses to
  include their available result tables.

## Archive

[`archive/README.md`](archive/README.md) describes the earlier Phase 1 submission and initial
data downloads. `data/archive/` preserves the first raw IEX and SIP downloads and Phase 1
processed tables. The current pipeline does not use archived files as inputs, except for an
optional consistency comparison in `src/03_build_tables.py` when the Phase 1 cleaned bars are
present.
