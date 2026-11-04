# Intraday Patterns in U.S. Stocks

EECS 4412 Data Mining, Fall 2026. Omid Firouzinia (221953930) and Diego Romero (219431865).

Five-minute price bars for 59 U.S. stocks (Alpaca, SIP feed) and company details (SEC EDGAR),
organised into linked tables and used to find recurring trading patterns and to predict
next-day volatility, always testing on later days than the models are trained on.

## The data

| Window | Dates | Trading days | Used for |
|---|---|---|---|
| Calibration | Aug 14 - Aug 20, 2026 | 5 | Setting the Low / Medium / High cutoffs only |
| Main | Aug 21 - Oct 23, 2026 | 45 | All analysis and models |
| History | Aug 1, 2025 - Aug 13, 2026 | 258 | Extra training data for the sequence (LSTM) model |

Tables: **Stock_Profile** (one row per stock), **Stock_Day** (one row per stock per day),
**Intraday_Bars** (one row per stock per 5 minutes), and **Classification_Data** (today's
features with tomorrow's volatility tier).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in this folder with your own Alpaca keys (never commit it):

```
ALPACA_KEY=...
ALPACA_SECRET=...
```

## Run the pipeline

Run the steps in order; each one can be started from any folder. The download steps
skip anything already downloaded.

| Step | Command | What it produces |
|---|---|---|
| 1 | `python src/01_download_prices.py` | `data/raw/prices/` raw Alpaca bars (Parquet) + `manifest.json` |
| 2 | `python src/02_download_company_info.py` | `data/raw/sec/` SEC snapshot, `data/processed/stock_metadata.csv` |
| 3 | `python src/03_build_tables.py` | all tables in `data/processed/`, coverage reports, `build_summary.json` |
| 4 | `python src/04_export_workbook.py` | `221953930-219431865.xlsx` (run last: it also collects result tables) |
| 5 | `python src/05_figures.py` | `figures/` |

Steps 06-08 (PCA, classification, clustering) are being added next.

## Folder layout

```
config.json        tickers, date windows, coverage rule, frozen tier cutoffs, train/test split
src/common.py      shared rules used by every step: cleaning, Stock_Day, tiers, Excel export
src/01-05_*.py     the pipeline, in order
notebooks/         exploration notebooks, numbered in the order the work happened
                   (pilot test, IEX vs SIP check, first cleaning and tables); they use
                   the folder layout of that time, so the scripts in src/ are the
                   version to run
data/raw/          downloads, never edited
data/processed/    everything the pipeline builds
figures/           report figures
archive/           Phase 1 as submitted and the first 25-day download (see archive/README.md)
```

## Key decisions

- **SIP feed, not IEX.** IEX sees about 2.5% of U.S. trading volume, so several stocks had
  no complete trading days. SIP covers all exchanges: 99.99% of possible bars are present.
- **Coverage rule.** A stock-day needs at least 75 of 78 regular-hours bars (95%).
  JBHT failed it and was dropped, leaving 59 stocks.
- **No look-ahead.** Volatility and liquidity cutoffs come only from the 5 calibration days
  and are frozen in `config.json`. Models train on earlier days and are tested on later ones;
  the data is never shuffled.
- **Split-adjusted prices**, so a stock split never shows up as a fake price jump.
- **Data revisions are tracked.** Comparing with the first download (Sep 24) shows identical
  prices; only Sept 15 volume was revised upward by at most 0.6% (late-reported trades).
