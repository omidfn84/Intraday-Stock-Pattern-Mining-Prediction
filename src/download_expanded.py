"""Download extended 5-minute SIP bars for Phase 2 and Phase 3.

Two windows (both split-adjusted, regular + extended hours kept as raw):
  study   = 2026-08-14 .. 2026-10-23   calibration days + Phase 2 main days
  history = 2025-08-01 .. 2026-08-13   Phase 3 LSTM training only

Run from the project folder:
  python src/download_expanded.py --dry-run   # only checks the trading days
  python src/download_expanded.py             # downloads everything
It is safe to rerun: tickers already saved are skipped.
"""
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import pandas_market_calendars as mcal
from dotenv import load_dotenv

# The two date windows we download (first and last trading day, inclusive).
WINDOWS = {
    "study": ("2026-08-14", "2026-10-23"),
    "history": ("2025-08-01", "2026-08-13"),
}
OUT_DIR = "data/raw_v2"


def trading_days(start, end):
    """Full NYSE trading days in [start, end], plus the early-close days we exclude."""
    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date=start, end_date=end)
    early = nyse.early_closes(sched)
    full = sched.drop(early.index)
    return (full.index.strftime("%Y-%m-%d").tolist(),
            early.index.strftime("%Y-%m-%d").tolist())


def download_one(client, ticker, first_day, last_day):
    """One ticker, one window, as a table straight from Alpaca."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=datetime.fromisoformat(first_day).replace(tzinfo=timezone.utc),
        end=(datetime.fromisoformat(last_day) + timedelta(days=1)).replace(tzinfo=timezone.utc),
        feed=DataFeed.SIP,
        adjustment=Adjustment.SPLIT,
    )
    return client.get_stock_bars(request).df.reset_index()


def main():
    # Work from the project folder, wherever the script is started from.
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dry_run = "--dry-run" in sys.argv
    try:
        import pyarrow  # noqa: F401  (needed to save Parquet files)
    except ImportError:
        sys.exit("pyarrow is missing. Run:  python -m pip install pyarrow   then try again.")
    config = json.load(open("config.json"))
    tickers = config["tickers"]

    # 1. Work out the exact trading days with the official NYSE calendar.
    manifest = {"feed": "sip", "adjustment": "split", "timeframe": "5Min",
                "tickers": tickers, "windows": {}}
    for name, (start, end) in WINDOWS.items():
        days, early = trading_days(start, end)
        manifest["windows"][name] = {"first_day": days[0], "last_day": days[-1],
                                     "n_days": len(days), "early_closes_excluded": early,
                                     "days": days}
        print(f"{name}: {len(days)} full trading days, {days[0]} to {days[-1]}; "
              f"early closes excluded: {early or 'none'}")

    if dry_run:
        print("Dry run only: nothing downloaded.")
        return

    # 2. Connect to Alpaca with the keys in .env.
    from alpaca.data.historical import StockHistoricalDataClient
    load_dotenv()
    client = StockHistoricalDataClient(os.getenv("ALPACA_KEY"), os.getenv("ALPACA_SECRET"))

    # 3. Download every ticker for every window, one file each.
    failed, rows = [], {}
    for name, (start, end) in WINDOWS.items():
        folder = os.path.join(OUT_DIR, name)
        os.makedirs(folder, exist_ok=True)
        for i, ticker in enumerate(tickers, 1):
            path = os.path.join(folder, f"bars_{ticker}.parquet")
            if os.path.exists(path):
                continue
            for attempt in range(3):
                try:
                    df = download_one(client, ticker, start, end)
                    df.to_parquet(path, index=False)
                    rows[f"{name}/{ticker}"] = len(df)
                    print(f"[{name} {i}/{len(tickers)}] {ticker}: {len(df):,} rows")
                    break
                except Exception as e:
                    print(f"[{name}] {ticker} attempt {attempt + 1} failed: {e}")
                    time.sleep(5 * (attempt + 1))
            else:
                failed.append(f"{name}/{ticker}")
            time.sleep(0.3)

    # 4. Record exactly what was downloaded and when.
    manifest["downloaded_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["failed"] = failed
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Done. {len(rows)} files saved this run. Failed: {failed or 'none'}")


if __name__ == "__main__":
    main()
