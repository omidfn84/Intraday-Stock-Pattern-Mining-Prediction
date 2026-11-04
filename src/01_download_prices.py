"""Step 1: download 5-minute price bars for all 59 stocks from Alpaca.

Two windows, both from config.json:
  study   = first calibration day .. main_end      (calibration + main analysis days)
  history = history_start .. history_end           (extra year for the LSTM model)
Settings: SIP feed (all U.S. exchanges), split-adjusted, regular + extended hours.
Saves one Parquet file per stock per window in data/raw/prices/, plus manifest.json
recording the exact trading days and settings. Files already saved are skipped.

  python src/01_download_prices.py --dry-run    # only checks the trading days
"""
import json
import sys
import time
from datetime import datetime, timedelta, timezone

from common import RAW_PRICES, alpaca_client, load_config, trading_days


def windows(config):
    return {"study": (config["calibration_days"][0], config["main_end"]),
            "history": (config["history_start"], config["history_end"])}


def download_one(client, ticker, first_day, last_day):
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=datetime.fromisoformat(first_day).replace(tzinfo=timezone.utc),
        end=(datetime.fromisoformat(last_day) + timedelta(days=1)).replace(tzinfo=timezone.utc),
        feed=DataFeed.SIP,
        adjustment=Adjustment.SPLIT,          # no fake price jumps from stock splits
    )
    return client.get_stock_bars(request).df.reset_index()


def main():
    try:
        import pyarrow  # noqa: F401  (needed to save Parquet files)
    except ImportError:
        sys.exit("pyarrow is missing. Run:  python -m pip install pyarrow   then try again.")
    config = load_config()
    tickers = config["tickers"]

    manifest = {"feed": "sip", "adjustment": "split", "timeframe": "5Min",
                "tickers": tickers, "windows": {}}
    for name, (start, end) in windows(config).items():
        days, early = trading_days(start, end)
        manifest["windows"][name] = {"first_day": days[0], "last_day": days[-1],
                                     "n_days": len(days), "early_closes_excluded": early,
                                     "days": days}
        print(f"{name}: {len(days)} full trading days, {days[0]} to {days[-1]}; "
              f"early closes excluded: {early or 'none'}")
    if "--dry-run" in sys.argv:
        print("Dry run only: nothing downloaded.")
        return

    client = alpaca_client()
    failed, saved = [], 0
    for name, (start, end) in windows(config).items():
        folder = RAW_PRICES / name
        folder.mkdir(parents=True, exist_ok=True)
        for i, ticker in enumerate(tickers, 1):
            path = folder / f"bars_{ticker}.parquet"
            if path.exists():
                continue
            for attempt in range(3):
                try:
                    df = download_one(client, ticker, start, end)
                    df.to_parquet(path, index=False)
                    saved += 1
                    print(f"[{name} {i}/{len(tickers)}] {ticker}: {len(df):,} rows")
                    break
                except Exception as e:
                    print(f"[{name}] {ticker} attempt {attempt + 1} failed: {e}")
                    time.sleep(5 * (attempt + 1))
            else:
                failed.append(f"{name}/{ticker}")
            time.sleep(0.3)

    manifest["downloaded_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["failed"] = failed
    (RAW_PRICES / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Done. {saved} files saved this run. Failed: {failed or 'none'}")


if __name__ == "__main__":
    main()
