"""Shared code for the EECS 4412 stock project.

Every numbered script in src/ imports from here, so each rule (cleaning,
daily summaries, tiers, Excel export) is written exactly once.
"""
import json
import os
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # the project folder
DATA = ROOT / "data"
RAW_PRICES = DATA / "raw" / "prices"                  # Alpaca downloads (Parquet, never edited)
RAW_SEC = DATA / "raw" / "sec"                        # SEC EDGAR snapshot (never edited)
PROCESSED = DATA / "processed"                        # everything the pipeline builds
FIGURES = ROOT / "figures"
CONFIG_PATH = ROOT / "config.json"
WORKBOOK = ROOT / "221953930-219431865.xlsx"          # course deliverable (both student numbers)

EXPECTED_BARS = 78                                    # 9:30-16:00 = 78 five-minute bars
STAGE_BINS = [570, 600, 690, 810, 930, 960]           # minutes after midnight, New York time
STAGE_LABELS = ["Opening", "Morning", "Midday", "Afternoon", "Closing"]
BAR_COLUMNS = ["symbol", "feed", "bar_start_utc", "session_date", "session_stage",
               "open", "high", "low", "close", "volume", "trade_count", "vwap"]


# ------------------------------------------------------------------ settings
def load_config():
    """Read config.json (tickers, dates, cutoffs)."""
    return json.loads(CONFIG_PATH.read_text())


def save_config(config):
    CONFIG_PATH.write_text(json.dumps(config, indent=2))


def min_bars(config):
    """Bars a stock-day needs to pass the coverage rule (95% of 78 = 75)."""
    return int(np.ceil(config["coverage_threshold"] * EXPECTED_BARS))


def trading_days(start, end):
    """Full NYSE trading days in [start, end], and the early-close days left out."""
    import pandas_market_calendars as mcal
    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date=start, end_date=end)
    early = nyse.early_closes(sched)
    full = sched.drop(early.index)
    return (full.index.strftime("%Y-%m-%d").tolist(),
            early.index.strftime("%Y-%m-%d").tolist())


def alpaca_client():
    """Alpaca data client, using the keys in .env (never printed or committed)."""
    from alpaca.data.historical import StockHistoricalDataClient
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    return StockHistoricalDataClient(os.getenv("ALPACA_KEY"), os.getenv("ALPACA_SECRET"))


# ------------------------------------------------------------------ cleaning
def clean_bars(raw, days=None):
    """Raw Alpaca bars -> regular-hours bars ready for analysis.

    Converts UTC to New York time, keeps 9:30-16:00, optionally keeps only the
    given trading days, adds session_date / session_stage / feed, and runs the
    four quality checks. Returns (bars, checks).
    """
    bar_start = pd.to_datetime(raw["timestamp"], utc=True)
    ny = bar_start.dt.tz_convert("America/New_York")
    keep = (ny.dt.time >= time(9, 30)) & (ny.dt.time < time(16, 0))

    bars = raw[keep].drop(columns="timestamp").copy()
    bars["bar_start_utc"] = bar_start[keep]
    bars["session_date"] = ny[keep].dt.strftime("%Y-%m-%d")
    if days is not None:
        bars = bars[bars["session_date"].isin(set(days))]

    minutes = ny[bars.index].dt.hour * 60 + ny[bars.index].dt.minute
    bars["session_stage"] = pd.cut(minutes, bins=STAGE_BINS, labels=STAGE_LABELS,
                                   right=False).astype(str)
    bars["feed"] = "sip"

    prices = bars[["open", "high", "low", "close"]]
    checks = {
        "duplicates": int(bars.duplicated(["symbol", "bar_start_utc"]).sum()),
        "non_positive_prices": int((prices <= 0).any(axis=1).sum()),
        "open_close_outside_range": int(((bars["low"] > bars[["open", "close"]].min(axis=1)) |
                                         (bars["high"] < bars[["open", "close"]].max(axis=1))).sum()),
        "negative_volume": int((bars["volume"] < 0).sum()),
    }
    bars = bars.drop_duplicates(["symbol", "bar_start_utc"])
    return bars[BAR_COLUMNS].reset_index(drop=True), checks


def coverage_report(bars, tickers, days, config):
    """Per stock: how many days pass the 95% coverage rule."""
    counts = (bars.groupby(["symbol", "session_date"]).size()
                  .unstack(fill_value=0).reindex(index=tickers, columns=days, fill_value=0))
    report = pd.DataFrame({"days": len(days),
                           "good_days": (counts >= min_bars(config)).sum(axis=1),
                           "min_bars": counts.min(axis=1)})
    report["bad_days"] = report["days"] - report["good_days"]
    report["bad_pct"] = (100 * report["bad_days"] / report["days"]).round(2)
    report.index.name = "ticker"
    return report


# ------------------------------------------------------------ daily summaries
def stock_day(bars):
    """One row per stock per day: open/close/high/low, volume, direction, volatility.

    Realized volatility = sqrt(sum of squared 5-minute log returns), using only
    bars exactly 5 minutes apart on the same day (gaps are never bridged).
    """
    bars = bars.sort_values(["symbol", "bar_start_utc"]).reset_index(drop=True)
    g = bars.groupby(["symbol", "session_date"])
    day = g.agg(session_open=("open", "first"), session_close=("close", "last"),
                session_high=("high", "max"), session_low=("low", "min"),
                session_volume=("volume", "sum"), observed_bars=("close", "size")).reset_index()
    day["day_direction"] = np.select([day["session_close"] > day["session_open"],
                                      day["session_close"] < day["session_open"]],
                                     ["Up", "Down"], default="Flat")
    back_to_back = g["bar_start_utc"].diff() == pd.Timedelta(minutes=5)
    log_ret = pd.Series(np.where(back_to_back, np.log(bars["close"] / g["close"].shift(1)), np.nan),
                        index=bars.index)
    rv = (log_ret ** 2).groupby([bars["symbol"], bars["session_date"]]).sum(min_count=1) ** 0.5
    return day.merge(rv.rename("realized_volatility").reset_index(), on=["symbol", "session_date"])


def tercile_cutoffs(values):
    """The two values that split `values` into three equal-sized groups."""
    q1, q2 = values.quantile([1 / 3, 2 / 3])
    return float(q1), float(q2)


def to_tier(values, cutoffs):
    """Low / Medium / High using fixed cutoffs (at or below the cutoff = lower tier)."""
    return pd.cut(values, bins=[-np.inf, cutoffs[0], cutoffs[1], np.inf],
                  labels=["Low", "Medium", "High"]).astype(str)


def stock_profile(day, metadata, config, liquidity_cutoffs=None):
    """One row per stock: SEC details + calibration-day averages + liquidity tier.

    The liquidity cutoffs are computed from the calibration days unless given.
    """
    calib = day[day["session_date"].isin(config["calibration_days"])]
    prof = calib.groupby("symbol").agg(
        mean_daily_volume=("session_volume", "mean"),
        mean_close_usd=("session_close", "mean"),
        valid_calibration_days=("observed_bars", lambda b: (b >= min_bars(config)).sum()),
    ).reset_index()
    if liquidity_cutoffs is None:
        liquidity_cutoffs = tercile_cutoffs(prof["mean_daily_volume"])
    prof["liquidity_tier"] = to_tier(prof["mean_daily_volume"], liquidity_cutoffs)
    profile = metadata.merge(prof, left_on="ticker", right_on="symbol").drop(columns="symbol")
    profile = profile.rename(columns={"name": "company_name"})
    profile["calibration_end_date"] = config["calibration_days"][-1]
    return profile, liquidity_cutoffs


# --------------------------------------------------------------------- export
def write_workbook(path, sheets):
    """Write {sheet name: DataFrame} to one Excel file, 20-wide columns, no index.

    Excel cannot store time zones, so timezone-aware columns are written as UTC
    without the zone tag (the column names say UTC).
    """
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df = df.copy()
            for col in df.columns:
                if isinstance(df[col].dtype, pd.DatetimeTZDtype):
                    df[col] = df[col].dt.tz_convert("UTC").dt.tz_localize(None)
            df.to_excel(writer, sheet_name=name, index=False)
        for sheet in writer.sheets.values():
            for col in sheet.columns:
                sheet.column_dimensions[col[0].column_letter].width = 20
