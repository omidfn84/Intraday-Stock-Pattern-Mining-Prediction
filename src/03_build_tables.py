"""Step 3: clean the raw bars and build every table the analysis uses.

  1. Clean      New York time, regular hours (9:30-16:00), full trading days only,
                four quality checks, session_stage
  2. Coverage   95% rule (at least 75 of 78 bars) per stock-day, both windows
  3. Cutoffs    volatility and liquidity tiers from the 5 calibration days only,
                frozen in config.json (no information from the analysis days)
  4. Tables     Stock_Profile, Stock_Day, Intraday_Bars
  5. Modelling  today's features -> tomorrow's volatility tier, split by date

Outputs in data/processed/ (every check is also saved in build_summary.json):
  stock_profile.csv, stock_day.csv, intraday_bars.parquet, classification_data.csv,
  history_bars.parquet, coverage_study.csv, coverage_history.csv
"""
import json

import numpy as np
import pandas as pd

from common import (DATA, PROCESSED, RAW_PRICES, clean_bars, coverage_report, load_config,
                    min_bars, save_config, stock_day, stock_profile, tercile_cutoffs, to_tier)

FEATURES = ["realized_volatility", "range_pct", "return_pct", "log_volume",
            "rel_volume", "liquidity_rank"]
PROFILE_COLUMNS = ["ticker", "cik", "company_name", "exchange", "snapshot_date",
                   "calibration_end_date", "mean_daily_volume", "mean_close_usd",
                   "valid_calibration_days", "liquidity_tier"]

config = load_config()
manifest = json.loads((RAW_PRICES / "manifest.json").read_text())
TICKERS = config["tickers"]
CALIB_DAYS = config["calibration_days"]
summary = {}


def clean_window(window):
    """Steps 1-2 for one download window."""
    days = manifest["windows"][window]["days"]
    raw = pd.concat([pd.read_parquet(RAW_PRICES / window / f"bars_{t}.parquet") for t in TICKERS],
                    ignore_index=True)
    bars, checks = clean_bars(raw, days)
    bars = bars.sort_values(["symbol", "bar_start_utc"]).reset_index(drop=True)
    possible = len(TICKERS) * len(days) * 78
    report = coverage_report(bars, TICKERS, days, config)
    report.to_csv(PROCESSED / f"coverage_{window}.csv")
    summary[window] = {"days": len(days), "raw_rows": len(raw), "clean_rows": len(bars),
                       "max_possible": possible,
                       "coverage_pct": round(100 * len(bars) / possible, 3), "checks": checks,
                       "bad_stock_days": int(report["bad_days"].sum()),
                       "stocks_over_5pct_bad": report.index[report["bad_pct"] > 5].tolist()}
    print(f"[{window}] {len(days)} days: raw {len(raw):,} -> clean {len(bars):,} of {possible:,} "
          f"({summary[window]['coverage_pct']}%); checks {checks}; "
          f"stock-days below {min_bars(config)} bars: {summary[window]['bad_stock_days']}; "
          f"stocks over 5% bad: {summary[window]['stocks_over_5pct_bad'] or 'none'}")
    return bars


def freeze_cutoffs(day, metadata):
    """Step 3. Cutoffs come from the calibration days only. If config.json already
    holds the same values (up to rounding noise between computers), the stored ones
    are kept, so a day sitting exactly on a cutoff always gets the same tier."""
    calib = day["session_date"].isin(CALIB_DAYS)
    computed = {"volatility_cutoffs": tercile_cutoffs(day.loc[calib, "realized_volatility"]),
                "liquidity_cutoffs": stock_profile(day, metadata, config)[1]}
    for key, new in computed.items():
        old = config.get(key)
        if old is None or not np.allclose(old, new, rtol=1e-9):
            print(f"[cutoffs] {key} set to {new}")
            config[key] = list(new)
    config["quantile_method"] = "pandas default (linear interpolation)"
    save_config(config)
    summary["cutoffs"] = {k: config[k] for k in computed}
    print(f"[cutoffs] volatility {config['volatility_cutoffs']}, "
          f"liquidity {config['liquidity_cutoffs']}")


def modelling_table(day, profile):
    """Step 5. One row per stock per day: today's features and tomorrow's tier."""
    main_days = [d for d in manifest["windows"]["study"]["days"] if d not in CALIB_DAYS]
    feature_days = main_days[:-1]                  # the last day has no "tomorrow"
    n_train, n_val = config["split"]["train_days"], config["split"]["validation_days"]
    split = {d: "train" if i < n_train else "validation" if i < n_train + n_val else "test"
             for i, d in enumerate(feature_days)}

    d = day[day["session_date"].isin(main_days)].sort_values(["ticker", "session_date"]).copy()
    tomorrow = d.groupby("ticker")[["session_date", "volatility_tier", "coverage_ok"]].shift(-1)
    d["label_date"] = tomorrow["session_date"]
    d["next_volatility_tier"] = tomorrow["volatility_tier"]
    d["label_coverage_ok"] = tomorrow["coverage_ok"]
    d = d[d["session_date"].isin(feature_days)]

    d = d.merge(profile[["ticker", "mean_daily_volume", "liquidity_tier"]], on="ticker")
    d["range_pct"] = (d["session_high"] - d["session_low"]) / d["session_open"]
    d["return_pct"] = (d["session_close"] - d["session_open"]) / d["session_open"]
    d["log_volume"] = np.log(d["session_volume"])
    d["rel_volume"] = d["session_volume"] / d["mean_daily_volume"]
    d["liquidity_rank"] = d["liquidity_tier"].map({"Low": 1, "Medium": 2, "High": 3})
    d["split"] = d["session_date"].map(split)

    n_all = len(d)
    usable = d["coverage_ok"] & d["label_coverage_ok"].astype(bool) & d["realized_volatility"].notna()
    d = d[usable].rename(columns={"session_date": "feature_date"})
    d = d[["ticker", "feature_date", "label_date"] + FEATURES +
          ["next_volatility_tier", "split"]].reset_index(drop=True)

    counts = pd.crosstab(d["split"], d["next_volatility_tier"]).reindex(
        index=["train", "validation", "test"], columns=["Low", "Medium", "High"])
    counts["rows"] = counts.sum(axis=1)
    dates = d.groupby("split")["feature_date"].agg(["min", "max", "nunique"])
    summary["modelling"] = {"rows": len(d), "dropped_for_coverage": n_all - len(d),
                            "split_counts": counts.to_dict("index"),
                            "split_dates": dates.to_dict("index")}
    print(f"[modelling] {len(d):,} rows ({n_all - len(d)} dropped)\n{counts.join(dates).to_string()}")
    return d


def check_against_first_download(bars):
    """Compare with the project's first download (Aug 14 - Sep 18, taken Sep 24 and kept
    in data/archive). A later stock split would change prices; a late trade correction
    by the exchanges would change volume. Both are recorded in build_summary.json."""
    first = DATA / "archive" / "processed_phase1" / "bars_clean.csv"
    if not first.exists():
        return
    old = pd.read_csv(first, parse_dates=["bar_start_utc"])
    both = bars.merge(old, on=["symbol", "bar_start_utc"], suffixes=("_new", "_old"))
    ratio = (both["close_new"] / both["close_old"]).groupby(both["symbol"]).agg(["min", "max"])
    split_like = ratio[(ratio["min"] < 0.999) | (ratio["max"] > 1.001)].index.tolist()
    revised = both[both["volume_new"] != both["volume_old"]]
    by_day = revised.groupby("session_date_new").agg(
        bars=("symbol", "size"), stocks=("symbol", "nunique"),
        extra_volume=("volume_new", "sum")).reset_index()
    by_day["extra_volume"] -= revised.groupby("session_date_new")["volume_old"].sum().values
    summary["first_download_check"] = {
        "matched_bars": len(both), "first_download_bars": len(old),
        "tickers_with_split_like_price_changes": split_like,
        "bars_with_revised_volume": len(revised),
        "revised_volume_by_day": by_day.rename(columns={"session_date_new": "day"}).to_dict("records")}
    print(f"[first download check] {len(both):,} of {len(old):,} bars matched; "
          f"split-like price changes: {split_like or 'none'}; "
          f"bars with revised volume: {len(revised):,} on {by_day['session_date_new'].tolist() or 'no days'}")


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(PROCESSED / "stock_metadata.csv", dtype={"cik": str})

    study = clean_window("study")
    check_against_first_download(study)

    day = stock_day(study)
    freeze_cutoffs(day, metadata)
    day["volatility_tier"] = to_tier(day["realized_volatility"], config["volatility_cutoffs"])
    day["coverage_ok"] = day["observed_bars"] >= min_bars(config)
    day["is_calibration_day"] = day["session_date"].isin(CALIB_DAYS)
    day = day.rename(columns={"symbol": "ticker"})

    profile, _ = stock_profile(day.rename(columns={"ticker": "symbol"}), metadata, config,
                               config["liquidity_cutoffs"])
    profile = profile[PROFILE_COLUMNS]

    profile.to_csv(PROCESSED / "stock_profile.csv", index=False)
    day.to_csv(PROCESSED / "stock_day.csv", index=False)
    study.rename(columns={"symbol": "ticker"}).to_parquet(PROCESSED / "intraday_bars.parquet",
                                                          index=False)
    modelling_table(day, profile).to_csv(PROCESSED / "classification_data.csv", index=False)
    main = ~day["is_calibration_day"]
    summary["tables"] = {"stock_profile": len(profile), "stock_day_main": int(main.sum()),
                         "intraday_bars_main": int((~study["session_date"].isin(CALIB_DAYS)).sum())}

    history = clean_window("history")
    history.rename(columns={"symbol": "ticker"}).to_parquet(PROCESSED / "history_bars.parquet",
                                                            index=False)
    (PROCESSED / "build_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[tables] {summary['tables']}\nSaved everything to {PROCESSED}")


if __name__ == "__main__":
    main()
