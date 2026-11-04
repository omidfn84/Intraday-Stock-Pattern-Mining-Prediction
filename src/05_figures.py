"""Step 5: report figures.

  figures/fig1_exchange_bar.png      stocks per listing exchange (bar chart)
  figures/fig2_aapl_price_line.png   AAPL 5-minute closes on Sep 18, 2026 (line chart)
"""
import matplotlib
matplotlib.use("Agg")                          # save files without opening windows
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from common import FIGURES, PROCESSED


def main():
    FIGURES.mkdir(exist_ok=True)

    profile = pd.read_csv(PROCESSED / "stock_profile.csv", dtype={"cik": str})
    counts = profile["exchange"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.5, str(v), ha="center")
    ax.set_title(f"Number of stocks by listing exchange (N = {len(profile)})")
    ax.set_xlabel("Exchange")
    ax.set_ylabel("Number of stocks")
    fig.savefig(FIGURES / "fig1_exchange_bar.png", dpi=200, bbox_inches="tight")

    bars = pd.read_parquet(PROCESSED / "intraday_bars.parquet")
    one = bars[(bars["ticker"] == "AAPL") & (bars["session_date"] == "2026-09-18")]
    ny_time = one["bar_start_utc"].dt.tz_convert("America/New_York")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ny_time, one["close"])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ny_time.dt.tz))
    ax.set_title("AAPL 5-minute closing price, Sept 18, 2026 (SIP feed)")
    ax.set_xlabel("Time (New York, ET)")
    ax.set_ylabel("Close price (USD per share)")
    fig.savefig(FIGURES / "fig2_aapl_price_line.png", dpi=200, bbox_inches="tight")
    print("Saved 2 figures to", FIGURES)


if __name__ == "__main__":
    main()
