"""Step 2: official company name, listing exchange and CIK from SEC EDGAR.

Saves the untouched SEC file in data/raw/sec/ and our 59 stocks in
data/processed/stock_metadata.csv. Skipped if that file exists, so the snapshot
date stays fixed; pass --refresh to download a new snapshot.
"""
import json
import sys
from datetime import date

import pandas as pd
import requests

from common import PROCESSED, RAW_SEC, load_config

URL = "https://www.sec.gov/files/company_tickers_exchange.json"
HEADERS = {"User-Agent": "Omid Firouzinia omidfiruzi0084@gmail.com"}   # SEC requires a contact
OUT = PROCESSED / "stock_metadata.csv"


def main():
    if OUT.exists() and "--refresh" not in sys.argv:
        print(f"{OUT.name} already exists (snapshot kept). Use --refresh to replace it.")
        return
    resp = requests.get(URL, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()
    RAW_SEC.mkdir(parents=True, exist_ok=True)
    (RAW_SEC / f"sec_company_tickers_exchange_{date.today()}.json").write_text(json.dumps(data))

    sec = pd.DataFrame(data["data"], columns=data["fields"])
    tickers = load_config()["tickers"]
    ours = sec[sec["ticker"].isin(tickers)].copy()
    missing = set(tickers) - set(ours["ticker"])
    if missing:
        sys.exit(f"Not found in SEC file: {missing}")
    ours["cik"] = ours["cik"].astype(str).str.zfill(10)        # official 10-digit text form
    ours["snapshot_date"] = date.today().isoformat()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    ours.to_csv(OUT, index=False)
    print(len(ours), "stocks saved;", ours["exchange"].value_counts().to_dict())


if __name__ == "__main__":
    main()
