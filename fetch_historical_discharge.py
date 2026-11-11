"""
fetch_historical_discharge.py

Fetches raw historical river discharge for every state and saves it
permanently to disk. Resumable: states already saved in the output file
are skipped on re-run, so a rate-limit failure partway through doesn't
cost you the states that already succeeded.
"""

import pandas as pd
import requests
import time
from datetime import date

HIST_START = "2015-01-01"
CAPITALS_PATH = "data/state_capitals.csv"
DAILY_OUT_PATH = "data/historical_discharge_daily.csv"
MONTHLY_OUT_PATH = "data/historical_discharge_monthly.csv"

SLEEP_BETWEEN_REQUESTS = 3.0   # seconds; increase if you still hit 429s
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 10.0      # seconds; doubles each retry


def fetch_historical_discharge(lat, lon):
    end_date = date.today().isoformat()
    url = (
        "https://flood-api.open-meteo.com/v1/flood"
        f"?latitude={lat}&longitude={lon}&daily=river_discharge"
        f"&start_date={HIST_START}&end_date={end_date}"
    )
    for attempt in range(MAX_RETRIES):
        response = requests.get(url, timeout=30)
        if response.status_code == 429:
            wait = RETRY_BACKOFF_BASE * (2 ** attempt)
            print(f"  Rate limited. Waiting {wait:.0f}s before retry "
                  f"({attempt + 1}/{MAX_RETRIES})...")
            time.sleep(wait)
            continue
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame({
            "date": pd.to_datetime(data["daily"]["time"]),
            "discharge": data["daily"]["river_discharge"],
        })
    raise RuntimeError(f"Still rate limited after {MAX_RETRIES} retries")


def load_existing(path, key_col="state"):
    try:
        df = pd.read_csv(path)
        return df, set(df[key_col].unique())
    except FileNotFoundError:
        return None, set()


def main():
    capitals = pd.read_csv(CAPITALS_PATH)

    existing_daily, done_states = load_existing(DAILY_OUT_PATH)
    existing_monthly, _ = load_existing(MONTHLY_OUT_PATH)

    all_daily = [existing_daily] if existing_daily is not None else []
    all_monthly = [existing_monthly] if existing_monthly is not None else []

    if done_states:
        print(f"Resuming — {len(done_states)} states already saved, skipping them.")

    for _, row in capitals.iterrows():
        state = row["state"]
        if state in done_states:
            continue

        print(f"Fetching historical discharge for {state}...")
        try:
            df = fetch_historical_discharge(row["lat"], row["lon"])
        except Exception as e:
            print(f"  FAILED for {state}: {e}")
            continue

        df["state"] = state
        all_daily.append(df[["state", "date", "discharge"]])

        monthly = df.copy()
        monthly["year"] = monthly["date"].dt.year
        monthly["month"] = monthly["date"].dt.month
        monthly = (
            monthly.groupby(["year", "month"])["discharge"]
            .mean()
            .reset_index()
        )
        monthly["state"] = state
        all_monthly.append(monthly[["state", "year", "month", "discharge"]])

        # Save progress after every state so a later failure never loses work.
        pd.concat(all_daily, ignore_index=True).to_csv(DAILY_OUT_PATH, index=False)
        pd.concat(all_monthly, ignore_index=True).to_csv(MONTHLY_OUT_PATH, index=False)

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    if not all_daily:
        print("No data fetched — nothing written.")
        return

    daily_df = pd.concat(all_daily, ignore_index=True)
    monthly_df = pd.concat(all_monthly, ignore_index=True)
    print(f"\nDone. {len(daily_df)} daily rows, {len(monthly_df)} monthly rows "
          f"across {daily_df['state'].nunique()} states.")


if __name__ == "__main__":
    main()