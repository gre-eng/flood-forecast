import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta

capitals = pd.read_csv("data/state_capitals.csv")
HIST_START = "2015-01-01"


def fetch_discharge(lat, lon):
    end_date = (date.today() + timedelta(days=200)).isoformat()
    url = (
        "https://flood-api.open-meteo.com/v1/flood"
        f"?latitude={lat}&longitude={lon}&daily=river_discharge"
        f"&start_date={HIST_START}&end_date={end_date}"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame({
        "date": pd.to_datetime(data["daily"]["time"]),
        "discharge": data["daily"]["river_discharge"],
    })
    return df


def classify_monthly_risk(df):
    today = pd.Timestamp(date.today())
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    monthly = df.groupby(["year", "month"])["discharge"].mean().reset_index()
    historical = monthly[monthly["year"] < today.year]
    future = monthly[
        (monthly["year"] > today.year) |
        ((monthly["year"] == today.year) & (monthly["month"] >= today.month))
    ]
    results = []
    for _, row in future.iterrows():
        hist_for_month = historical[historical["month"] == row["month"]]["discharge"]
        if len(hist_for_month) < 3:
            continue
        percentile = (hist_for_month < row["discharge"]).mean()
        if percentile >= 0.90:
            level = "high"
        elif percentile >= 0.70:
            level = "medium"
        else:
            level = "low"
        results.append({
            "year": int(row["year"]), "month": int(row["month"]),
            "forecast_discharge": round(row["discharge"], 2),
            "historical_percentile": round(percentile, 3),
            "level": level,
        })
    return results


if __name__ == "__main__":
    all_rows = []
    for _, row in capitals.iterrows():
        state = row["state"]
        print(f"Fetching {state}...")
        try:
            df = fetch_discharge(row["lat"], row["lon"])
            results = classify_monthly_risk(df)
            for r in results:
                all_rows.append({"state": state, **r})
        except Exception as e:
            print(f"  FAILED for {state}: {e}")

    result_df = pd.DataFrame(all_rows)
    result_df.to_csv("data/monthly_flood_forecast_all_states.csv", index=False)
    print(f"\nSaved {len(result_df)} state-month forecast rows")
    print(f"\nStates/months flagged medium or high risk in the upcoming months:")
    print(result_df[result_df["level"] != "low"].sort_values(["year", "month", "state"]).to_string(index=False))