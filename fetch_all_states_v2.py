import pandas as pd
import requests
import time

capitals = pd.read_csv("data/state_capitals.csv")

START_YEAR = 2010
END_YEAR = 2024

all_rows = []

for _, row in capitals.iterrows():
    state = row["state"]
    lat = row["lat"]
    lon = row["lon"]
    print(f"Fetching {state}...")

    url = (
        "https://power.larc.nasa.gov/api/temporal/monthly/point"
        f"?parameters=PRECTOTCORR&community=AG"
        f"&longitude={lon}&latitude={lat}"
        f"&start={START_YEAR}&end={END_YEAR}&format=JSON"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    monthly_values = data["properties"]["parameter"]["PRECTOTCORR"]

    for year_month, rainfall in monthly_values.items():
        month = year_month[4:6]
        if month == "13":
            continue
        year = year_month[0:4]
        all_rows.append({
            "state": state,
            "year": int(year),
            "month": int(month),
            "rainfall_mm": rainfall,
        })

    time.sleep(1)

df = pd.DataFrame(all_rows)
df.to_csv("data/all_states_rainfall_2010_2024.csv", index=False)
print(f"Saved {len(df)} rows to data/all_states_rainfall_2010_2024.csv")
print(df["state"].nunique(), "unique states,", len(df), "total rows")