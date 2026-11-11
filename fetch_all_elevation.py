import pandas as pd
import requests

capitals = pd.read_csv("data/state_capitals.csv")

lats = ",".join(str(x) for x in capitals["lat"])
lons = ",".join(str(x) for x in capitals["lon"])

url = f"https://api.open-meteo.com/v1/elevation?latitude={lats}&longitude={lons}"

response = requests.get(url, timeout=30)
response.raise_for_status()
data = response.json()

print("Number of elevation values returned:", len(data["elevation"]))
print("Number of states requested:", len(capitals))

elevations = dict(zip(capitals["state"], data["elevation"]))

elev_df = pd.DataFrame(list(elevations.items()), columns=["state", "elevation_m"])
elev_df.to_csv("data/state_elevation_all.csv", index=False)

print()
print(elev_df.sort_values("elevation_m").to_string(index=False))