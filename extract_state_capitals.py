import pandas as pd

df = pd.read_csv("data/ng.csv")

# 'admin' = state capital, 'primary' = national capital (Abuja) — we want both
capitals = df[df["capital"].isin(["admin", "primary"])].copy()

capitals = capitals[["admin_name", "city", "lat", "lng"]].rename(
    columns={"admin_name": "state", "city": "capital_city", "lng": "lon"}
)

print(f"Found {len(capitals)} state capitals")
print(capitals.sort_values("state").to_string(index=False))

capitals.to_csv("data/state_capitals.csv", index=False)
print("\nSaved to data/state_capitals.csv")