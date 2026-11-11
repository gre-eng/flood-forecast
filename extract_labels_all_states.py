import pandas as pd
import json
import ast

emdat = pd.read_excel("data/public_emdat_custom_request_2026-08-12_81f66211-19af-4a0f-95ad-df2daed242c9.xlsx")
capitals = pd.read_csv("data/state_capitals.csv")

STATES = capitals["state"].tolist()

# EM-DAT uses different names for these two — map EM-DAT's name -> our name
NAME_FIXES = {
    "Abuja": "Federal Capital Territory",
    "Nassarawa": "Nasarawa",
}

flood_labels = []

for _, row in emdat.iterrows():
    admin_units = row["Admin Units"]
    year = row["Start Year"]
    month = row["Start Month"]

    if pd.isna(admin_units) or pd.isna(month):
        continue

    try:
        units = json.loads(admin_units)
    except (json.JSONDecodeError, TypeError):
        units = ast.literal_eval(admin_units)

    states_hit = set()
    for u in units:
        if "adm1_name" in u:
            name = u["adm1_name"]
            name = NAME_FIXES.get(name, name)  # apply correction if needed
            states_hit.add(name)

    for state in STATES:
        if state in states_hit:
            flood_labels.append({"state": state, "year": int(year), "month": int(month), "flooded": 1})

labels_df = pd.DataFrame(flood_labels).drop_duplicates()
labels_df.to_csv("data/flood_labels_all_states.csv", index=False)

print(f"Total state-month flood labels: {len(labels_df)}")
print(f"States with at least one flood label: {labels_df['state'].nunique()} out of {len(STATES)}")
print()
print(labels_df.groupby("state").size().sort_values(ascending=False))