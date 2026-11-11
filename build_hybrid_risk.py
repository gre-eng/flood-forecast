"""
build_hybrid_risk.py

Combines your two already-validated systems into one complete, reliable
risk table covering all 37 states -- no state is left unassessed.

For each state/month in your GloFAS forecast:
  - If forecast discharge is above the reliable threshold (derived from
    your own real data: the natural gap found between 3.44 and 26.49
    m3/s), use the GloFAS-based percentile risk directly. This is your
    genuine forward-looking signal, and it's trustworthy here because
    there's real, substantial river flow behind it.
  - Otherwise, fall back to your trained classifier, using that state's
    real climatological rainfall for the month. This doesn't pretend to
    be a live forecast for these states -- it's a historical-pattern
    model, not a forward-looking one. Its accuracy on years it has never
    seen is about 0.74 ROC-AUC (not 0.86), because the flood labels are
    uneven across months. Treat these rows as low confidence.

Only the months that are present in the GloFAS forecast file are
produced. No months are added beyond the forecast window.

Every output row is tagged with which method produced it, and where
possible, with the historical_percentile that determined the level --
so every row is independently checkable without re-running the pipeline.

Requires (all already in your data/ folder from earlier steps):
  - flood_model_all_states.joblib, flood_model_all_states_scaler.joblib,
    flood_model_all_states_features.joblib
  - all_states_rainfall_2010_2024.csv
  - state_elevation_all.csv
  - monthly_flood_forecast_all_states.csv
"""

import pandas as pd
import joblib

# The reliable-discharge threshold, derived directly from the real gap
# found in your own forecast data (midpoint between 3.44 and 26.49 m3/s)
RELIABLE_DISCHARGE_THRESHOLD = 14.96

model = joblib.load("data/flood_model_all_states.joblib")
scaler = joblib.load("data/flood_model_all_states_scaler.joblib")
feature_columns = joblib.load("data/flood_model_all_states_features.joblib")

rainfall = pd.read_csv("data/all_states_rainfall_2010_2024.csv")
elevation = pd.read_csv("data/state_elevation_all.csv")
glofas_forecast = pd.read_csv("data/monthly_flood_forecast_all_states.csv")

avg_rainfall_by_state_month = (
    rainfall.groupby(["state", "month"])["rainfall_mm"].mean().to_dict()
)
elevation_by_state = dict(zip(elevation["state"], elevation["elevation_m"]))


def classifier_risk(state, month):
    """Falls back to the trained classifier using climatological rainfall."""
    row = {col: 0 for col in feature_columns}
    row["rainfall_mm"] = avg_rainfall_by_state_month.get((state, month), 5.0)
    row["elevation_m"] = elevation_by_state.get(state, 100.0)
    state_col = f"state_{state}"
    if state_col in row:
        row[state_col] = 1
    month_col = f"month_{month}"
    if month_col in row:
        row[month_col] = 1
    X_new = pd.DataFrame([row])[feature_columns]
    X_scaled = scaler.transform(X_new)
    proba = model.predict_proba(X_scaled)[0][1]
    level = "high" if proba >= 0.6 else "medium" if proba >= 0.3 else "low"
    return proba, level


results = []
for _, row in glofas_forecast.iterrows():
    state, month = row["state"], row["month"]
    discharge = row["forecast_discharge"]

    if discharge >= RELIABLE_DISCHARGE_THRESHOLD:
        results.append({
            "state": state, "year": row["year"], "month": month,
            "level": row["level"],
            "method": "GloFAS forecast (river discharge)",
            "discharge_or_probability": discharge,
            # This was already being computed upstream in
            # forecast_monthly_flood_risk_all_states.py but was previously
            # dropped here instead of being carried through. Restoring it
            # is what makes every discharge-based row independently
            # checkable in the app's evidence table.
            "historical_percentile": row.get("historical_percentile"),
        })
    else:
        proba, level = classifier_risk(state, month)
        results.append({
            "state": state, "year": row["year"], "month": month,
            "level": level,
            "method": "Historical model (climatological rainfall)",
            "discharge_or_probability": round(proba, 3),
            # The rainfall-fallback classifier is a trained probability
            # model, not a percentile-vs-history comparison, so there is
            # no equivalent percentile to report for these rows.
            "historical_percentile": None,
        })

result_df = pd.DataFrame(results)
result_df.to_csv("data/hybrid_risk_all_states.csv", index=False)

print(f"Total entries: {len(result_df)}")
print(f"Using GloFAS forecast: {(result_df['method'] == 'GloFAS forecast (river discharge)').sum()}")
print(f"Using historical model fallback: {(result_df['method'] == 'Historical model (climatological rainfall)').sum()}")
print()
print(result_df.to_string(index=False))