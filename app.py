from flask import Flask, jsonify, render_template
import pandas as pd
import json
from datetime import date

app = Flask(__name__)

capitals = pd.read_csv("data/state_capitals.csv")
hybrid = pd.read_csv("data/hybrid_risk_all_states.csv")

with open("data/nigeria_states.geojson") as f:
    STATE_GEOJSON = json.load(f)
GEOJSON_NAME_FIXES = {"Nassarawa": "Nasarawa"}
for feature in STATE_GEOJSON["features"]:
    raw_name = feature["properties"]["name"]
    feature["properties"]["name"] = GEOJSON_NAME_FIXES.get(raw_name, raw_name)

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
COASTAL_STATES = {"Lagos", "Bayelsa", "Delta", "Rivers", "Ondo"}

# True once the merge pipeline (build_hybrid_risk.py) keeps this column.
# Until then, percentile is simply omitted from the API response —
# the frontend already handles that gracefully.
HAS_PERCENTILE_COLUMN = "historical_percentile" in hybrid.columns

NEMA_ADVICE = {
    "high": ["Move to higher ground now. Do not wait for further instructions.",
              "Do not drive or walk through floodwater.",
              "Keep important documents and valuables in waterproof storage.",
              "Report emergencies to NEMA or your local emergency service."],
    "medium": ["Stay alert and monitor official flood advisories.",
                "Avoid driving through low-lying roads during heavy rain.",
                "Clear drains and gutters near your home if blocked."],
    # "Low" must never read as "safe": a model can miss local flooding.
    "low": ["Low risk is not zero risk. Heavy rain can still cause local flooding.",
             "Keep following official alerts from NiMet, NIHSA and NEMA.",
             "Keep drains and gutters near your home clear."],
}

# Used when the forecast file has run out and nothing current is available.
NO_FORECAST_ADVICE = [
    "Do not rely on this tool for now. Check official alerts from NiMet, NIHSA and NEMA.",
    "During heavy rain, avoid low-lying roads and floodwater.",
    "Report emergencies to NEMA or your local emergency service.",
]


@app.route("/api/states_geojson")
def states_geojson():
    return jsonify(STATE_GEOJSON)


@app.route("/api/hybrid_risk")
def hybrid_risk():
    today = date.today()
    results = []

    for state, group in hybrid.groupby("state"):
        group = group.sort_values(["year", "month"])
        monthly = []
        for _, row in group.iterrows():
            entry = {
                "year": int(row["year"]), "month": int(row["month"]),
                "month_label": MONTH_NAMES[int(row["month"]) - 1],
                "level": row["level"],
                "method": row["method"],
                "value": row["discharge_or_probability"],
            }
            if HAS_PERCENTILE_COLUMN:
                pct = row["historical_percentile"]
                entry["historical_percentile"] = float(pct) if pd.notna(pct) else None
            monthly.append(entry)

        # Pick the row that actually matches today's year/month.
        # Falls back to the nearest future month. If every forecast month
        # is already in the past, the latest row is used but flagged as
        # out of date, so the app never presents an old month as "now".
        data_is_outdated = False
        current = next(
            (m for m in monthly if m["year"] == today.year and m["month"] == today.month),
            None
        )
        if current is None:
            future = [m for m in monthly if (m["year"], m["month"]) >= (today.year, today.month)]
            if future:
                current = future[0]
            elif monthly:
                current = monthly[-1]
                data_is_outdated = True

        if current is None:
            continue

        results.append({
            "state": state,
            "current_level": current["level"],
            "current_method": current["method"],
            "current_year": current["year"],
            "current_month": current["month"],
            "is_this_month": current["year"] == today.year and current["month"] == today.month,
            "data_is_outdated": data_is_outdated,
            "is_coastal": state in COASTAL_STATES,
            "advice": NEMA_ADVICE[current["level"]],
            "monthly": monthly,
        })
    return jsonify(results)


def translate_for_community(entry):
    level = entry["current_level"]
    is_live = "GloFAS" in entry["current_method"]
    period = f"{MONTH_NAMES[entry['current_month'] - 1]} {entry['current_year']}"

    # Forecast file has run out: say so plainly instead of showing an old month.
    if entry["data_is_outdated"]:
        return {
            "state": entry["state"],
            "level": "unknown",
            "period": period,
            "headline": "No current forecast is available.",
            "basis": (f"The latest forecast we have is for {period}, which has passed. "
                      "Please check NiMet, NIHSA and NEMA for current alerts."),
            "confidence": "none",
            "confidence_label": "No current forecast",
            "advice": NO_FORECAST_ADVICE,
            "is_coastal": entry["is_coastal"],
        }

    when_text = "this month" if entry["is_this_month"] else f"in {period}"

    if is_live:
        headline = f"{level.upper()} flood risk {when_text} — based on river levels."
        basis = "This is based on live river-flow data for your area."
        confidence = "higher"
        confidence_label = "Higher confidence: based on live river data"
    else:
        headline = f"{level.upper()} seasonal flood risk {when_text}."
        basis = ("River-flow data was not reliable enough for your area, so this rating "
                 "comes from a model of past flooding and typical rainfall for this state "
                 "and month. It is a seasonal pattern, not a live forecast, so it is less certain.")
        confidence = "lower"
        confidence_label = "Lower confidence: seasonal pattern, not a live forecast"

    return {
        "state": entry["state"],
        "level": level,
        "period": period,
        "headline": headline,
        "basis": basis,
        "confidence": confidence,
        "confidence_label": confidence_label,
        "advice": NEMA_ADVICE[level],
        "is_coastal": entry["is_coastal"],
    }


@app.route("/api/community_risk")
def community_risk():
    full = hybrid_risk().get_json()
    return jsonify([translate_for_community(s) for s in full])


@app.route("/community")
def community():
    return render_template("community.html")


@app.route("/")
def index():
    return render_template("index_v11.html")


if __name__ == "__main__":
    app.run(debug=True, port=5055)