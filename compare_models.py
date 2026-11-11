"""
compare_models.py

Read-only check. It does NOT save or overwrite any model or data file.

It answers three questions:
  1. In which months were floods actually recorded in your labels?
  2. How well does the model predict a year it has never seen,
     with and without the "month" feature?
  3. What month-by-month risk curve does each version give for
     sample states (using normal rainfall, exactly like the app does)?

Run from the project folder:
    python compare_models.py
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

data = pd.read_csv("data/full_dataset_all_states.csv")
y = data["flooded"]
groups = data["year"]

# ---------------------------------------------------------------
# 1. Where do the recorded floods sit in the calendar?
# ---------------------------------------------------------------
print("=== 1. Recorded floods by month ===")
print("(sum = flooded state-months, mean = share of state-months flooded)")
print(data.groupby("month")["flooded"].agg(["sum", "mean"]).round(3))
print()

# ---------------------------------------------------------------
# Feature sets: with and without the month dummies
# ---------------------------------------------------------------
state_dummies = pd.get_dummies(data["state"], prefix="state", dtype=float)
month_dummies = pd.get_dummies(data["month"], prefix="month", dtype=float)


def build_features(use_month):
    parts = [data[["rainfall_mm", "elevation_m"]], state_dummies]
    if use_month:
        parts.append(month_dummies)
    return pd.concat(parts, axis=1)


def make_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000),
    )


# ---------------------------------------------------------------
# 2. Honest validation: hold out whole YEARS
# ---------------------------------------------------------------
print("=== 2. ROC-AUC when whole years are held out ===")
print("(1.0 = perfect, 0.5 = no better than guessing)")
cv = GroupKFold(n_splits=5)
fitted = {}
for name, use_month in [("WITH month feature   ", True), ("WITHOUT month feature", False)]:
    X = build_features(use_month)
    proba = cross_val_predict(
        make_model(), X, y, cv=cv, groups=groups, method="predict_proba"
    )[:, 1]
    print(f"{name}: {roc_auc_score(y, proba):.3f}")
    pipe = make_model().fit(X, y)
    fitted[use_month] = (pipe, list(X.columns))
print()

# ---------------------------------------------------------------
# 3. Month-by-month curve using normal (climatological) rainfall
# ---------------------------------------------------------------
clim = data.groupby(["state", "month"])["rainfall_mm"].mean()
elev = data.groupby("state")["elevation_m"].first()


def month_curve(use_month, state):
    pipe, columns = fitted[use_month]
    rows = []
    for m in range(1, 13):
        r = {c: 0.0 for c in columns}
        r["rainfall_mm"] = clim[(state, m)]
        r["elevation_m"] = elev[state]
        r[f"state_{state}"] = 1.0
        if use_month:
            r[f"month_{m}"] = 1.0
        rows.append(r)
    X_new = pd.DataFrame(rows)[columns]
    return pipe.predict_proba(X_new)[:, 1]


print("=== 3. Risk curve through the year (model score, not a true chance) ===")
for state in ["Abia", "Kano", "Lagos"]:
    if state not in elev.index:
        continue
    curve = pd.DataFrame({
        "month": range(1, 13),
        "rain_mm": [round(clim[(state, m)], 1) for m in range(1, 13)],
        "with_month": month_curve(True, state).round(3),
        "without_month": month_curve(False, state).round(3),
    }).set_index("month")
    print(f"\n{state}")
    print(curve.to_string())