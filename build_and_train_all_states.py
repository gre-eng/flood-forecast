import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

capitals = pd.read_csv("data/state_capitals.csv")
STATES = capitals["state"].tolist()
YEARS = range(2010, 2025)
MONTHS = range(1, 13)

rainfall = pd.read_csv("data/all_states_rainfall_2010_2024.csv")
labels = pd.read_csv("data/flood_labels_all_states.csv")
elevation = pd.read_csv("data/state_elevation_all.csv")

grid = pd.MultiIndex.from_product([STATES, YEARS, MONTHS], names=["state", "year", "month"]).to_frame(index=False)
dataset = grid.merge(rainfall, on=["state", "year", "month"], how="left")
dataset = dataset.merge(labels, on=["state", "year", "month"], how="left")
dataset["flooded"] = dataset["flooded"].fillna(0).astype(int)
dataset = dataset.merge(elevation, on="state", how="left")

print(f"Full dataset: {len(dataset)} rows, {dataset['flooded'].sum()} flooded")
print("Any missing rainfall:", dataset["rainfall_mm"].isna().any())
print("Any missing elevation:", dataset["elevation_m"].isna().any())

dataset.to_csv("data/full_dataset_all_states.csv", index=False)

state_dummies = pd.get_dummies(dataset["state"], prefix="state")
month_dummies = pd.get_dummies(dataset["month"], prefix="month")

X = pd.concat([dataset[["rainfall_mm", "elevation_m"]], state_dummies, month_dummies], axis=1)
y = dataset["flooded"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(class_weight="balanced", max_iter=2000)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

proba = cross_val_predict(model, X_scaled, y, cv=cv, method="predict_proba")[:, 1]
preds = (proba >= 0.5).astype(int)

print()
print("Baseline (always predict not-flooded):", round((y == 0).mean(), 3))
print("ROC-AUC (out-of-fold):", round(roc_auc_score(y, proba), 3))
print()
print(classification_report(y, preds, target_names=["not flooded", "flooded"]))

model.fit(X_scaled, y)

joblib.dump(model, "data/flood_model_all_states.joblib")
joblib.dump(scaler, "data/flood_model_all_states_scaler.joblib")
joblib.dump(list(X.columns), "data/flood_model_all_states_features.joblib")
print("Model saved.")