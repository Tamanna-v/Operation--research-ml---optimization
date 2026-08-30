"""
Time-series forecasting experiment.

Uses chronological train/test splitting to avoid leakage. Compares:
- naive lag-7 baseline
- Linear Regression
- Random Forest
- HistGradientBoostingRegressor

The best model is selected by test MAE and its forecasts are saved for the
inventory optimization stage.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "daily_demand_features.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

df = pd.read_csv(DATA, parse_dates=["date"]).set_index("date")
target = "demand"
features = [c for c in df.columns if c != target]

# Chronological split: final 20% is the test period.
split = int(len(df) * 0.80)
train, test = df.iloc[:split].copy(), df.iloc[split:].copy()
X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

models = {
    "LinearRegression": Pipeline([
        ("scale", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "RandomForest": RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1,
        min_samples_leaf=2
    ),
    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
        random_state=42
    )
}

rows = []
preds = pd.DataFrame(index=test.index)
preds["actual"] = y_test.values
preds["NaiveLag7"] = test["lag_7"].values

for name, model in models.items():
    model.fit(X_train, y_train)
    p = np.maximum(model.predict(X_test), 0)
    preds[name] = p
    rows.append({
        "model": name,
        "MAE": mean_absolute_error(y_test, p),
        "RMSE": mean_squared_error(y_test, p) ** 0.5
    })
    joblib.dump(model, RESULTS / f"{name}.joblib")

rows.append({
    "model": "NaiveLag7",
    "MAE": mean_absolute_error(y_test, preds["NaiveLag7"]),
    "RMSE": mean_squared_error(y_test, preds["NaiveLag7"]) ** 0.5
})

metrics = pd.DataFrame(rows).sort_values("MAE")
metrics.to_csv(RESULTS / "forecast_metrics.csv", index=False)

best = metrics.iloc[0]["model"]
preds["best_forecast"] = preds[best]
preds.to_csv(RESULTS / "test_forecasts.csv", index_label="date")

with open(RESULTS / "best_model.json", "w") as f:
    json.dump({"best_model_by_test_MAE": best}, f, indent=2)

plt.figure(figsize=(12, 5))
n = min(120, len(preds))
plt.plot(preds.index[-n:], preds["actual"].tail(n), label="Actual")
plt.plot(preds.index[-n:], preds["best_forecast"].tail(n), label=f"Forecast ({best})")
plt.title("Demand Forecast: Holdout Period")
plt.xlabel("Date")
plt.ylabel("Units")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS / "forecast_vs_actual.png", dpi=180)
plt.close()

print(metrics.to_string(index=False))
print("Best model:", best)
