"""
Inventory decision experiment.

This is a transparent, reproducible one-item periodic-review simulation.
The ML forecast is converted into an order-up-to level using a service-level
safety stock. A traditional baseline uses a rolling historical mean.

Decision:
    order-up-to level S_t = forecast_demand_horizon + z * sigma_demand

The simulation tracks:
- holding cost
- shortage/stockout cost
- ordering cost
- total cost
- service level

This is an OR-style policy model with explicit decision logic and constraints.
It is intentionally simple enough to explain in a CV/interview and can later
be upgraded to stochastic programming or MILP.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FORECASTS = RESULTS / "test_forecasts.csv"

df = pd.read_csv(FORECASTS, parse_dates=["date"]).set_index("date")
history = pd.read_csv(ROOT / "data" / "processed" / "daily_demand_features.csv",
                      parse_dates=["date"]).set_index("date")

# Align demand history and test forecasts.
df["actual"] = df["actual"].astype(float)

# Cost assumptions are explicit and should be sensitivity-tested.
HOLDING_COST = 0.50       # cost per unit of end inventory per day
SHORTAGE_COST = 4.00      # penalty per unit of unmet demand
ORDER_FIXED_COST = 10.00  # fixed cost whenever an order is placed
LEAD_TIME = 1
SERVICE_LEVEL = 0.95
MAX_ORDER_UP_TO = 1500

z = norm.ppf(SERVICE_LEVEL)

def simulate(policy_forecast_col, name):
    on_hand = 0.0
    rows = []

    # demand variability estimated from training history only
    train_end = df.index.min()
    sigma = history.loc[history.index < train_end, "demand"].std()
    sigma = float(max(sigma, 1.0))

    for date, r in df.iterrows():
        forecast = float(max(r[policy_forecast_col], 0))
        # One-day lead-time demand. Safety stock uses historical volatility.
        S = min(MAX_ORDER_UP_TO, forecast * LEAD_TIME + z * sigma)

        order_qty = max(0.0, S - on_hand)
        fixed = ORDER_FIXED_COST if order_qty > 0 else 0.0
        on_hand += order_qty

        demand = float(r["actual"])
        sales = min(on_hand, demand)
        shortage = max(0.0, demand - sales)
        on_hand -= sales

        holding = HOLDING_COST * on_hand
        shortage_cost = SHORTAGE_COST * shortage
        total = fixed + holding + shortage_cost

        rows.append({
            "date": date,
            "policy": name,
            "forecast": forecast,
            "order_qty": order_qty,
            "demand": demand,
            "sales": sales,
            "ending_inventory": on_hand,
            "shortage_units": shortage,
            "ordering_cost": fixed,
            "holding_cost": holding,
            "shortage_cost": shortage_cost,
            "total_cost": total
        })

    out = pd.DataFrame(rows)
    return out

# ML policy.
ml = simulate("best_forecast", "ML+OR")

# Traditional baseline: use a lagged 28-day rolling average from the features.
baseline = df.copy()
baseline["baseline_forecast"] = (
    history["demand"].reindex(df.index).rolling(28).mean()
)
# If dates are not present in history due to feature dropping, use the
# previous 28 days from the original demand series.
raw_demand = history["demand"]
baseline["baseline_forecast"] = [
    raw_demand.loc[:d].tail(28).mean() for d in df.index
]
bl = baseline[["actual", "baseline_forecast"]].copy()
bl["best_forecast"] = bl["baseline_forecast"]
bl = bl.rename(columns={"actual": "actual"})
old = df.copy()
old["baseline_forecast"] = bl["baseline_forecast"]
base = simulate_from_frame = None

# Reuse simulation with temporary column.
df["baseline_forecast"] = baseline["baseline_forecast"]
traditional = simulate("baseline_forecast", "Traditional")

combined = pd.concat([ml, traditional], ignore_index=True)
combined.to_csv(RESULTS / "inventory_simulation.csv", index=False)

summary = combined.groupby("policy").agg(
    total_cost=("total_cost", "sum"),
    average_daily_inventory=("ending_inventory", "mean"),
    total_shortage_units=("shortage_units", "sum"),
    stockout_days=("shortage_units", lambda x: int((x > 0).sum())),
    total_order_quantity=("order_qty", "sum")
).reset_index()

summary["service_level"] = 1 - (
    summary["total_shortage_units"] /
    combined.groupby("policy")["demand"].sum().values
)
summary.to_csv(RESULTS / "inventory_summary.csv", index=False)

print(summary.to_string(index=False))
