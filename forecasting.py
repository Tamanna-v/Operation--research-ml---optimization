import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def add_time_features(df, date_col="date"):
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out["dayofweek"] = out[date_col].dt.dayofweek
    out["month"] = out[date_col].dt.month
    out["dayofyear"] = out[date_col].dt.dayofyear
    return out

def train_random_forest(df, date_col="date", target_col="demand"):
    data = add_time_features(df, date_col)
    data = data.sort_values(date_col).dropna()
    features = ["dayofweek", "month", "dayofyear"]
    X, y = data[features], data[target_col]
    split = int(len(data) * 0.8)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X.iloc[:split], y.iloc[:split])
    pred = model.predict(X.iloc[split:])
    mae = mean_absolute_error(y.iloc[split:], pred)
    rmse = mean_squared_error(y.iloc[split:], pred) ** 0.5
    return model, {"MAE": mae, "RMSE": rmse}, data.iloc[split:].assign(predicted_demand=pred)
