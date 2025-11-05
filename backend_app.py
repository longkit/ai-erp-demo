from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from prophet import Prophet
from datetime import date, timedelta

app = FastAPI(title="AI ERP Intelligence API")

# -----------------------------
# 1️⃣ SALES FORECASTING MODEL
# -----------------------------

# Generate synthetic sales data (daily sales with trend + seasonality)
np.random.seed(42)
dates = pd.date_range("2023-01-01", periods=400)
sales = 1000 + np.linspace(0, 300, 400) + 100*np.sin(np.arange(400)/15) + np.random.normal(0, 50, 400)
df_sales = pd.DataFrame({"ds": dates, "y": sales})

# Train Prophet model
forecast_model = Prophet(daily_seasonality=True)
forecast_model.fit(df_sales)

@app.get("/forecast")
def forecast(days: int = 14):
    """Predict next N days of sales"""
    future = forecast_model.make_future_dataframe(periods=days)
    forecast = forecast_model.predict(future)[["ds", "yhat"]].tail(days)
    forecast.rename(columns={"ds": "date", "yhat": "sales"}, inplace=True)
    return forecast.to_dict(orient="records")

# -----------------------------
# 2️⃣ CHURN PREDICTION MODEL
# -----------------------------

# Create synthetic customer dataset
n_customers = 500
X = np.random.normal(size=(n_customers, 4))
y = (X[:, 0] + 0.5*X[:, 1] - X[:, 2] + np.random.randn(n_customers)) > 0.2
y = y.astype(int)

clf = LogisticRegression(max_iter=500)
clf.fit(X, y)

@app.get("/churn")
def churn(n_customers: int = 200):
    """Simulate churn predictions for N customers"""
    Xn = np.random.normal(size=(n_customers, 4))
    probs = clf.predict_proba(Xn)[:, 1]
    churn_rate = float(probs.mean())
    high_risk = int((probs > 0.6).sum())
    return {"churn_rate": churn_rate, "high_risk_customers": high_risk}

@app.get("/status")
def status():
    return {"status": "ok", "models": ["Prophet forecast", "Logistic Regression churn"]}
