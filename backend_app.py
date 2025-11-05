from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
from sklearn.linear_model import LogisticRegression
from datetime import date, timedelta

app = FastAPI(title="AI ERP Intelligence API")

_rng = np.random.default_rng(42)
X = _rng.normal(size=(500, 3))
y = (_rng.random(500) > 0.7).astype(int)
_clf = LogisticRegression(max_iter=300).fit(X, y)

class ForecastPoint(BaseModel):
    date: str
    sales: float

@app.get("/status")
def status():
    return {"status": "ok"}

@app.get("/forecast", response_model=List[ForecastPoint])
def forecast(days: int = 7):
    today = date.today()
    data = []
    for i in range(days):
        data.append({
            "date": (today + timedelta(days=i+1)).isoformat(),
            "sales": float(1000 + 8*i + _rng.normal(0, 40))
        })
    return data

@app.get("/churn")
def churn(n_customers: int = 300):
    Xn = _rng.normal(size=(n_customers, 3))
    probs = _clf.predict_proba(Xn)[:,1]
    return {
        "churn_rate": float(probs.mean()),
        "high_risk_customers": int((probs > 0.6).sum())
    }
