from fastapi import FastAPI, UploadFile, File
import pandas as pd
import numpy as np
import joblib
from prophet import Prophet
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

app = FastAPI(title="AI ERP Intelligence API")

# -------------------- FORECAST --------------------
@app.post("/train_forecast")
async def train_forecast(train_file: UploadFile = File(...)):
    try:
        df = pd.read_csv(train_file.file)
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df["y"] = df["y"].astype(float)
        model = Prophet()
        model.fit(df)
        joblib.dump(model, "forecast_model.joblib")
        return {"status": "Forecast model trained successfully"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/evaluate_forecast")
async def evaluate_forecast(test_file: UploadFile = File(...)):
    try:
        model = joblib.load("forecast_model.joblib")
        df = pd.read_csv(test_file.file)
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df["y"] = df["y"].astype(float)
        forecast = model.predict(df[["ds"]])
        forecast["ds"] = pd.to_datetime(forecast["ds"])
        merged = pd.merge(forecast[["ds", "yhat"]], df, on="ds", how="inner")
        mae = np.mean(np.abs(merged["yhat"] - merged["y"]))
        rmse = np.sqrt(np.mean((merged["yhat"] - merged["y"]) ** 2))
        return {"✅ Evaluation Results": {"MAE": mae, "RMSE": rmse, "n_test": len(merged)}}
    except Exception as e:
        return {"error": str(e)}

# -------------------- CHURN --------------------
@app.post("/train_churn")
async def train_churn(train_file: UploadFile = File(...)):
    try:
        df = pd.read_csv(train_file.file)
        X = df.drop("churn", axis=1)
        y = df["churn"]
        model = XGBClassifier()
        model.fit(X, y)
        joblib.dump(model, "churn_model.joblib")
        return {"status": "Churn model trained successfully"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/evaluate_churn")
async def evaluate_churn(test_file: UploadFile = File(...)):
    try:
        model = joblib.load("churn_model.joblib")
        df = pd.read_csv(test_file.file)
        X = df.drop("churn", axis=1)
        y = df["churn"]
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        auc = roc_auc_score(y, preds)
        return {"✅ Evaluation Results": {"Accuracy": acc, "AUC": auc, "n_test": len(y)}}
    except Exception as e:
        return {"error": str(e)}
