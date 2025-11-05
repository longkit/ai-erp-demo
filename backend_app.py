from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd, numpy as np, joblib, os
from prophet import Prophet
from xgboost import XGBClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, roc_auc_score
)

app = FastAPI(title="AI ERP Intelligence API 🚀",
              description="Upload your train/test data and get ML-based forecasting & churn predictions.",
              version="2.0")

# Allow API access from any origin (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Paths for saved models
FORECAST_MODEL_PATH = "forecast_model.joblib"
CHURN_MODEL_PATH = "churn_model.joblib"

# ============================================================
# 🏪 SALES FORECASTING MODULE
# ============================================================
@app.post("/train_forecast", tags=["Sales Forecasting"])
async def train_forecast(train_file: UploadFile = File(...)):
    """Train Prophet model on uploaded training CSV (columns: ds, y)."""
    try:
        df = pd.read_csv(train_file.file)
        if not {'ds','y'}.issubset(df.columns):
            return JSONResponse(content={"error": "CSV must contain columns 'ds' and 'y'."}, status_code=400)
        model = Prophet(daily_seasonality=True)
        model.fit(df)
        joblib.dump(model, FORECAST_MODEL_PATH)
        return {"message": "✅ Forecast model trained successfully.", "records": len(df)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/evaluate_forecast")
async def evaluate_forecast(test_file: UploadFile = File(...)):
    import pandas as pd
    import numpy as np
    import joblib

    try:
        model = joblib.load("forecast_model.joblib")
    except Exception as e:
        return {"error": f"No trained model found or failed to load: {e}"}

    try:
        df = pd.read_csv(test_file.file)
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = df['y'].astype(float)
    except Exception as e:
        return {"error": f"Failed to read or parse CSV: {e}"}

    try:
        forecast = model.predict(df[['ds']])
        forecast['ds'] = pd.to_datetime(forecast['ds'])

        merged = pd.merge(forecast[['ds', 'yhat']], df, on='ds', how='inner')
        mae = np.mean(np.abs(merged['yhat'] - merged['y']))
        rmse = np.sqrt(np.mean((merged['yhat'] - merged['y'])**2))

        return {"✅ Evaluation Results": {"MAE": mae, "RMSE": rmse, "n_test": len(merged)}}

    except Exception as e:
        return {"error": f"Evaluation failed: {e}"}


@app.post("/predict_forecast", tags=["Sales Forecasting"])
async def predict_forecast(days: int = Form(7)):
    """Predict N future days using trained Prophet model."""
    if not os.path.exists(FORECAST_MODEL_PATH):
        return {"error": "❌ No trained forecast model found."}
    try:
        model = joblib.load(FORECAST_MODEL_PATH)
        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)[['ds','yhat']].tail(days)
        return forecast.to_dict(orient='records')
    except Exception as e:
        return {"error": str(e)}

# ============================================================
# 👥 CHURN PREDICTION MODULE
# ============================================================
@app.post("/train_churn", tags=["Customer Churn"])
async def train_churn(train_file: UploadFile = File(...)):
    """Train churn model on CSV with a 'churn' column (0/1)."""
    try:
        df = pd.read_csv(train_file.file)
        if 'churn' not in df.columns:
            return JSONResponse(content={"error": "CSV must contain a 'churn' column."}, status_code=400)
        X = df.drop(columns=['churn'])
        y = df['churn']
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X, y)
        joblib.dump(model, CHURN_MODEL_PATH)
        return {"message": "✅ Churn model trained successfully.", "records": len(df)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/evaluate_churn", tags=["Customer Churn"])
async def evaluate_churn(test_file: UploadFile = File(...)):
    """Evaluate churn model on test CSV with 'churn' column."""
    if not os.path.exists(CHURN_MODEL_PATH):
        return {"error": "❌ No trained churn model found."}
    try:
        df = pd.read_csv(test_file.file)
        if 'churn' not in df.columns:
            return {"error": "CSV must contain 'churn' column."}
        X = df.drop(columns=['churn'])
        y = df['churn']
        model = joblib.load(CHURN_MODEL_PATH)
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
        acc = accuracy_score(y, preds)
        auc = roc_auc_score(y, probs)
        return {"✅ Evaluation Results": {"Accuracy": acc, "AUC": auc, "n_test": len(df)}}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_churn", tags=["Customer Churn"])
async def predict_churn(test_file: UploadFile = File(...)):
    """Predict churn probability for unlabeled customers."""
    if not os.path.exists(CHURN_MODEL_PATH):
        return {"error": "❌ No trained churn model found."}
    try:
        df = pd.read_csv(test_file.file)
        model = joblib.load(CHURN_MODEL_PATH)
        probs = model.predict_proba(df)[:, 1]
        df['churn_probability'] = probs
        return df.to_dict(orient='records')
    except Exception as e:
        return {"error": str(e)}

# ============================================================
# 🔍 HEALTH CHECK
# ============================================================
@app.get("/status", tags=["System"])
def status():
    """Check API health and model availability."""
    return {
        "status": "✅ OK",
        "models": {
            "forecast": os.path.exists(FORECAST_MODEL_PATH),
            "churn": os.path.exists(CHURN_MODEL_PATH)
        }
    }
