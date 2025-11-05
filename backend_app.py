from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import pandas as pd, numpy as np, joblib, os, io
from prophet import Prophet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, roc_auc_score
from xgboost import XGBClassifier

app = FastAPI(title="AI ERP Intelligence API")

# ------------------------------
#   File paths for saved models
# ------------------------------
FORECAST_MODEL_PATH = "forecast_model.joblib"
CHURN_MODEL_PATH = "churn_model.joblib"

# ============================================================
# 🏪 SALES FORECASTING MODULE
# ============================================================
@app.post("/train_forecast")
async def train_forecast(train_file: UploadFile = File(...)):
    """Train Prophet model on uploaded training CSV (columns: ds, y)."""
    try:
        df = pd.read_csv(train_file.file)
        if not {'ds','y'}.issubset(df.columns):
            return JSONResponse(content={"error": "CSV must contain columns 'ds' and 'y'."}, status_code=400)

        model = Prophet(daily_seasonality=True)
        model.fit(df)
        joblib.dump(model, FORECAST_MODEL_PATH)
        return {"message": "Forecast model trained.", "records": len(df)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/evaluate_forecast")
async def evaluate_forecast(test_file: UploadFile = File(...)):
    """Evaluate saved Prophet model on test CSV."""
    if not os.path.exists(FORECAST_MODEL_PATH):
        return {"error": "No trained forecast model found."}

    try:
        df = pd.read_csv(test_file.file)
        model = joblib.load(FORECAST_MODEL_PATH)
        forecast = model.predict(df[['ds']])
        merged = forecast[['ds','yhat']].merge(df, on='ds', how='inner')
        mae = mean_absolute_error(merged['y'], merged['yhat'])
        rmse = mean_squared_error(merged['y'], merged['yhat'], squared=False)
        return {"mae": mae, "rmse": rmse, "n_test": len(merged)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_forecast")
async def predict_forecast(days: int = Form(7)):
    """Forecast N future days."""
    if not os.path.exists(FORECAST_MODEL_PATH):
        return {"error": "No trained forecast model found."}

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
@app.post("/train_churn")
async def train_churn(train_file: UploadFile = File(...)):
    """Train churn model on uploaded dataset with a 'churn' column."""
    try:
        df = pd.read_csv(train_file.file)
        if 'churn' not in df.columns:
            return JSONResponse(content={"error": "CSV must contain a 'churn' column."}, status_code=400)

        X = df.drop(columns=['churn'])
        y = df['churn']
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X, y)
        joblib.dump(model, CHURN_MODEL_PATH)
        return {"message": "Churn model trained.", "records": len(df)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/evaluate_churn")
async def evaluate_churn(test_file: UploadFile = File(...)):
    """Evaluate churn model accuracy and AUC."""
    if not os.path.exists(CHURN_MODEL_PATH):
        return {"error": "No trained churn model found."}
    try:
        df = pd.read_csv(test_file.file)
        if 'churn' not in df.columns:
            return {"error": "Test CSV must contain 'churn' column."}

        X = df.drop(columns=['churn'])
        y = df['churn']
        model = joblib.load(CHURN_MODEL_PATH)
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]

        acc = accuracy_score(y, preds)
        auc = roc_auc_score(y, probs)
        return {"accuracy": acc, "auc": auc, "n_test": len(df)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_churn")
async def predict_churn(test_file: UploadFile = File(...)):
    """Predict churn probabilities for uploaded CSV without labels."""
    if not os.path.exists(CHURN_MODEL_PATH):
        return {"error": "No trained churn model found."}
    try:
        df = pd.read_csv(test_file.file)
        model = joblib.load(CHURN_MODEL_PATH)
        probs = model.predict_proba(df)[:, 1]
        df['churn_probability'] = probs
        return df.to_dict(orient='records')
    except Exception as e:
        return {"error": str(e)}

@app.get("/status")
def status():
    """Quick API health check."""
    return {
        "status": "ok",
        "available_models": {
            "forecast": os.path.exists(FORECAST_MODEL_PATH),
            "churn": os.path.exists(CHURN_MODEL_PATH)
        }
    }
