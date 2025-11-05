import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import io

# ========================
# CONFIG
# ========================
BACKEND_URL = "https://your-backend.onrender.com"  # 👈 replace this

st.set_page_config(page_title="AI ERP ML Dashboard", layout="wide")
st.title("🤖 AI ERP ML Dashboard")

st.sidebar.title("⚙️ Settings")
module = st.sidebar.radio("Select Module", ["Sales Forecasting", "Customer Churn"])

st.markdown("---")

# ========================
# FORECASTING SECTION
# ========================
if module == "Sales Forecasting":
    st.header("🏪 Sales Forecasting")

    train_file = st.file_uploader("Upload Training CSV (columns: ds, y)", type=["csv"], key="train_forecast")
    test_file = st.file_uploader("Upload Test CSV (columns: ds, y)", type=["csv"], key="test_forecast")

    col1, col2, col3 = st.columns(3)

    if col1.button("🚀 Train Model"):
        if train_file:
            res = requests.post(f"{BACKEND_URL}/train_forecast", files={"train_file": train_file.getvalue()})
            st.success(res.json())
        else:
            st.error("Please upload a training CSV first.")

    if col2.button("📊 Evaluate Model"):
        if test_file:
            res = requests.post(f"{BACKEND_URL}/evaluate_forecast", files={"test_file": test_file.getvalue()})
            result = res.json()
            st.json(result)
            if "✅ Evaluation Results" in result:
                metrics = result["✅ Evaluation Results"]
                st.metric("MAE", f"{metrics['MAE']:.2f}")
                st.metric("RMSE", f"{metrics['RMSE']:.2f}")
        else:
            st.error("Please upload a test CSV first.")

    if col3.button("🔮 Predict Next 14 Days"):
        res = requests.post(f"{BACKEND_URL}/predict_forecast", data={"days": 14})
        df_pred = pd.DataFrame(res.json())
        if not df_pred.empty:
            st.subheader("📈 Forecast Plot")
            fig = px.line(df_pred, x="ds", y="yhat", markers=True, title="14-Day Forecast")
            st.plotly_chart(fig, use_container_width=True)

# ========================
# CHURN SECTION
# ========================
elif module == "Customer Churn":
    st.header("👥 Customer Churn Prediction")

    train_file = st.file_uploader("Upload Training CSV (features + churn)", type=["csv"], key="train_churn")
    test_file = st.file_uploader("Upload Test CSV (features + churn)", type=["csv"], key="test_churn")
    predict_file = st.file_uploader("Upload Prediction CSV (features only)", type=["csv"], key="predict_churn")

    col1, col2, col3 = st.columns(3)

    if col1.button("🚀 Train Model"):
        if train_file:
            res = requests.post(f"{BACKEND_URL}/train_churn", files={"train_file": train_file.getvalue()})
            st.success(res.json())
        else:
            st.error("Please upload a training CSV.")

    if col2.button("📊 Evaluate Model"):
        if test_file:
            res = requests.post(f"{BACKEND_URL}/evaluate_churn", files={"test_file": test_file.getvalue()})
            result = res.json()
            st.json(result)
            if "✅ Evaluation Results" in result:
                metrics = result["✅ Evaluation Results"]
                st.metric("Accuracy", f"{metrics['Accuracy']*100:.2f}%")
                st.metric("AUC", f"{metrics['AUC']:.3f}")
        else:
            st.error("Please upload a test CSV.")

    if col3.button("🔮 Predict Churn"):
        if predict_file:
            res = requests.post(f"{BACKEND_URL}/predict_churn", files={"test_file": predict_file.getvalue()})
            df = pd.DataFrame(res.json())
            st.subheader("Predicted Churn Probabilities")
            st.dataframe(df.head())
            if "churn_probability" in df.columns:
                fig = px.histogram(df, x="churn_probability", nbins=20, title="Churn Probability Distribution")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Please upload a CSV for prediction.")
