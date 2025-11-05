import streamlit as st, pandas as pd, plotly.express as px, requests

# paste your backend URL after Render deployment below
BACKEND_URL = "https://your-backend.onrender.com"

st.set_page_config(page_title="AI ERP Demo", layout="wide")
st.title("🧠 AI ERP Intelligence Dashboard")

try:
    st.caption(f"Connected to backend: {BACKEND_URL}")
    status = requests.get(f"{BACKEND_URL}/status", timeout=5).json()
    st.success("Backend status: OK")
except Exception as e:
    st.error(f"Backend not reachable: {e}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Sales Forecast")
    try:
        data = requests.get(f"{BACKEND_URL}/forecast", timeout=5).json()
        df = pd.DataFrame(data)
        fig = px.line(df, x="date", y="sales", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Forecast error: {e}")

with col2:
    st.subheader("💔 Churn Risk")
    try:
        data = requests.get(f"{BACKEND_URL}/churn", timeout=5).json()
        st.metric("Churn Rate", f"{data['churn_rate']*100:.1f}%")
        st.metric("High-Risk Customers", data["high_risk_customers"])
    except Exception as e:
        st.error(f"Churn error: {e}")
