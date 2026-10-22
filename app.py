"""Streamlit demo:  streamlit run app.py"""
import pandas as pd
import streamlit as st
from src.predict import load_artifact, score_dataframe

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="wide")
st.title("💳 Credit Card Fraud Detector")
st.caption("Upload transactions (columns: Time, V1..V28, Amount). The model flags likely fraud.")

try:
    art = load_artifact()
except FileNotFoundError:
    st.error("Model not found. Run `python -m src.train` first.")
    st.stop()

st.sidebar.write(f"**Model:** {art['name']}")
thr = st.sidebar.slider("Decision threshold", 0.01, 0.99, float(min(max(art["threshold"], 0.01), 0.99)), 0.01)
st.sidebar.caption("Lower = catch more fraud, more false alarms.")

file = st.file_uploader("Upload CSV", type="csv")
if file:
    df = pd.read_csv(file)
    res = score_dataframe(df, art)
    res["is_fraud_flag"] = (res["fraud_score"] >= thr).astype(int)
    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions", f"{len(res):,}")
    c2.metric("Flagged as fraud", int(res["is_fraud_flag"].sum()))
    c3.metric("Flagged amount", f"{res.loc[res.is_fraud_flag == 1, 'Amount'].sum():,.2f}")
    st.subheader("Highest-risk transactions")
    st.dataframe(res.sort_values("fraud_score", ascending=False).head(50))
    st.download_button("Download predictions", res.to_csv(index=False), "predictions.csv")
