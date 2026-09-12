import streamlit as st
import pandas as pd
import requests
from pathlib import Path

# Page Config
st.set_page_config(page_title="RecSys Dashboard", layout="wide")

st.title("🚀 Multi-Objective Recommender Dashboard")
st.markdown("This dashboard compares a **Standard AI (SVD)** vs. our **Multi-Objective Innovation**.")

# Sidebar for API configuration
st.sidebar.header("Settings")
api_url = st.sidebar.text_input("API URL", value="http://localhost:8000/recommend")
user_id = st.sidebar.text_input("Enter User ID", value="user_1")
num_recs = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

if st.button("Generate Recommendations"):
    try:
        # 1. Get Naive Recommendations
        naive_payload = {"user_id": user_id, "n": num_recs, "mode": "naive"}
        naive_res = requests.post(api_url, json=naive_payload).json()

        # 2. Get Multi-Objective Recommendations
        mo_payload = {"user_id": user_id, "n": num_recs, "mode": "multi_objective"}
        mo_res = requests.post(api_url, json=mo_payload).json()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📉 Naive AI (SVD)")
            st.write("Focuses only on predicted rating.")
            st.table(pd.DataFrame(naive_res['recommendations'], columns=["Item ID"]))

        with col2:
            st.subheader("✨ Multi-Objective AI")
            st.write("Balances Relevance, Novelty, and Diversity.")
            st.table(pd.DataFrame(mo_res['recommendations'], columns=["Item ID"]))

        # Comparison Analysis
        naive_set = set(naive_res['recommendations'])
        mo_set = set(mo_res['recommendations'])
        overlap = len(naive_set & mo_set)

        st.divider()
        st.subheader("Analysis")
        st.info(f"The systems agree on {overlap} items. The Multi-Objective system introduced {num_recs - overlap} new discoveries!")

    except Exception as e:
        st.error(f"Could not connect to API. Make sure you ran `python api/main.py`. Error: {e}")
