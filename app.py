import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from predict import score_new_data
from score import find_saved_model, ARTIFACT_DIR

# --- Page Config ---
st.set_page_config(
    page_title="Adversarial Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

# --- Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2632/2632281.png", width=100)
    st.title("Settings")

    st.info("Ensure training has been run and models exist in the `artifacts/` folder.")

    # Check for models
    try:
        model, metadata, model_name = find_saved_model(ARTIFACT_DIR)
        st.success(f"Loaded: {model_name}")
    except FileNotFoundError:
        st.error("No saved models found. Please train a model first.")
        st.stop()

# --- Main Content ---
st.title("🛡️ Adversarial Fraud Detection")
st.markdown("Automated pipeline for merging transaction/identity data and scoring fraud risk.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📁 Upload Data")
    transaction_file = st.file_uploader("Transactions CSV", type=["csv"])
    identity_file = st.file_uploader("Identity CSV", type=["csv"])

with col2:
    st.subheader("💡 Instructions")
    st.write("""
    1. Upload your **Transaction** and **Identity** CSV files.
    2. The system will automatically merge and engineer features.
    3. Click 'Run Inference' to see the fraud risk scores.
    """)

    run_btn = st.button("🚀 Run Inference")

if run_btn:
    if transaction_file and identity_file:
        with st.spinner("Processing data and running inference..."):
            # Save files temporarily
            temp_trans = Path("temp_trans.csv")
            temp_id = Path("temp_id.csv")

            with open(temp_trans, "wb") as f:
                f.write(transaction_file.getbuffer())
            with open(temp_id, "wb") as f:
                f.write(identity_file.getbuffer())

            results = None
            try:
                # Run inference
                results = score_new_data(
                    new_transaction_path=str(temp_trans),
                    new_identity_path=str(temp_id),
                    best_model_name=model_name,
                    models={model_name: model},
                    threshold=metadata["threshold"],
                    categorical_cols=metadata["categorical_cols"],
                    numeric_cols=metadata["numeric_cols"],
                    cat_num_medians=metadata.get("cat_num_medians"),
                )
            except Exception as e:
                st.error(f"Error during inference: {str(e)}")
            finally:
                # Always clean up temp files, even if inference raises
                temp_trans.unlink(missing_ok=True)
                temp_id.unlink(missing_ok=True)

            if results is not None:
                # --- Results Visualization ---
                st.success("Analysis Complete!")

                # Summary Metrics
                m1, m2, m3 = st.columns(3)
                total_cases = len(results)
                fraud_count = int(results['fraud_prediction'].sum())
                fraud_rate = (fraud_count / total_cases) * 100

                m1.metric("Total Transactions", total_cases)
                m2.metric("Fraudulent Flagged", fraud_count)
                m3.metric("Fraud Rate", f"{fraud_rate:.2f}%")

                # Charts
                st.divider()
                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    st.subheader("🔥 Fraud Probability Distribution")
                    fig, ax = plt.subplots()
                    sns.histplot(results['fraud_probability'], bins=30, kde=True, color='red', ax=ax)
                    ax.set_xlabel("Probability")
                    st.pyplot(fig)

                with chart_col2:
                    st.subheader("🔍 Class Breakdown")
                    fig, ax = plt.subplots()
                    results['fraud_prediction'].value_counts().plot.pie(
                        autopct='%1.1f%%', labels=['Legit', 'Fraud'], colors=['#2ecc71', '#e74c3c'], ax=ax
                    )
                    st.pyplot(fig)

                # Results Table
                st.subheader("📄 Top High-Risk Transactions")
                st.dataframe(results.sort_values('fraud_probability', ascending=False).head(50), use_container_width=True)

                # Download
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Results CSV",
                    data=csv,
                    file_name='fraud_scores.csv',
                    mime='text/csv',
                )
    else:
        st.warning("Please upload both files before running.")

# Footer
st.markdown("---")
st.caption("Adversarial Fraud Detection Pipeline • Powered by Streamlit")
