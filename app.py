import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")

FEATURE_COLUMNS = [
    "CreditScore", "Gender", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
    "Geography_Germany", "Geography_Spain",
]


@st.cache_resource
def load_artifacts():
    model = load_model("models/churn_model.keras")
    scaler = pickle.load(open("pickle/scaler.pkl", "rb"))
    geo_encoder = pickle.load(open("pickle/onehot_encoder_geo.pkl", "rb"))
    return model, scaler, geo_encoder


def predict_churn(model, scaler, geo_encoder, input_data):
    input_df = pd.DataFrame([input_data])
    input_df["Gender"] = input_df["Gender"].map({"Male": 1, "Female": 0})

    geo_encoded = geo_encoder.transform(input_df[["Geography"]])
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=geo_encoder.get_feature_names_out(["Geography"]),
        index=input_df.index,
    )
    input_df = pd.concat([input_df.drop("Geography", axis=1), geo_encoded_df], axis=1)
    input_df = input_df[FEATURE_COLUMNS]

    input_scaled = scaler.transform(input_df)
    probability = float(model.predict(input_scaled, verbose=0)[0][0])
    return probability


def risk_tier(probability):
    if probability < 0.3:
        return "Low", "#0ca30c"
    if probability < 0.6:
        return "Medium", "#fab219"
    return "High", "#d03b3b"


st.markdown(
    """
    <style>
    :root {
        --surface-1: #fcfcfb;
        --text-primary: #0b0b0b;
        --text-secondary: #52514e;
        --border: rgba(11,11,11,0.10);
        --track: #e1e0d9;
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --surface-1: #1a1a19;
            --text-primary: #ffffff;
            --text-secondary: #c3c2b7;
            --border: rgba(255,255,255,0.10);
            --track: #2c2c2a;
        }
    }
    .app-header {
        padding: 1.75rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #2a78d6 0%, #1c5cab 100%);
        color: #ffffff;
        margin-bottom: 1.5rem;
    }
    .app-header h1 { margin: 0; font-size: 1.6rem; }
    .app-header p { margin: 0.35rem 0 0; opacity: 0.9; font-size: 0.95rem; }

    .result-card {
        background: var(--surface-1);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.75rem;
    }
    .result-hero {
        font-size: 3rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1;
        margin-bottom: 0.25rem;
    }
    .result-caption {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        background: var(--track);
        color: var(--text-primary);
        margin-bottom: 1.25rem;
    }
    .status-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }
    .meter-track {
        width: 100%;
        height: 10px;
        border-radius: 999px;
        background: var(--track);
        overflow: hidden;
    }
    .meter-fill {
        height: 100%;
        border-radius: 999px;
    }
    .placeholder {
        color: var(--text-secondary);
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <h1>Customer Churn Predictor</h1>
        <p>Enter a customer's profile to estimate their likelihood of churning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model, scaler, geo_encoder = load_artifacts()
geography_options = list(geo_encoder.categories_[0])

input_col, result_col = st.columns([3, 2], gap="large")

with input_col:
    with st.form("churn_form"):
        st.subheader("Customer profile")

        c1, c2, c3 = st.columns(3)
        with c1:
            credit_score = st.number_input("Credit score", 300, 850, 650)
            age = st.number_input("Age", 18, 100, 35)
            tenure = st.number_input("Tenure (years)", 0, 10, 3)
        with c2:
            balance = st.number_input("Balance", 0.0, 250000.0, 50000.0, step=1000.0)
            num_products = st.selectbox("Number of products", [1, 2, 3, 4])
            salary = st.number_input("Estimated salary", 0.0, 250000.0, 100000.0, step=1000.0)
        with c3:
            geography = st.selectbox("Geography", geography_options)
            gender = st.selectbox("Gender", ["Male", "Female"])
            has_cr_card = st.selectbox("Has credit card?", ["Yes", "No"])
            is_active = st.selectbox("Active member?", ["Yes", "No"])

        submitted = st.form_submit_button("Predict churn", use_container_width=True)

if submitted:
    input_data = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_cr_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active == "Yes" else 0,
        "EstimatedSalary": salary,
    }
    probability = predict_churn(model, scaler, geo_encoder, input_data)
    st.session_state["probability"] = probability

with result_col:
    st.subheader("Prediction")
    probability = st.session_state.get("probability")

    if probability is None:
        st.markdown(
            '<div class="result-card"><p class="placeholder">'
            "Fill in the form and click <b>Predict churn</b> to see the result here."
            "</p></div>",
            unsafe_allow_html=True,
        )
    else:
        tier, color = risk_tier(probability)
        pct = probability * 100
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-hero">{pct:.1f}%</div>
                <div class="result-caption">Estimated probability of churn</div>
                <div class="status-pill">
                    <span class="status-dot" style="background:{color};"></span>
                    {tier} risk
                </div>
                <div class="meter-track">
                    <div class="meter-fill" style="width:{pct:.1f}%; background:{color};"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
