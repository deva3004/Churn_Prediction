import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Customer Insights Predictor", page_icon="📉", layout="wide")

CHURN_FEATURE_COLUMNS = [
    "CreditScore", "Gender", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
    "Geography_Germany", "Geography_Spain",
]

SALARY_FEATURE_COLUMNS = [
    "CreditScore", "Gender", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "Exited",
    "Geography_Germany", "Geography_Spain",
]


@st.cache_resource
def load_churn_artifacts():
    model = load_model("models/churn_model.keras")
    scaler = pickle.load(open("pickle/scaler.pkl", "rb"))
    geo_encoder = pickle.load(open("pickle/onehot_encoder_geo.pkl", "rb"))
    return model, scaler, geo_encoder


@st.cache_resource
def load_salary_artifacts():
    model = load_model("models/salary_regression_model.keras")
    scaler = pickle.load(open("pickle/scaler_salary.pkl", "rb"))
    geo_encoder = pickle.load(open("pickle/onehot_encoder_geo_salary.pkl", "rb"))
    return model, scaler, geo_encoder


def _encode_input(geo_encoder, input_data, feature_columns):
    input_df = pd.DataFrame([input_data])
    input_df["Gender"] = input_df["Gender"].map({"Male": 1, "Female": 0})

    geo_encoded = geo_encoder.transform(input_df[["Geography"]])
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=geo_encoder.get_feature_names_out(["Geography"]),
        index=input_df.index,
    )
    input_df = pd.concat([input_df.drop("Geography", axis=1), geo_encoded_df], axis=1)
    return input_df[feature_columns]


def predict_churn(model, scaler, geo_encoder, input_data):
    input_df = _encode_input(geo_encoder, input_data, CHURN_FEATURE_COLUMNS)
    input_scaled = scaler.transform(input_df)
    return float(model.predict(input_scaled, verbose=0)[0][0])


def predict_salary(model, scaler, geo_encoder, input_data):
    input_df = _encode_input(geo_encoder, input_data, SALARY_FEATURE_COLUMNS)
    input_scaled = scaler.transform(input_df)
    return float(model.predict(input_scaled, verbose=0)[0][0])


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
        <h1>Customer Insights Predictor</h1>
        <p>Enter a customer's profile to estimate churn risk or their expected salary.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

mode = st.radio(
    "What would you like to predict?",
    ["Customer Churn", "Estimated Salary"],
    horizontal=True,
)
is_churn_mode = mode == "Customer Churn"

if is_churn_mode:
    model, scaler, geo_encoder = load_churn_artifacts()
else:
    model, scaler, geo_encoder = load_salary_artifacts()
geography_options = list(geo_encoder.categories_[0])

input_col, result_col = st.columns([3, 2], gap="large")

with input_col:
    with st.form("prediction_form"):
        st.subheader("Customer profile")

        c1, c2, c3 = st.columns(3)
        with c1:
            credit_score = st.number_input("Credit score", 300, 850, 650)
            age = st.number_input("Age", 18, 100, 35)
            tenure = st.number_input("Tenure (years)", 0, 10, 3)
        with c2:
            balance = st.number_input("Balance", 0.0, 250000.0, 50000.0, step=1000.0)
            num_products = st.selectbox("Number of products", [1, 2, 3, 4])
            if is_churn_mode:
                salary = st.number_input("Estimated salary", 0.0, 250000.0, 100000.0, step=1000.0)
            else:
                has_churned = st.selectbox("Has the customer churned?", ["Yes", "No"])
        with c3:
            geography = st.selectbox("Geography", geography_options)
            gender = st.selectbox("Gender", ["Male", "Female"])
            has_cr_card = st.selectbox("Has credit card?", ["Yes", "No"])
            is_active = st.selectbox("Active member?", ["Yes", "No"])

        button_label = "Predict churn" if is_churn_mode else "Predict salary"
        submitted = st.form_submit_button(button_label, use_container_width=True)

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
    }
    if is_churn_mode:
        input_data["EstimatedSalary"] = salary
        result_value = predict_churn(model, scaler, geo_encoder, input_data)
    else:
        input_data["Exited"] = 1 if has_churned == "Yes" else 0
        result_value = predict_salary(model, scaler, geo_encoder, input_data)
    st.session_state["result"] = {"mode": mode, "value": result_value}

with result_col:
    st.subheader("Prediction")
    result = st.session_state.get("result")

    if result is None or result["mode"] != mode:
        action = "Predict churn" if is_churn_mode else "Predict salary"
        st.markdown(
            '<div class="result-card"><p class="placeholder">'
            f"Fill in the form and click <b>{action}</b> to see the result here."
            "</p></div>",
            unsafe_allow_html=True,
        )
    elif is_churn_mode:
        probability = result["value"]
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
    else:
        salary_value = result["value"]
        meter_pct = max(0.0, min(100.0, salary_value / 200000 * 100))
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-hero">${salary_value:,.0f}</div>
                <div class="result-caption">Estimated annual salary</div>
                <div class="meter-track">
                    <div class="meter-fill" style="width:{meter_pct:.1f}%; background:#2a78d6;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
