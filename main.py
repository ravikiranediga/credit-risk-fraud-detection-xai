import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Credit Risk Assessment System",
    page_icon="💳",
    layout="centered"
)

# Load Model Artifacts
model = joblib.load("models/credit_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# Header
st.title("💳 Credit Risk Assessment System")

st.markdown("""
This application predicts the probability of customer credit default using machine learning.
It provides risk-tiered decisions with human-readable explanations to support transparent and responsible credit approvals.
""")

st.divider()

# User Inputs 
st.subheader("📋 Customer Profile")

AGE = st.number_input("Age", min_value=18, max_value=100, value=28)
SEX = st.selectbox("Gender", ["Male", "Female"])
LIMIT_BAL = st.number_input("Credit Limit", min_value=1000, value=50000)

st.markdown("### Recent Repayment Behaviour")
PAY_0 = st.selectbox(
    "Recent Payment Status",
    options=[0, 1, 2, 3, 4],
    help="0 = On-time, 1–4 = Increasing delay severity"
)


# Prediction
if st.button("🔍 Assess Credit Risk"):

    
    # Input Safety Caps 
    MAX_LIMIT_BAL = 1_000_000  # based on training data range
    limit_was_capped = False

    if LIMIT_BAL > MAX_LIMIT_BAL:
        LIMIT_BAL = MAX_LIMIT_BAL
        limit_was_capped = True

    
    # Build Full Feature Vector
    input_data = dict.fromkeys(feature_columns, 0)

    # User-controlled core features
    input_data["AGE"] = AGE
    input_data["LIMIT_BAL"] = LIMIT_BAL
    input_data["SEX"] = 1 if SEX == "Male" else 2
    input_data["PAY_0"] = PAY_0

    # Hidden but realistic defaults 
    defaults = {
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "PAY_2": PAY_0,
        "PAY_3": PAY_0,
        "PAY_4": PAY_0,
        "PAY_5": PAY_0,
        "PAY_6": PAY_0,
        "BILL_AMT1": LIMIT_BAL * 0.30,
        "BILL_AMT2": LIMIT_BAL * 0.25,
        "BILL_AMT3": LIMIT_BAL * 0.20,
        "BILL_AMT4": LIMIT_BAL * 0.15,
        "BILL_AMT5": LIMIT_BAL * 0.10,
        "BILL_AMT6": LIMIT_BAL * 0.05,
        "PAY_AMT1": LIMIT_BAL * 0.05,
        "PAY_AMT2": LIMIT_BAL * 0.04,
        "PAY_AMT3": LIMIT_BAL * 0.03,
        "PAY_AMT4": LIMIT_BAL * 0.02,
        "PAY_AMT5": LIMIT_BAL * 0.02,
        "PAY_AMT6": LIMIT_BAL * 0.01,
    }

    for k, v in defaults.items():
        if k in input_data:
            input_data[k] = v

    # Create dataframe in correct order
    input_df = pd.DataFrame([input_data], columns=feature_columns)

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict probability
    probability = model.predict_proba(input_scaled)[0][1]

    # Risk Tiers 
    
    if probability < 0.30:
        risk_label = "LOW RISK"
    elif probability < 0.60:
        risk_label = "MEDIUM RISK"
    else:
        risk_label = "HIGH RISK"

    # Clip probability for display 
    display_prob = max(min(probability, 0.99), 0.01)

    
    # Output
    st.divider()
    st.subheader("📊 Credit Risk Assessment")

    if risk_label == "HIGH RISK":
        st.error("🚨 High Credit Risk")
    elif risk_label == "MEDIUM RISK":
        st.warning("⚠️ Medium Credit Risk")
    else:
        st.success("✅ Low Credit Risk")

    st.metric(
        label="Estimated Probability of Default",
        value=f"{display_prob:.2%}"
    )

    # Inform user if input was adjusted
    if limit_was_capped:
        st.info(
            "ℹ️ Entered credit limit exceeds typical historical range. "
            "For prediction reliability, the value was adjusted to realistic bounds."
        )

    # Explanation 
    
    st.subheader("🧠 Why this decision was made")

    reasons = []

    if PAY_0 >= 2:
        reasons.append("Recent repayment delays indicate higher default risk")
    if LIMIT_BAL >= 300000:
        reasons.append("High credit exposure requires closer monitoring")
    if AGE < 25:
        reasons.append("Younger age combined with repayment behaviour increases uncertainty")

    if not reasons:
        reasons.append("Consistent repayment behaviour and stable credit profile")

    for r in reasons:
        st.write(f"• {r}")

    
    # Recommendation 
   
    st.subheader("✅ Recommendation")

    if risk_label == "LOW RISK":
        st.write("• Consistent repayment behaviour observed")
        st.write("• Credit exposure is well supported")
        st.write("• Suitable for standard credit approval workflow")

    elif risk_label == "MEDIUM RISK":
        st.write("• Moderate repayment risk detected")
        st.write("• Manual review or reduced exposure recommended")
        st.write("• Approval subject to internal policy thresholds")

    else:
        st.write("• High likelihood of repayment default")
        st.write("• Payment delay history is a key concern")
        st.write("• Additional guarantees or rejection recommended")

    st.caption(
        "This system provides decision support only and does not replace human judgment."
    )
