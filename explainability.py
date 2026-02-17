import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import os

# Setup

os.makedirs("outputs", exist_ok=True)

# Load artifacts

model = joblib.load("models/credit_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

X_test = pd.read_csv("data/processed/X_test_scaled.csv")
X_test.columns = feature_columns


# Predict probabilities

probs = model.predict_proba(X_test)[:, 1]

high_risk = probs >= 0.5
low_risk = probs < 0.5

X_high = X_test[high_risk]
X_low = X_test[low_risk]


# SHAP explainer

explainer = shap.LinearExplainer(model, X_test)
shap_values = explainer.shap_values(X_test)

# HIGH RISK SHAP

plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values[high_risk],
    X_high,
    show=False
)
plt.title("SHAP Summary – High Risk Customers")
plt.tight_layout()
plt.savefig("outputs/shap_high_risk.png")
plt.close()

# LOW RISK SHAP

plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values[low_risk],
    X_low,
    show=False
)
plt.title("SHAP Summary – Low Risk Customers")
plt.tight_layout()
plt.savefig("outputs/shap_low_risk.png")
plt.close()

print("✅ SHAP plots generated:")
print(" - outputs/shap_high_risk.png")
print(" - outputs/shap_low_risk.png")
