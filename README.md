💳 AI-Driven Credit Risk Assessment System (Explainable AI)

An AI-driven, explainable credit risk assessment system designed to support transparent and responsible credit decision-making.
The system predicts the probability of customer default, categorizes risk into Low / Medium / High tiers, and provides human-readable explanations to assist decision-makers.

⚠️ This is a decision-support system, not an automated approval engine.

📌 Key Features

End-to-end Machine Learning pipeline

Probability-based credit default prediction

Risk tiers (Low / Medium / High) aligned with industry practice

Explainable AI (XAI) for transparent decisions

Input validation and out-of-distribution safety handling

Professional Streamlit dashboard

Offline SHAP analysis for model interpretability

🧠 Why This Project Matters

In real banking and fintech systems:

Models must be interpretable

Decisions must be explainable

Humans must remain in the loop

This project reflects how credit risk models are actually used in production, not just how they are trained.

📊 Dataset

UCI Credit Card Default Dataset (Taiwan)
A real-world financial dataset widely used in academic research and industry benchmarking.

Target Variable

default.payment.next.month → Indicates whether the customer defaulted

Example Features

LIMIT_BAL – Credit limit

AGE, SEX – Demographics

PAY_0 … PAY_6 – Repayment status

BILL_AMT* – Outstanding bills

PAY_AMT* – Payments made

🏗️ System Architecture
User Input (UI)
      ↓
Input Validation & Safety Capping
      ↓
Feature Engineering
      ↓
Scaling (StandardScaler)
      ↓
ML Model (Logistic Regression)
      ↓
Probability of Default
      ↓
Risk Tier (Low / Medium / High)
      ↓
Human-Readable Explanation
      ↓
Decision Recommendation

⚙️ Tech Stack

Language: Python

ML: Scikit-learn (Logistic Regression)

Data: Pandas, NumPy

Explainability: SHAP (offline analysis)

UI: Streamlit

Model Persistence: Joblib


🚀 How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/ravikiranediga/credit-risk-xai.git
cd credit-risk-xai

2️⃣ Create Virtual Environment
python -m venv venv


Activate:

Windows

venv\Scripts\activate


Mac / Linux

source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Pipeline
python src/data_processing.py
python src/train_models.py
python src/explainability.py

5️⃣ Launch the Application
streamlit run app/main.py

🧠 Explainable AI (XAI)

Explainability is handled at two levels:

User-level (UI):
Business-friendly explanations such as repayment behavior and credit exposure.

Model-level (Offline):
SHAP visualizations saved in the outputs/ directory for audit and analysis.

This ensures both usability and model transparency.

📈 Risk Interpretation
Probability of Default	Risk Level
< 30%	Low Risk
30–60%	Medium Risk
> 60%	High Risk
🔒 Disclaimer

This system provides decision support only.
Final credit approval decisions must always involve human judgment and institutional policy checks.

👤 Author & Contact

Name: Ravi Kiran Ediga
Role: Aspiring Data Scientist / Machine Learning Engineer

GitHub: https://github.com/ravikiranediga

LinkedIn: https://www.linkedin.com/in/ravikiranediga

⭐ Final Note

This project demonstrates end-to-end ownership, explainable AI, and real-world ML deployment thinking, making it suitable as a major project for interviews and professional portfolios.

