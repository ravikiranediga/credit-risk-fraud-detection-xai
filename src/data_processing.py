import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


# Create folders
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load dataset

df = pd.read_csv("data/raw/credit_data.csv")
print(f"✅ Loaded dataset with shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f'Sample data:\n{df.head()}')

# Drop useless index column
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

print(f"Missing values:\n{df.isnull().sum()}")


# Target & Features

TARGET = "default.payment.next.month"

y = df[TARGET]
X = df.drop(columns=[TARGET])


# Save feature 

joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")


# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# Scaling

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save processed data
pd.DataFrame(X_train_scaled, columns=X.columns)\
    .to_csv("data/processed/X_train_scaled.csv", index=False)

pd.DataFrame(X_test_scaled, columns=X.columns)\
    .to_csv("data/processed/X_test_scaled.csv", index=False)

y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

joblib.dump(scaler, "models/scaler.pkl")

print("✅ Data processing completed successfully")
