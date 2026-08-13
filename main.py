"""
main.py
-------
Loan Approval Prediction - Machine Learning Assignment 2

Trains and evaluates 5 classification models on the loan approval dataset:
    1. Logistic Regression
    2. Decision Tree
    3. K-Nearest Neighbors (kNN)
    4. Naive Bayes (Gaussian)
    5. Random Forest (Ensemble)

For each model, computes: Accuracy, AUC, Precision, Recall, F1, MCC.
"""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)


# ---------------------------------------------------------------
# Reusable function: train a model and print/return its metrics
# ---------------------------------------------------------------
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """
    Fits `model` on the training data, predicts on the test data,
    and computes all 6 required evaluation metrics.
    """
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probability of class 1 (Approved)
    # needed for AUC, which measures ranking quality across all thresholds,
    # not just the final 0/1 prediction

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    print(f"\n=== {name} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"MCC:       {mcc:.4f}")

    return {
        "Model": name, "Accuracy": acc, "AUC": auc,
        "Precision": prec, "Recall": rec, "F1": f1, "MCC": mcc
    }


# ---------------------------------------------------------------
# 1. Load and clean the data
# ---------------------------------------------------------------
df = pd.read_csv("loan_approval_dataset.csv")

# the raw CSV has leading spaces in column names and string values - strip them
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include=["object", "str"]).columns:
    df[col] = df[col].str.strip()

# loan_id is just a row identifier, not a real feature - drop it
df = df.drop(columns=["loan_id"])

# ---------------------------------------------------------------
# 2. Encode categorical columns and the target as numbers
# ---------------------------------------------------------------
df["education"] = df["education"].map({"Graduate": 1, "Not Graduate": 0})
df["self_employed"] = df["self_employed"].map({"Yes": 1, "No": 0})
df["loan_status"] = df["loan_status"].map({"Approved": 1, "Rejected": 0})

# ---------------------------------------------------------------
# 2b. Engineered feature: total assets = sum of all 4 asset-value columns
# ---------------------------------------------------------------
# The dataset ships with 11 raw predictor features, one short of the
# assignment's minimum of 12. total_assets_value is a genuinely meaningful
# underwriting signal (lenders look at total collateral, not just each
# asset type separately) rather than a throwaway padding feature.
df["total_assets_value"] = (
    df["residential_assets_value"]
    + df["commercial_assets_value"]
    + df["luxury_assets_value"]
    + df["bank_asset_value"]
)

# ---------------------------------------------------------------
# 3. Split features / target, then train / test
# ---------------------------------------------------------------
X = df.drop(columns=["loan_status"])
y = df["loan_status"]

print("Number of features:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    # stratify keeps the Approved/Rejected ratio consistent in both splits
)

# ---------------------------------------------------------------
# 4. Scale features (helps Logistic Regression & kNN; harmless for the rest)
# ---------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # learn mean/std from train only
X_test_scaled = scaler.transform(X_test)          # apply same scaling to test

# ---------------------------------------------------------------
# 5. Train and evaluate all 5 models
# ---------------------------------------------------------------
results = []

log_reg = LogisticRegression(max_iter=1000, random_state=42)
results.append(evaluate_model(
    "Logistic Regression", log_reg,
    X_train_scaled, X_test_scaled, y_train, y_test
))

dt = DecisionTreeClassifier(random_state=42)
results.append(evaluate_model(
    "Decision Tree", dt,
    X_train, X_test, y_train, y_test   # trees don't need scaled data
))

knn = KNeighborsClassifier(n_neighbors=5)
results.append(evaluate_model(
    "K-Nearest Neighbors", knn,
    X_train_scaled, X_test_scaled, y_train, y_test   # kNN is distance-based, needs scaling
))

nb = GaussianNB()
results.append(evaluate_model(
    "Naive Bayes", nb,
    X_train, X_test, y_train, y_test
))

rf = RandomForestClassifier(n_estimators=200, random_state=42)
results.append(evaluate_model(
    "Random Forest (Ensemble)", rf,
    X_train, X_test, y_train, y_test   # trees don't need scaled data
))

# ---------------------------------------------------------------
# 6. Print a final comparison table
# ---------------------------------------------------------------
results_df = pd.DataFrame(results)
print("\n\n=== Model Comparison ===")
print(results_df.to_string(index=False))

# save it - you'll need this table for your README and Streamlit app
results_df.to_csv("model/metrics_comparison.csv", index=False)

# ---------------------------------------------------------------
# 7. Save trained models + scaler so the Streamlit app can load them
#    without retraining
# ---------------------------------------------------------------
joblib.dump(log_reg, "model/logistic_regression.pkl")
joblib.dump(dt, "model/decision_tree.pkl")
joblib.dump(knn, "model/knn.pkl")
joblib.dump(nb, "model/naive_bayes.pkl")
joblib.dump(rf, "model/random_forest.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\nAll models and scaler saved to model/")