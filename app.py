"""
app.py
------
Streamlit app for the Loan Approval Prediction assignment.

Features:
    - Upload test CSV data
    - Select a model from a dropdown
    - View predictions
    - View evaluation metrics, confusion matrix, and classification report
      (only when the uploaded CSV includes the true loan_status labels)
    - Compare all 5 trained models side-by-side
"""

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Loan Approval Classifier", layout="wide")
st.title("🏦 Loan Approval Prediction — Model Comparison")

# ---------------------------------------------------------------
# Load saved models, scaler, and the pre-computed comparison table
# ---------------------------------------------------------------
models = {
    "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "K-Nearest Neighbors": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest (Ensemble)": joblib.load("model/random_forest.pkl"),
}
scaler = joblib.load("model/scaler.pkl")
comparison_df = pd.read_csv("model/metrics_comparison.csv")

# models that were trained on SCALED data - need scaling applied at prediction time too
SCALED_MODELS = ["Logistic Regression", "K-Nearest Neighbors"]

# ---------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------
st.sidebar.header("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"])
model_choice = st.sidebar.selectbox("Select a model", list(models.keys()))

tab1, tab2 = st.tabs(["🔍 Predict & Evaluate", "📊 Compare All Models"])

# ---------------------------------------------------------------
# Tab 1: Predict + Evaluate on uploaded data
# ---------------------------------------------------------------
with tab1:
    if uploaded_file is None:
        st.info("👈 Upload a CSV file from the sidebar to get predictions.")
    else:
        data = pd.read_csv(uploaded_file)
        data.columns = data.columns.str.strip()

        st.subheader("Uploaded Data Preview")
        st.dataframe(data.head())

        # ---- Preprocess exactly like training: clean, drop id, encode ----
        processed = data.copy()
        for col in processed.select_dtypes(include=["object", "str"]).columns:
            processed[col] = processed[col].str.strip()

        if "loan_id" in processed.columns:
            processed = processed.drop(columns=["loan_id"])

        processed["education"] = processed["education"].map({"Graduate": 1, "Not Graduate": 0})
        processed["self_employed"] = processed["self_employed"].map({"Yes": 1, "No": 0})

        # engineered feature - must match main.py exactly, since the models
        # were trained expecting this column
        processed["total_assets_value"] = (
            processed["residential_assets_value"]
            + processed["commercial_assets_value"]
            + processed["luxury_assets_value"]
            + processed["bank_asset_value"]
        )

        has_labels = "loan_status" in processed.columns
        if has_labels:
            processed["loan_status"] = processed["loan_status"].map({"Approved": 1, "Rejected": 0})
            X_new = processed.drop(columns=["loan_status"])
            y_true = processed["loan_status"]
        else:
            X_new = processed

        # ---- Predict using the selected model ----
        model = models[model_choice]

        if model_choice in SCALED_MODELS:
            X_input = scaler.transform(X_new)
        else:
            X_input = X_new

        y_pred = model.predict(X_input)
        y_proba = model.predict_proba(X_input)[:, 1]
        pred_labels = pd.Series(y_pred).map({1: "Approved", 0: "Rejected"})

        result_df = data.copy()
        result_df["Predicted_Status"] = pred_labels.values

        st.subheader(f"Predictions — {model_choice}")
        st.dataframe(result_df)

        # ---------------------------------------------------------------
        # Evaluation metrics - only possible if the uploaded CSV has true labels
        # ---------------------------------------------------------------
        if has_labels:
            acc = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_proba)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)

            st.subheader("📈 Evaluation Metrics")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Accuracy", f"{acc:.3f}")
            c2.metric("AUC", f"{auc:.3f}")
            c3.metric("Precision", f"{prec:.3f}")
            c4.metric("Recall", f"{rec:.3f}")
            c5.metric("F1 Score", f"{f1:.3f}")
            c6.metric("MCC", f"{mcc:.3f}")

            col_a, col_b = st.columns([1, 1.3])

            with col_a:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(4, 3.5))
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Rejected", "Approved"],
                    yticklabels=["Rejected", "Approved"],
                    ax=ax
                )
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

            with col_b:
                st.subheader("Classification Report")
                report = classification_report(
                    y_true, y_pred,
                    target_names=["Rejected", "Approved"],
                    output_dict=True, zero_division=0
                )
                report_df = pd.DataFrame(report).transpose().round(3)
                st.dataframe(report_df)
        else:
            st.warning(
                "No 'loan_status' column found in the uploaded file — showing "
                "predictions only. Include the true label column to see evaluation metrics."
            )

# ---------------------------------------------------------------
# Tab 2: Compare all 5 models (using metrics computed during training)
# ---------------------------------------------------------------
with tab2:
    st.subheader("Model Comparison (on held-out test split)")
    st.dataframe(comparison_df)

    metric_to_plot = st.selectbox(
        "Metric to visualize", ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    )
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    sns.barplot(data=comparison_df, x="Model", y=metric_to_plot, ax=ax2)
    ax2.set_ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    st.pyplot(fig2)

    best_model = comparison_df.loc[comparison_df["F1"].idxmax(), "Model"]
    st.success(f"🏆 Best performing model (by F1 score): **{best_model}**")