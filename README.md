# Loan Approval Prediction — ML Assignment 2

## a. Problem Statement

Loan approval is one of the most common decisions financial institutions make, and
it directly affects both the bank's risk exposure and the applicant's access to
credit. Manually evaluating every application against income, credit history, and
asset details is slow and inconsistent across reviewers. This project builds and
compares six machine learning classification models that predict whether a loan
application will be **Approved** or **Rejected**, based on the applicant's
financial and demographic details. An interactive Streamlit web app is built and
deployed so the models can be demonstrated and evaluated on new/test data.

## b. Dataset Description

- **Name:** Loan Approval Prediction Dataset
- **Source:** Kaggle (`https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset`)
- **Instances:** 4,269
- **Features:** 12 total —
  - 11 raw features: number of dependents, education, employment status, annual
    income, loan amount, loan term, CIBIL (credit) score, and four asset-value
    columns (residential, commercial, luxury, bank assets)
  - 1 engineered feature: `total_assets_value`, the sum of the four individual
    asset-value columns — a genuine underwriting signal (lenders look at total
    collateral, not just each asset type in isolation), and also brings the
    dataset up to the assignment's minimum of 12 features
- **Target variable:** `loan_status` — binary: `Approved` / `Rejected`.
- **Class balance:** 2,656 Approved vs. 1,613 Rejected (~62% / 38%) — reasonably
  balanced, no resampling needed.
- **Missing values:** None.
- **Data cleaning notes:** the raw CSV had leading whitespace in column names and
  string values, which was stripped before use. `loan_id` was dropped since it's
  just a row identifier with no predictive value. `education` and `self_employed`
  were label-encoded (binary categories), and the target was encoded to 1/0.
- **Split used:** 80% train / 20% test, stratified by class (`random_state=42`).

## c. GitHub Repository Link

https://github.com/Nandu2801/ml-assignment-2

## d. Models Used

Six classification models were trained on identical train/test splits.
Logistic Regression and kNN were trained on standardized (scaled) features;
Decision Tree, Naive Bayes, and Random Forest were trained on the raw features,
since tree-based and probability-based models don't require scaling.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9133 | 0.9726 | 0.9208 | 0.9416 | 0.9311 | 0.8148 |
| Decision Tree | 0.9836 | 0.9808 | 0.9814 | 0.9925 | 0.9869 | 0.9651 |
| kNN | 0.8852 | 0.9535 | 0.8972 | 0.9209 | 0.9089 | 0.7544 |
| Naive Bayes | 0.6218 | 0.7995 | 0.6218 | 1.0000 | 0.7668 | 0.0000 |
| Random Forest (Ensemble) | 0.9824 | 0.9989 | 0.9813 | 0.9906 | 0.9859 | 0.9626 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong baseline (F1 ≈ 0.93). CIBIL score and income are close to linearly related to approval, so a linear model already captures most of the signal. AUC of 0.97 shows it ranks applicants well even where the final approve/reject cutoff isn't perfect. |
| Decision Tree | Very high performance (F1 ≈ 0.99) — a single tree can carve out sharp threshold rules (e.g. "CIBIL score < 550 → reject") that mirror how real underwriting decisions are often made, which suits this dataset well. |
| kNN | Weaker than the linear/tree models (F1 ≈ 0.91) — with 12 features of very different scales, distance-based similarity is noisier, and it's more sensitive to borderline applicants near the approval threshold. |
| Naive Bayes | By far the weakest model (F1 ≈ 0.77, MCC = 0.00, meaning it performs no better than chance). Adding `total_assets_value` — a feature that is mathematically just the sum of four other features already in the dataset — makes Naive Bayes' independence assumption fail even harder, since it's now strongly correlated with several other inputs. Recall is a perfect 1.00 because the model essentially defaults to predicting "Approved" for almost everyone. |
| Random Forest (Ensemble) | Best overall on most metrics (F1 ≈ 0.99, MCC ≈ 0.96) — averaging many decision trees keeps the sharp threshold-style splits that work well here while reducing the overfitting risk of a single tree. Slightly behind Decision Tree on this particular test split, but with a notably higher AUC (0.999), suggesting more consistent ranking across thresholds. |
| **Overall Winner for this dataset** | **Decision Tree**, narrowly ahead of **Random Forest** on this test split (F1 0.9869 vs 0.9859) — though Random Forest's near-perfect AUC (0.999) makes it the safer, more consistent choice in practice. |

## Project Structure

```
ml-assignment2/
│-- app.py                       # Streamlit app
│-- main.py                      # trains all 6 models, saves them + metrics
│-- generate_sample_csv.py       # creates a stratified sample test CSV
│-- requirements.txt
│-- README.md
│-- test_data.csv                # sample test data used for app demo/evaluation
│-- loan_approval_dataset.csv    # full source dataset
│-- model/
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- random_forest.pkl
│   │-- scaler.pkl
│   │-- metrics_comparison.csv
```

## How to Run Locally

```bash
pip install -r requirements.txt
python3 main.py          # (optional) retrain models, regenerate metrics_comparison.csv
streamlit run app.py
```

## How to Use the App

1. Upload `test_data.csv` (or any CSV with the same feature columns, and optionally
   the `loan_status` column) using the sidebar uploader.
2. Choose a model from the dropdown.
3. View predictions, evaluation metrics, confusion matrix, and classification report
   on the **Predict & Evaluate** tab.
4. Compare all 5 models side-by-side on the **Compare All Models** tab.

## Live App

https://ml-assignment-2-jsgshdjcz8s2k6kqdyv32m.streamlit.app/