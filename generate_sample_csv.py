"""
generate_sample_csv.py
-----------------------
Creates a small sample test CSV (with true loan_status labels) from the
full dataset, for uploading to the Streamlit app to demo predictions + metrics.
"""

import pandas as pd

df = pd.read_csv("loan_approval_dataset.csv")
df.columns = df.columns.str.strip()

for col in df.select_dtypes(include=["object", "str"]).columns:
    df[col] = df[col].str.strip()

# sample each class separately, keeping roughly the same Approved/Rejected ratio
n_total = 50
fraction = n_total / len(df)

approved = df[df["loan_status"] == "Approved"].sample(frac=fraction, random_state=1)
rejected = df[df["loan_status"] == "Rejected"].sample(frac=fraction, random_state=1)

sample = pd.concat([approved, rejected]).sample(frac=1, random_state=1)  # shuffle rows

sample.to_csv("test_data.csv", index=False)
print(f"Saved test_data.csv with {len(sample)} rows")
print(sample["loan_status"].value_counts())