import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
import os

# === Config ===
train_filename = "Manager_EndShiftRemark_List_13-05-2025 09_59_18.xlsx"
other_files = ["complex_worker_remarks.xlsx", "Manager_EndShiftRemark_List_Demo.xlsx"]

# === Load model ===
model = SentenceTransformer("all-MiniLM-L6-v2")
scaler = MinMaxScaler()
mlp = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)

# === STEP 1: Load training data ===
train_df = pd.read_excel(train_filename)
train_df.columns = train_df.columns.str.strip()

# Auto-detect remark column
remark_col = next((col for col in train_df.columns if "remark" in col.lower()), None)
if not remark_col:
    raise ValueError("No remark column found in training file.")

# Drop rows with missing grade and remark
train_labeled = train_df.dropna(subset=[remark_col, "Grade"])
X_train = np.stack(train_labeled[remark_col].apply(model.encode))
y_train = train_labeled["Grade"].astype(float).values

# Train model
X_train_scaled = scaler.fit_transform(X_train)
mlp.fit(X_train_scaled, y_train)

# === STEP 2: Predict missing grades in training file itself ===
train_missing = train_df[train_df["Grade"].isna()]
if not train_missing.empty:
    X_missing = np.stack(train_missing[remark_col].apply(model.encode))
    X_missing_scaled = scaler.transform(X_missing)
    preds_A = np.rint(mlp.predict(X_missing_scaled)).astype(int)

    # Fill missing values in "Grade" column
    train_df.loc[train_df["Grade"].isna(), "Grade"] = preds_A

# Save output
output_filename = train_filename.replace(".xlsx", "_predicted.xlsx")
train_df.to_excel(output_filename, index=False)
print(f"Saved updated training file: {output_filename}")

# === STEP 3: Predict for other Excel files ===
for file in other_files:
    try:
        df = pd.read_excel(file)
        df.columns = df.columns.str.strip()

        # Use same remark column name if present, or find similar one
        if remark_col in df.columns:
            current_remark_col = remark_col
        else:
            # Try to find one that includes 'remark'
            current_remark_col = next((col for col in df.columns if "remark" in col.lower()), None)

        if not current_remark_col:
            print(f"Skipping {file} — No remark column found.")
            continue

        df = df.dropna(subset=[current_remark_col])
        X = np.stack(df[current_remark_col].apply(model.encode))
        X_scaled = scaler.transform(X)

        preds = np.rint(mlp.predict(X_scaled)).astype(int)
        df["Predicted_Grade"] = preds

        output_file = file.replace(".xlsx", "_predicted.xlsx")
        df.to_excel(output_file, index=False)
        print(f"Processed and saved: {output_file}")

    except Exception as e:
        print(f"Error processing {file}: {e}")
