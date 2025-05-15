import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer

# Load Excel file
file_path = r"C:\Users\Mtl\Downloads\Study\Python\Manager_EndShiftRemark_List_13-05-2025 09_59_18.xlsx"
df = pd.read_excel(file_path)

# Column names
remarks_column = 'END SHIFT REMARK'
grade_column = 'Grade'

# Drop rows with missing remarks (you can modify if needed)
df = df.dropna(subset=[remarks_column])

# Separate training data: rows 0 to 28 inclusive (first 29 rows)
train_df = df.loc[:28].dropna(subset=[grade_column, remarks_column])

# Separate prediction data: rows 29 to end
predict_df = df.loc[28:].dropna(subset=[remarks_column])

print(f"Training on {len(train_df)} rows; predicting on {len(predict_df)} rows.")

# Load SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    return model.encode(text)

print("Generating embeddings for training data...")
train_embeddings = np.stack(train_df[remarks_column].apply(get_embedding).values)
train_grades = train_df[grade_column].values

print("Training model...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(train_embeddings)

mlp = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
mlp.fit(X_train_scaled, train_grades)

print("Generating embeddings for prediction data...")
predict_embeddings = np.stack(predict_df[remarks_column].apply(get_embedding).values)
X_predict_scaled = scaler.transform(predict_embeddings)

print("Predicting grades...")
predicted_grades = np.rint(mlp.predict(X_predict_scaled)).astype(int)

# Fill predicted grades back into original Grade column for rows 29 onward
df.loc[predict_df.index, grade_column] = predicted_grades

# Save updated dataframe to a new Excel file
output_path = file_path.replace(".xlsx", "_with_filled_grades.xlsx")
df.to_excel(output_path, index=False)

print(f"Done! Grades updated starting from row 30 in the original '{grade_column}' column.\nSaved at:\n{output_path}")
