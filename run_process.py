import pandas as pd
from ml.data import process_data

# Load data
df = pd.read_csv("data/census.csv")

# Define categorical columns (as per data dictionary)
cat_features = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]

# Run process_data
X, y, encoder, lb = process_data(
    df, categorical_features=cat_features, label="salary", training=True
)

print("Processed X shape:", X.shape)
print("Processed y shape:", y.shape)
print("Sample y labels:", y[:5])
