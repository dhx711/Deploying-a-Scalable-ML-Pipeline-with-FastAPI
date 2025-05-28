import pytest
import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
from sklearn.ensemble import RandomForestClassifier


# Load a small sample of the dataset
df = pd.read_csv("data/census.csv")
cat_features = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]
df_sample = df.sample(n=100, random_state=42)
X, y, encoder, lb = process_data(
    df_sample, categorical_features=cat_features, label="salary", training=True
)


def test_model_is_random_forest():
    """
    Ensure train_model returns a RandomForestClassifier.
    """
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_inference_shape_matches_labels():
    """
    Check that inference predictions match the shape of input labels.
    """
    model = train_model(X, y)
    preds = inference(model, X)
    assert preds.shape == y.shape


def test_metrics_computation_correctness():
    """
    Ensure compute_model_metrics returns correct values for known inputs.
    """
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert np.isclose(precision, 1.0)
    assert np.isclose(recall, 0.5)
    assert np.isclose(fbeta, 0.6666, atol=1e-2)
