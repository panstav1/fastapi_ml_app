import sys
sys.path.append('./')
import constants
import pytest
import os
import numpy as np
from ml.data import process_data
from ml.model import compute_model_metrics, train_model, load_model, save_model
import pandas as pd


def test_process_data():
    # Create a sample DataFrame
    data = pd.DataFrame({
        'cat_feature': ['A', 'B', 'A'],
        'num_feature': [1, 2, 3],
        'label': [0, 1, 1]
    })

    # Define the categorical feature and label
    cat_features = ['cat_feature']
    label = 'label'

    # Process the data
    X, y, encoder, lb = process_data(data, cat_features, label, training=True)

    # Check if the output shapes are correct
    assert X.shape == (3, 3)  # Adjust based on your expectation
    assert y.shape == (3,)

    # Check if categorical encoding is correct
    assert np.array_equal(encoder.categories_[0], np.array(['A', 'B']))

    # Check if label binarization is correct
    assert set(y) == {0, 1}

def test_compute_model_metrics():
    # Mock ground truth and predictions
    y_ground = np.array([0, 1, 1, 0, 1])
    preds = np.array([0, 1, 0, 0, 1])

    # Compute metrics
    precision, recall, fbeta = compute_model_metrics(y_ground, preds)

    # Expected results (calculated manually or using a trusted function)
    expected_precision = 1.0  # 2 true positives, 0 false positives
    expected_recall = 0.66    # 2 true positives, 1 false negative (approximated)
    expected_fbeta = 0.80     # F1 score (approximated)

    # Assert the results are as expected
    assert np.isclose(precision, expected_precision, atol=0.01)
    assert np.isclose(recall, expected_recall, atol=0.01)
    assert np.isclose(fbeta, expected_fbeta, atol=0.01)

def test_model_save_load():
    # Assume 'train_model' function and a sample dataset are available
    X_train = np.random.rand(10, 5)
    y_train = np.random.randint(2, size=10)

    # Train a model (replace with actual training function if different)
    model = train_model(X_train, y_train)

    # Save the model
    filename = 'test_model.pkl'
    save_model(model, filename)

    # Load the model
    loaded_model = load_model(filename)

    # Check if the loaded model is not None
    assert loaded_model is not None

    # Clean up: delete the saved model file
    os.remove(filename)

    # Optional: Additional checks can be added here based on model type and properties