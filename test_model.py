import pytest
from src.train import load_data, train_model
import os

def test_load_data():
    data = load_data()
    assert data is not None
    assert len(data.columns) == 5  # 4 features + 1 target

def test_train_model():
    train_model()
    assert os.path.exists('models/model.joblib')