import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import train

def test_training_accuracy():
    acc = train.train_model()
    assert acc >= 0.9, f"Accuracy too low: {acc}"

def test_model_prediction():
    model = joblib.load("model.pkl")
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
    assert set(preds).issubset({0,1,2})
