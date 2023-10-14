import unittest
from MLops import random_forest_model, evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot
import pandas


class TestMLFunctions(unittest.TestCase):
    def setUp(self):
        # Load a sample dataset for testing
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def test_random_forest_model(self):
        model = random_forest_model(self.X_train, self.y_train)
        self.assertIsNotNone(model)

    def test_evaluate_model(self):
        model = random_forest_model(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)
        metrics = evaluate_model(self.y_test, y_pred, y_pred_proba)
        self.assertIsNotNone(metrics)
        self.assertIsInstance(metrics, dict)

if __name__ == '__main__':
    unittest.main()
