#regression.py

from sklearn.cluster import KMeans
import pandas as pd
from nr14_data_analyser.clustering import Clustering
from nr14_data_analyser.preprocessor import EDA, ETL
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


class Regression:
    def __init__(self):
        pass

    def linear_regression(self, X, y):
        model = LinearRegression()
        model.fit(X, y)
        metrics = self.evaluate_model(model, X, y)
        return model, metrics


    def polynomial_regression(self, X, y, degree=2):
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import make_pipeline
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X, y)
        metrics = self.evaluate_model(model, X, y)
        return model, metrics

    def ridge_regression(self, X, y, alpha=1.0):
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=alpha)
        model.fit(X, y)
        metrics = self.evaluate_model(model, X, y)
        return model, metrics

    def lasso_regression(self, X, y, alpha=1.0):
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=alpha)
        model.fit(X, y)
        metrics = self.evaluate_model(model, X, y)
        return model, metrics
    
    def logistic_regression(self, X, y):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        metrics = self.evaluate_model(model, X, y)
        return model, metrics
    
    def evaluate_model(self, model, X_test, y_test):
        from sklearn.metrics import mean_squared_error, r2_score
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2
    
    def predict(self, model, X):
        return model.predict(X)

class RegressionAnalysis:
    """Wrapper for running regression models with consistent API."""

    def __init__(self):
        self.regression = Regression()
        self.data = None

    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)
        return self.data

    def perform_regression(self, X, y, model_type="linear", **kwargs):
        if model_type == "linear":
            return self.regression.linear_regression(X, y)
        elif model_type == "polynomial":
            degree = kwargs.get("degree", 2)
            return self.regression.polynomial_regression(X, y, degree=degree)
        elif model_type == "ridge":
            alpha = kwargs.get("alpha", 1.0)
            return self.regression.ridge_regression(X, y, alpha=alpha)
        elif model_type == "lasso":
            alpha = kwargs.get("alpha", 1.0)
            return self.regression.lasso_regression(X, y, alpha=alpha)
        elif model_type == "logistic":
            return self.regression.logistic_regression(X, y)
        else:
            raise ValueError(
                "Unsupported model type. Choose from 'linear', 'polynomial', 'ridge', 'lasso', 'logistic'."
            )
