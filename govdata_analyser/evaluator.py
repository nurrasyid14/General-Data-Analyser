from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import numpy as np
from typing import Optional

class Evaluator:
    def __init__(self, data):
        self.data = data

    # -------- Clustering Evaluation --------
    def evaluate_clustering(self, X, labels):
        metrics = {}
        # Remove noise label (-1) if using DBSCAN
        valid_labels = np.array(labels)
        if -1 in valid_labels:
            valid_labels = valid_labels[valid_labels != -1]

        n_clusters = len(np.unique(valid_labels))
        
        if n_clusters < 2:
            # Not enough clusters to compute any metrics
            metrics["silhouette"] = None
            metrics["davies_bouldin"] = None
            metrics["calinski_harabasz"] = None
            return metrics

        # Safe to compute metrics on original labels (including noise if desired)
        metrics["silhouette"] = silhouette_score(X, labels)
        metrics["davies_bouldin"] = davies_bouldin_score(X, labels)
        metrics["calinski_harabasz"] = calinski_harabasz_score(X, labels)
        return metrics

    # -------- Regression Evaluation --------
    def evaluate_regression(self, model, X, y):
        preds = model.predict(X)
        metrics = {
            "r2": r2_score(y, preds),
            "mse": mean_squared_error(y, preds),
            "mae": mean_absolute_error(y, preds)
        }
        return metrics

    # -------- Classification Evaluation --------
    def evaluate_classification(self, model, X, y):
        preds = model.predict(X)
        metrics = {
            "accuracy": accuracy_score(y, preds),
            "precision": precision_score(y, preds, average="weighted", zero_division=0),
            "recall": recall_score(y, preds, average="weighted", zero_division=0),
            "f1": f1_score(y, preds, average="weighted", zero_division=0)
        }
        return metrics
    def evaluate(self, model, X, y, task_type: Optional[str] = None):
        if task_type == "regression":
            return self.evaluate_regression(model, X, y)
        elif task_type == "classification":
            return self.evaluate_classification(model, X, y)
        else:
            raise ValueError("task_type must be either 'regression' or 'classification'")
        return metrics