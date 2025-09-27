import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from typing import Optional, Union
from govdata_analyser.preprocessor import Cleaner
from govdata_analyser.clustering import Clustering
from govdata_analyser.evaluator import Evaluator
from govdata_analyser.regression import Regression, RegressionAnalysis


class Visualizer:
    """Utility class for dataset visualization: correlation, PCA, t-SNE, clusters, regression."""

    @staticmethod
    def plot_correlation_matrix(
        data: pd.DataFrame,
        title: str = "Correlation Matrix",
        method: str = "pearson",
        cmap: str = "coolwarm",
        figsize: tuple = (10, 8)
    ) -> None:
        corr = data.corr(method=method, numeric_only=True)
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, square=True, ax=ax)
        ax.set_title(title)
        st.pyplot(fig)
        plt.close(fig)

    @staticmethod
    def _reduce_and_plot(
        data: pd.DataFrame,
        reducer,
        title: str,
        xlabel: str,
        ylabel: str,
        labels: Optional[Union[pd.Series, np.ndarray]] = None,
        figsize: tuple = (8, 6),
        alpha: float = 0.7,
        cmap: str = "viridis"
    ) -> None:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        reduced = reducer.fit_transform(scaled_data)

        fig, ax = plt.subplots(figsize=figsize)
        if labels is not None:
            scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap=cmap, alpha=alpha)
            fig.colorbar(scatter, ax=ax)
        else:
            ax.scatter(reduced[:, 0], reduced[:, 1], alpha=alpha)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        st.pyplot(fig)
        plt.close(fig)

    @staticmethod
    def plot_pca(
        data: pd.DataFrame,
        n_components: int = 2,
        title: str = "PCA Plot",
        labels: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> None:
        reducer = PCA(n_components=n_components)
        Visualizer._reduce_and_plot(
            data, reducer,
            title=title,
            xlabel="PCA Component 1",
            ylabel="PCA Component 2",
            labels=labels,
            **kwargs
        )

    @staticmethod
    def plot_tsne(
        data: pd.DataFrame,
        n_components: int = 2,
        perplexity: float = 30,
        title: str = "t-SNE Plot",
        labels: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> None:
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        Visualizer._reduce_and_plot(
            data, reducer,
            title=title,
            xlabel="t-SNE Component 1",
            ylabel="t-SNE Component 2",
            labels=labels,
            **kwargs
        )

    # ---------- Expert Graphs ----------

    @staticmethod
    def plot_clusters(X: pd.DataFrame, labels: np.ndarray, centers: Optional[np.ndarray] = None) -> None:
        fig, ax = plt.subplots(figsize=(6, 4))
        if X.shape[1] >= 2:
            ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap="viridis", s=30, alpha=0.7)
            if centers is not None:
                ax.scatter(centers[:, 0], centers[:, 1], c="red", marker="x", s=100)
            ax.set_title("Cluster Visualization")
            ax.set_xlabel(X.columns[0])
            ax.set_ylabel(X.columns[1])
            st.pyplot(fig)
        else:
            st.warning("Cluster plot needs at least 2 features.")
        plt.close(fig)

    @staticmethod
    def silhouette_plot(X: pd.DataFrame, labels: np.ndarray) -> None:
        if len(set(labels)) < 2 or -1 in set(labels):
            st.info("Silhouette plot requires at least 2 clusters without pure noise.")
            return

        silhouette_avg = silhouette_score(X, labels)
        sample_values = silhouette_samples(X, labels)

        fig, ax = plt.subplots(figsize=(6, 4))
        y_lower = 10
        for i in np.unique(labels):
            ith = sample_values[labels == i]
            ith.sort()
            size = ith.shape[0]
            y_upper = y_lower + size
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith)
            y_lower = y_upper + 10
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax.set_title("Silhouette Plot")
        st.pyplot(fig)
        plt.close(fig)

    @staticmethod
    def plot_regression_results(y_true: np.ndarray, y_pred: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(y_true, y_pred, alpha=0.7)
        ax.plot([y_true.min(), y_true.max()],
                [y_true.min(), y_true.max()],
                color="red", linestyle="--")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Predicted vs Actual")
        st.pyplot(fig)
        plt.close(fig)

    @staticmethod
    def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> None:
        residuals = y_true - y_pred
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(y_pred, residuals, alpha=0.7)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals Plot")
        st.pyplot(fig)
        plt.close(fig)
