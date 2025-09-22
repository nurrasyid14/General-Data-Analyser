# visualiser.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from typing import Optional, Union
from govdata_analyser.preprocessor import Cleaner
from govdata_analyser.clustering import Clustering
from govdata_analyser.evaluator import Evaluator
from govdata_analyser.regression import Regression, RegressionAnalysis

class Visualizer:
    """Utility class for dataset visualization: correlation, PCA, t-SNE."""

    @staticmethod
    def plot_correlation_matrix(
        data: pd.DataFrame,
        title: str = "Correlation Matrix",
        method: str = "pearson",
        cmap: str = "coolwarm",
        figsize: tuple = (10, 8)
    ) -> None:
        """
        Plots a correlation matrix heatmap.

        Parameters:
        - data: Input DataFrame
        - title: Title of the plot
        - method: Correlation method ('pearson', 'spearman', 'kendall')
        - cmap: Colormap
        - figsize: Figure size
        """
        corr = data.corr(method=method, numeric_only=True)
        plt.figure(figsize=figsize)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, square=True)
        plt.title(title)
        plt.show()

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
        """
        Helper method for dimensionality reduction plots (PCA, t-SNE).
        """
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        reduced = reducer.fit_transform(scaled_data)

        plt.figure(figsize=figsize)
        if labels is not None:
            scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap=cmap, alpha=alpha)
            plt.colorbar(scatter)
        else:
            plt.scatter(reduced[:, 0], reduced[:, 1], alpha=alpha)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    @staticmethod
    def plot_pca(
        data: pd.DataFrame,
        n_components: int = 2,
        title: str = "PCA Plot",
        labels: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> None:
        """
        Plots PCA reduced data.
        """
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
        """
        Plots t-SNE reduced data.
        """
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        Visualizer._reduce_and_plot(
            data, reducer,
            title=title,
            xlabel="t-SNE Component 1",
            ylabel="t-SNE Component 2",
            labels=labels,
            **kwargs
        )