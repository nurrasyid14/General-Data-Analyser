#visualiser.py
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from typing import Optional, Union


class Visualizer:
    """Utility class for dataset visualization."""

    @staticmethod
    def plot_clusters(X, labels, centers=None):
        """
        Plot clusters in 2D using PCA projection.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)

        df_plot = pd.DataFrame(X_2d, columns=["PC1", "PC2"])
        df_plot["Cluster"] = labels

        fig = px.scatter(
            df_plot,
            x="PC1",
            y="PC2",
            color="Cluster",
            opacity=0.7,
            title="Cluster Visualization (PCA Projection)"
        )

        if centers is not None:
            centers_2d = pca.transform(centers)
            centers_df = pd.DataFrame(centers_2d, columns=["PC1", "PC2"])
            fig.add_trace(
                go.Scatter(
                    x=centers_df["PC1"],
                    y=centers_df["PC2"],
                    mode="markers",
                    marker=dict(symbol="x", size=12, color="red"),
                    name="Centers"
                )
            )

        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_correlation_matrix(data: pd.DataFrame, method: str = "pearson") -> None:
        corr = data.corr(method=method, numeric_only=True)
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_pca(data: pd.DataFrame, n_components: int = 2,
                 labels: Optional[Union[pd.Series, np.ndarray]] = None) -> None:
        scaler = StandardScaler()
        reduced = PCA(n_components=n_components).fit_transform(scaler.fit_transform(data))
        df_plot = pd.DataFrame(reduced, columns=[f"PC{i+1}" for i in range(n_components)])
        if labels is not None:
            df_plot["Label"] = labels
            fig = px.scatter(df_plot, x="PC1", y="PC2", color="Label", title="PCA Plot")
        else:
            fig = px.scatter(df_plot, x="PC1", y="PC2", title="PCA Plot")
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_tsne(data: pd.DataFrame, n_components: int = 2, perplexity: float = 30,
                  labels: Optional[Union[pd.Series, np.ndarray]] = None) -> None:
        scaler = StandardScaler()
        reduced = TSNE(n_components=n_components, perplexity=perplexity,
                       random_state=42).fit_transform(scaler.fit_transform(data))
        df_plot = pd.DataFrame(reduced, columns=[f"tSNE{i+1}" for i in range(n_components)])
        if labels is not None:
            df_plot["Label"] = labels
            fig = px.scatter(df_plot, x="tSNE1", y="tSNE2", color="Label", title="t-SNE Plot")
        else:
            fig = px.scatter(df_plot, x="tSNE1", y="tSNE2", title="t-SNE Plot")
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def silhouette_plot(X: pd.DataFrame, labels: np.ndarray) -> None:
        if len(set(labels)) < 2 or -1 in set(labels):
            st.info("Silhouette plot requires at least 2 clusters without pure noise.")
            return
        silhouette_avg = silhouette_score(X, labels)
        sample_values = silhouette_samples(X, labels)
        df_sil = pd.DataFrame({"Silhouette": sample_values, "Cluster": labels})
        fig = px.violin(
            df_sil,
            y="Silhouette",
            x="Cluster",
            box=True,
            points="all",
            title=f"Silhouette Plot (avg={silhouette_avg:.2f})"
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_regression_results(y_true: np.ndarray, y_pred: np.ndarray) -> None:
        df = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
        fig = px.scatter(df, x="Actual", y="Predicted", opacity=0.7, title="Predicted vs Actual")
        fig.add_trace(go.Scatter(
            x=[df["Actual"].min(), df["Actual"].max()],
            y=[df["Actual"].min(), df["Actual"].max()],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Ideal Fit"
        ))
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> None:
        residuals = y_true - y_pred
        df = pd.DataFrame({"Predicted": y_pred, "Residuals": residuals})
        fig = px.scatter(df, x="Predicted", y="Residuals", opacity=0.7, title="Residuals Plot")
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_membership_heatmap(u: np.ndarray) -> None:
        """
        Plot a heatmap of the membership matrix (samples × clusters).
        """
        df_u = pd.DataFrame(u.T, columns=[f"Cluster {i}" for i in range(u.shape[0])])
        fig = px.imshow(
            df_u,
            aspect="auto",
            color_continuous_scale="viridis",
            title="Membership Matrix Heatmap"
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_soft_clusters(X: Union[pd.DataFrame, np.ndarray], u: np.ndarray, cluster_index: int = 0):
        """
        Scatter plot with membership strength as color intensity for one cluster.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)

        df_plot = pd.DataFrame(X_2d, columns=["PC1", "PC2"])
        df_plot["Membership"] = u[cluster_index]

        fig = px.scatter(
            df_plot,
            x="PC1",
            y="PC2",
            color="Membership",
            opacity=0.8,
            color_continuous_scale="viridis",
            title=f"Fuzzy Cluster {cluster_index} Membership"
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def boxplot(data: pd.DataFrame, column: str) -> None:
        fig = px.box(data, y=column, title=f"Boxplot of {column}")
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_distribution(data: pd.DataFrame, column: str) -> None:
        fig = px.histogram(data, x=column, nbins=30, title=f"Distribution of {column}")
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_missing_values_heatmap(data: pd.DataFrame) -> None:
        missing = data.isnull()
        fig = px.imshow(
            missing.T,
            aspect="auto",
            color_continuous_scale="gray",
            title="Missing Values Heatmap"
            )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_pairplot(data: pd.DataFrame) -> None:
        fig = px.scatter_matrix(data, title="Pairplot")
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_box(data: pd.DataFrame, x: str, y: str) -> None:
        fig = px.box(data, x=x, y=y, title=f"Boxplot of {y} by {x}")
        st.plotly_chart(fig, use_container_width=True)

class ClusteringVisualiser:
    """Utility class for visualizing clustering results."""

    @staticmethod
    def plot_clusters(X, labels, centers=None):
        """
        Plot clusters in 2D using PCA projection.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)

        df_plot = pd.DataFrame(X_2d, columns=["PC1", "PC2"])
        df_plot["Cluster"] = labels

        fig = px.scatter(
            df_plot,
            x="PC1",
            y="PC2",
            color="Cluster",
            opacity=0.7,
            title="Cluster Visualization (PCA Projection)"
        )

        if centers is not None:
            centers_2d = pca.transform(centers)
            centers_df = pd.DataFrame(centers_2d, columns=["PC1", "PC2"])
            fig.add_trace(
                go.Scatter(
                    x=centers_df["PC1"],
                    y=centers_df["PC2"],
                    mode="markers",
                    marker=dict(symbol="x", size=12, color="red"),
                    name="Centers"
                )
            )

        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def silhouette_plot(X: pd.DataFrame, labels: np.ndarray) -> None:
        """
        Silhouette violin plot for cluster quality.
        """
        if len(set(labels)) < 2 or -1 in set(labels):
            st.info("Silhouette plot requires at least 2 clusters without pure noise.")
            return

        silhouette_avg = silhouette_score(X, labels)
        sample_values = silhouette_samples(X, labels)
        df_sil = pd.DataFrame({"Silhouette": sample_values, "Cluster": labels})

        fig = px.violin(
            df_sil,
            y="Silhouette",
            x="Cluster",
            box=True,
            points="all",
            title=f"Silhouette Plot (avg={silhouette_avg:.2f})"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Fuzzy C-Means extras

    @staticmethod
    def plot_fuzzy_clusters(X, u: np.ndarray):
        """
        Scatter plot of fuzzy c-means clusters using hard labels (argmax of membership).
        If X has >2 features, reduce with PCA.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Hard labels from membership
        labels = np.argmax(u, axis=0)

        # If data >2D, reduce with PCA
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
            df_plot = pd.DataFrame(X_2d, columns=["PC1", "PC2"])
        else:
            df_plot = X.copy()
            df_plot.columns = ["X1", "X2"]

        df_plot["Cluster"] = labels.astype(str)  # make categorical for clean legend

        # Plot with only color (avoid duplicate legend)
        fig = px.scatter(
            df_plot,
            x=df_plot.columns[0],
            y=df_plot.columns[1],
            color="Cluster",
            title="Fuzzy C-Means Clusters (Hard Labels)",
        )

        # Improve legend placement
        fig.update_layout(
            legend=dict(
                title="Cluster",
                orientation="h",      # horizontal
                yanchor="bottom",
                y=1.02,               # above chart
                xanchor="center",
                x=0.5
            )
        )

        st.plotly_chart(fig, use_container_width=True)



    @staticmethod
    def plot_membership_heatmap(u: np.ndarray):
        """
        Heatmap of membership matrix (clusters × samples) using Plotly.
        """
        # u is shape (n_clusters, n_samples) → transpose so samples on x-axis
        fig = px.imshow(
            u,
            aspect="auto",
            color_continuous_scale="viridis",
            title="Fuzzy Membership Matrix",
        )
        fig.update_xaxes(title="Samples")
        fig.update_yaxes(title="Clusters")
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_soft_clusters(X, u, cluster_id=0):
        """
        Scatter plot showing membership strength for a chosen cluster.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)

        df_plot = pd.DataFrame(X_2d, columns=["PC1", "PC2"])
        df_plot["Membership"] = u[cluster_id]

        fig = px.scatter(
            df_plot,
            x="PC1",
            y="PC2",
            color="Membership",
            color_continuous_scale="viridis",
            title=f"Soft Cluster Visualization (Cluster {cluster_id})",
            opacity=0.8
        )
        fig.update_traces(marker=dict(size=8, line=dict(width=0)))
        st.plotly_chart(fig, use_container_width=True)



__all__ = ["Visualizer", "ClusteringVisualiser"]
