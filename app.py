# app.py

"""
General Data Analyser Streamlit app

* Upload CSV(s).
* Perform EDA (via EDA/Visualizer).
* Classic clustering (KMeans/DBSCAN/Agglomerative).
* Dedicated Fuzzy C-Means tab (hard labels + membership visualisations).
* Regression (Linear, Polynomial, Ridge, Lasso, Logistic).

Expect these modules to exist in `nr14_data_analyser`:

* clustering.Clustering
* regression.Regression
* visualiser.Visualizer (contains plot_membership_heatmap & plot_soft_clusters)
* evaluator.Evaluator
* preprocessor.EDA, ETL, Cleaner
* fuzzycmeans.FuzzyCMeans (wrapper around skfuzzy.cmeans)
  """

import streamlit as st
import pandas as pd
import numpy as np

from nr14_data_analyser.clustering import Clustering
from nr14_data_analyser.regression import Regression
from nr14_data_analyser.visualiser import Visualizer
from nr14_data_analyser.evaluator import Evaluator
from nr14_data_analyser.preprocessor import EDA, ETL, Cleaner
from nr14_data_analyser.fuzzycmeans import FuzzyCMeans

st.set_page_config(page_title="General Data Analyser", layout="wide")

st.title("General Data Analyser")
st.markdown("Upload one or more datasets and perform **EDA, Clustering, and Regression** with interactive visualizations.")

# -------------------------
# File upload
# -------------------------
uploaded_files = st.file_uploader(
    "Upload CSV files", type="csv", accept_multiple_files=True
)

datasets = {}

if uploaded_files:
    for file in uploaded_files:
        try:
            df = pd.read_csv(file)
            datasets[file.name] = df.copy()
        except Exception as e:
            st.error(f"Error loading {file.name}: {e}")

    # ✅ Track "Continue" in session state
    if "continue_flag" not in st.session_state:
        st.session_state.continue_flag = False

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Continue"):
            st.session_state.continue_flag = True
    with col2:
        if st.button("Reset"):
            st.session_state.continue_flag = False
            st.experimental_rerun()

    if st.session_state.continue_flag:
        dataset_name = st.selectbox("Choose a dataset", list(datasets.keys()))
        df = datasets[dataset_name]
        st.success(f"Using dataset: **{dataset_name}**")

        # 🚀 Show all your tabs here
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Data Preview", "EDA", "Clustering", "Fuzzy C-Means", "Regression"]
        )

        # ... (rest of your tab code unchanged)

else:
    st.info("Please upload at least one dataset to continue.")


    # -------------------------
    # Tab 1: Data Preview
    # -------------------------
    with tab1:
        st.header(f"Data Preview ({dataset_name})")
        st.dataframe(df.head())
        st.write("Shape:", df.shape)

    # -------------------------
    # Tab 2: EDA
    # -------------------------
    with tab2:
        st.header(f"Exploratory Data Analysis ({dataset_name})")
        eda = EDA(df)
        visualiser = Visualizer()

        if st.checkbox("Show Correlation Matrix"):
            visualiser.plot_correlation_matrix(df)

        if st.checkbox("Show PCA Projection"):
            numeric_df = df.select_dtypes(include=["number"])
            if numeric_df.shape[1] < 2:
                st.warning("PCA requires at least 2 numeric features.")
            else:
                visualiser.plot_pca(numeric_df)

        if st.checkbox("Show t-SNE Projection"):
            numeric_df = df.select_dtypes(include=["number"])
            if numeric_df.shape[1] < 2:
                st.warning("t-SNE requires at least 2 numeric features.")
            else:
                visualiser.plot_tsne(numeric_df)

        if st.checkbox("Show Pairplot"):
            try:
                visualiser.plot_pairplot(df)
            except Exception as e:
                st.error(f"Pairplot error: {e}")

        if st.checkbox("Show Missing Values Heatmap"):
            try:
                visualiser.plot_missing_values_heatmap(df)
            except Exception as e:
                st.error(f"Missing values heatmap error: {e}")

        if st.checkbox("Show Distribution Plots"):
            numeric_cols = df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) == 0:
                st.info("No numeric columns for distribution plots.")
            else:
                col = st.selectbox("Select column for distribution plot", numeric_cols)
                visualiser.plot_distribution(df, col)

        if st.checkbox("Show Box Plots"):
            numeric_cols = df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) == 0:
                st.info("No numeric columns for box plots.")
            else:
                col = st.selectbox("Select column for box plot", numeric_cols)
                visualiser.plot_box(df, col)

        if st.checkbox("Show Outliers using IQR"):
            numeric_cols = df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) == 0:
                st.info("No numeric columns for outlier detection.")
            else:
                col = st.selectbox("Select column for outlier detection", numeric_cols)
                outliers = eda.detect_outliers_iqr(col)
                st.write(f"Outliers in {col}:\n", outliers)

        if st.checkbox("Show Summary Statistics"):
            st.write(eda.summary_statistics())

        if st.checkbox("Show Data Types"):
            st.write(eda.data_types())

    # -------------------------
    # Tab 3: Clustering (classic)
    # -------------------------
    with tab3:
        st.header(f"Clustering ({dataset_name})")
        algo = st.selectbox("Choose clustering algorithm", ["KMeans", "DBSCAN", "Agglomerative"])
        numeric_df = df.select_dtypes(include=["number"])

        if numeric_df.shape[0] == 0 or numeric_df.shape[1] == 0:
            st.warning("No numeric columns available for clustering.")
        else:
            clustering = Clustering(numeric_df)
            visualiser = Visualizer()

            if algo == "KMeans":
                n_clusters = st.slider("Number of clusters", 2, 10, 3)
                labels, model = clustering.kmeans_clustering(X=clustering.data, n_clusters=n_clusters)
                evaluator = Evaluator(df)
                metrics = evaluator.evaluate_clustering(clustering.data, labels)
                st.write("### Evaluation Metrics", metrics)
                visualiser.plot_clusters(clustering.data, labels, centers=model.cluster_centers_)
                visualiser.silhouette_plot(clustering.data, labels)

            elif algo == "DBSCAN":
                eps = st.slider("Epsilon", 0.1, 10.0, 0.5)
                min_samples = st.slider("Min Samples", 2, 20, 5)
                labels, model = clustering.dbscan_clustering(X=clustering.data, eps=eps, min_samples=min_samples)
                evaluator = Evaluator(df)
                metrics = evaluator.evaluate_clustering(clustering.data, labels)
                st.write("### Evaluation Metrics", metrics)
                visualiser.plot_clusters(clustering.data, labels)
                visualiser.silhouette_plot(clustering.data, labels)

            elif algo == "Agglomerative":
                n_clusters = st.slider("Number of clusters", 2, 10, 3)
                labels, model = clustering.agglomerative_clustering(X=clustering.data, n_clusters=n_clusters)
                evaluator = Evaluator(df)
                metrics = evaluator.evaluate_clustering(clustering.data, labels)
                st.write("### Evaluation Metrics", metrics)
                visualiser.plot_clusters(clustering.data, labels)
                visualiser.silhouette_plot(clustering.data, labels)

    # -------------------------
    # Tab 4: Fuzzy C-Means
    # -------------------------
    with tab4:
        st.header(f"Fuzzy C-Means Clustering ({dataset_name})")
        numeric_df = df.select_dtypes(include=["number"])
        if numeric_df.shape[0] == 0 or numeric_df.shape[1] == 0:
            st.warning("No numeric columns available for Fuzzy C-Means.")
        else:
            clustering = Clustering(numeric_df)
            visualiser = Visualizer()

            # Parameters
            n_clusters = st.slider("Number of clusters (C)", 2, 10, 3, key="fcm_n_clusters")
            m = st.slider("Fuzziness parameter (m)", 1.5, 3.0, 2.0, 0.1, key="fcm_m")
            error = st.number_input("Error tolerance", min_value=0.0001, value=0.005, step=0.0001, key="fcm_error")
            maxiter = st.number_input("Max iterations", min_value=100, value=1000, step=100, key="fcm_maxiter")

            # ✅ Persistent run button
            if "fcm_run" not in st.session_state:
                st.session_state.fcm_run = False

            if st.button("Run Fuzzy C-Means"):
                st.session_state.fcm_run = True

            if st.session_state.fcm_run:
                try:
                    fcm = FuzzyCMeans(
                        clustering.data,
                        n_clusters=n_clusters,
                        m=m,
                        error=error,
                        maxiter=maxiter
                    )
                    fcm.fit()
                    labels = fcm.predict()

                    # Evaluation
                    evaluator = Evaluator(df)
                    metrics = evaluator.evaluate_clustering(clustering.data, labels)
                    st.write("### Evaluation Metrics", metrics)

                    # Visualisation
                    visualiser.plot_clusters(clustering.data, labels, centers=fcm.centers)
                    visualiser.silhouette_plot(clustering.data, labels)

                    # Membership Matrix
                    if st.checkbox("Show membership matrix"):
                        st.dataframe(
                            pd.DataFrame(
                                fcm.u.T,
                                columns=[f"Cluster {i}" for i in range(fcm.u.shape[0])]
                            )
                        )

                except Exception as e:
                    st.error(f"Fuzzy C-Means failed: {e}")
                    st.session_state.fcm_run = False  # reset on error

    # -------------------------
    # Tab 5: Regression
    # -------------------------
    with tab5:
        st.header(f"Regression ({dataset_name})")
        target = st.selectbox("Select target column", df.columns, index=0)
        feature_options = [c for c in df.columns if c != target]
        features = st.multiselect("Select feature columns", feature_options)

        if features and target:
            X = df[features].values
            y = df[target].values
            regression = Regression()
            visualiser = Visualizer()

            algo = st.selectbox("Choose regression algorithm", ["Linear", "Polynomial", "Ridge", "Lasso", "Logistic"])

            try:
                if algo == "Linear":
                    model, _ = regression.linear_regression(X, y)
                elif algo == "Polynomial":
                    degree = st.slider("Polynomial degree", 2, 5, 2)
                    model, _ = regression.polynomial_regression(X, y, degree=degree)
                elif algo == "Ridge":
                    alpha = st.slider("Alpha (Ridge)", 0.01, 10.0, 1.0)
                    model, _ = regression.ridge_regression(X, y, alpha=alpha)
                elif algo == "Lasso":
                    alpha = st.slider("Alpha (Lasso)", 0.01, 10.0, 1.0)
                    model, _ = regression.lasso_regression(X, y, alpha=alpha)
                elif algo == "Logistic":
                    model, _ = regression.logistic_regression(X, y)

                evaluator = Evaluator((X, y))
                if algo == "Logistic":
                    metrics = evaluator.evaluate_classification(model, X, y)
                else:
                    metrics = evaluator.evaluate_regression(model, X, y)

                st.write("### Evaluation Metrics", metrics)

                if algo != "Logistic":
                    y_pred = model.predict(X)
                    visualiser.plot_regression_results(y, y_pred)
                    visualiser.plot_residuals(y, y_pred)

            except Exception as e:
                st.error(f"Regression failed: {e}")


else:
    st.info("Please upload at least one dataset to continue.")
