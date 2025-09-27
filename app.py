import streamlit as st
import pandas as pd

from govdata_analyser.clustering import Clustering
from govdata_analyser.regression import Regression
from govdata_analyser.visualiser import Visualizer
from govdata_analyser.evaluator import Evaluator
from govdata_analyser.preprocessor import EDA, ETL  

st.set_page_config(page_title="General Data Analyser", layout="wide")

st.title("📊 General Data Analyser")
st.markdown("Upload one or more datasets and perform **EDA, Clustering, and Regression** with interactive visualizations.")

# 🔹 Multiple file uploader
uploaded_files = st.file_uploader(
    "Upload CSV files", type="csv", accept_multiple_files=True
)

datasets = {}

if uploaded_files:
    for file in uploaded_files:
        try:
            df = pd.read_csv(file)
            datasets[file.name] = df.copy()   # ✅ FIX: no Cleaner.clean() needed
        except Exception as e:
            st.error(f"Error loading {file.name}: {e}")

    # 🔹 Continue button
    if st.button("Continue"):
        dataset_name = st.selectbox("Choose a dataset", list(datasets.keys()))
        df = datasets[dataset_name]
        st.success(f"Using dataset: **{dataset_name}**")

        # 🔹 Tabs for workflow
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Data Preview", "EDA", "Clustering", "Regression"]
        )

        # --- Tab 1: Preview ---
        with tab1:
            st.header(f"Data Preview ({dataset_name})")
            st.dataframe(df.head())
            st.write("Shape:", df.shape)

        # --- Tab 2: EDA ---
        with tab2:
            st.header(f"Exploratory Data Analysis ({dataset_name})")
            eda = EDA(df)
            visualiser = Visualizer()
            if st.checkbox("Show Correlation Matrix"):
                visualiser.plot_correlation_matrix(df)
            if st.checkbox("Show PCA Projection"):
                visualiser.plot_pca(df)
            if st.checkbox("Show t-SNE Projection"):
                visualiser.plot_tsne(df)

        # --- Tab 3: Clustering ---
        with tab3:
            st.header(f"Clustering ({dataset_name})")
            algo = st.selectbox("Choose clustering algorithm", ["KMeans", "DBSCAN", "Agglomerative"])
            clustering = Clustering(df)

            if algo == "KMeans":
                n_clusters = st.slider("Number of clusters", 2, 10, 3)
                model, labels = clustering.kmeans(n_clusters=n_clusters)
                evaluator = Evaluator(df)
                metrics = evaluator.evaluate_clustering(df, labels)
                st.write("### Evaluation Metrics", metrics)
                visualiser.plot_clusters(df, labels, centers=model.cluster_centers_)
                visualiser.silhouette_plot(df, labels)

            elif algo == "DBSCAN":
                eps = st.slider("Epsilon", 0.1, 10.0, 0.5)
                min_samples = st.slider("Min Samples", 2, 20, 5)
                model, labels = clustering.dbscan(eps=eps, min_samples=min_samples)
                evaluator = Evaluator(df)
                metrics = evaluator.evaluate_clustering(df, labels)
                st.write("### Evaluation Metrics", metrics)
                visualiser.plot_clusters(df, labels)
                visualiser.silhouette_plot(df, labels)

            elif algo == "Agglomerative":
                n_clusters = st.slider("Number of clusters", 2, 10, 3)
                model, labels = clustering.hierarchical_clustering(n_clusters=n_clusters)
                evaluator = Evaluator(df)
                metrics = evaluator.evaluate_clustering(df, labels)
                st.write("### Evaluation Metrics", metrics)
                visualiser.plot_clusters(df, labels)
                visualiser.silhouette_plot(df, labels)

        # --- Tab 4: Regression ---
        with tab4:
            st.header(f"Regression ({dataset_name})")
            target = st.selectbox("Select target column", df.columns)
            features = st.multiselect(
                "Select feature columns", [c for c in df.columns if c != target]
            )

            if features and target:
                X = df[features].values
                y = df[target].values

                algo = st.selectbox("Choose regression algorithm", ["Linear", "Polynomial", "Ridge", "Lasso", "Logistic"])
                regression = Regression()
                visualiser = Visualizer()

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
                    visualiser.plot_regression_results(y, model.predict(X))
                    visualiser.plot_residuals(y, model.predict(X))
else:
    st.info("Please upload at least one dataset to continue.")
