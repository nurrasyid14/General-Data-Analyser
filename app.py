import streamlit as st
import pandas as pd

from govdata_analyser.preprocessor import Cleaner, EDA
from govdata_analyser.clustering import Clustering
from govdata_analyser.regression import Regression
from govdata_analyser.evaluator import Evaluator
from govdata_analyser.visualiser import Visualizer

# -----------------------
# Streamlit App Config
# -----------------------
st.set_page_config(page_title="AST : Government Data Analysis", layout="wide")
st.title("AST : Government Data Analysis")

# -----------------------
# File Upload
# -----------------------
uploaded_file = st.file_uploader("Upload CSV dataset", type="csv")

if "df" not in st.session_state:
    st.session_state.df = None

if uploaded_file:
    st.session_state.df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")

df = st.session_state.df

# -----------------------
# Main App
# -----------------------
if df is not None:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Data & Cleaning", "EDA", "Clustering", "Regression", "Visualization"]
    )

    # -----------------------
    # Tab 1: Data & Cleaning
    # -----------------------
    with tab1:
        st.header("Data Cleaning")
        st.dataframe(df.head())
        cleaner = Cleaner(df)
        if st.button("Clean Data"):
            df = cleaner.clean()
            st.session_state.df = df
            st.success("✅ Data cleaned!")

    # -----------------------
    # Tab 2: EDA
    # -----------------------
    with tab2:
        st.header("Exploratory Data Analysis")
        eda = EDA(df)
        st.write("### Summary Statistics")
        st.write(df.describe())

        st.write("### Missing Values")
        st.bar_chart(df.isnull().sum())

    # -----------------------
    # Tab 3: Clustering
    # -----------------------
    with tab3:
        st.header("Clustering")
        clustering = Clustering(df)
        evaluator = Evaluator()
        visualiser = Visualizer()

        algo = st.selectbox("Choose Clustering Algorithm", ["KMeans", "DBSCAN", "Agglomerative"])

        if algo == "KMeans":
            n_clusters = st.slider("Number of clusters", 2, 10, 3)
            if st.button("Run KMeans"):
                labels, model = clustering.kmeans_clustering(df, n_clusters=n_clusters)
                metrics = evaluator.evaluate_clustering(df, labels)
                st.write("### Evaluation Metrics", metrics)
                visualiser.plot_clusters(df, labels, centers=model.cluster_centers_)

        elif algo == "DBSCAN":
            eps = st.slider("eps", 0.1, 5.0, 0.5)
            min_samples = st.slider("min_samples", 2, 20, 5)
            if st.button("Run DBSCAN"):
                labels, model = clustering.dbscan_clustering(df, eps=eps, min_samples=min_samples)
                metrics = evaluator.evaluate_clustering(df, labels)
                st.write("### Evaluation Metrics", metrics)
                visualiser.plot_clusters(df, labels)

        elif algo == "Agglomerative":
            n_clusters = st.slider("Number of clusters", 2, 10, 3)
            if st.button("Run Agglomerative"):
                labels, model = clustering.agglomerative_clustering(df, n_clusters=n_clusters)
                metrics = evaluator.evaluate_clustering(df, labels)
                st.write("### Evaluation Metrics", metrics)
                visualiser.plot_clusters(df, labels)

    # -----------------------
    # Tab 4: Regression
    # -----------------------
    with tab4:
        st.header("Regression")
        regression = Regression()
        evaluator = Evaluator()
        visualiser = Visualizer()

        target_col = st.selectbox("Select Target Variable", df.columns)
        X = df.drop(columns=[target_col])
        y = df[target_col]

        algo = st.selectbox("Choose Regression Algorithm", ["Linear", "Polynomial", "Ridge"])

        if algo == "Linear":
            if st.button("Run Linear Regression"):
                model, metrics = regression.linear_regression(X, y)
                st.write("### Evaluation Metrics", metrics)
                visualiser.plot_regression_results(y, model.predict(X))

        elif algo == "Polynomial":
            degree = st.slider("Degree", 2, 5, 2)
            if st.button("Run Polynomial Regression"):
                model, metrics = regression.polynomial_regression(X, y, degree=degree)
                st.write("### Evaluation Metrics", metrics)
                visualiser.plot_regression_results(y, model.predict(X))

        elif algo == "Ridge":
            alpha = st.slider("Alpha", 0.1, 10.0, 1.0)
            if st.button("Run Ridge Regression"):
                model, metrics = regression.ridge_regression(X, y, alpha=alpha)
                st.write("### Evaluation Metrics", metrics)
                visualiser.plot_regression_results(y, model.predict(X))

    # -----------------------
    # Tab 5: Visualization
    # -----------------------
    with tab5:
        st.header("Expert Visualizations")
        st.write("Use this space to review advanced plots (silhouette, residuals, etc.)")
        # Placeholder: can be populated with custom visualiser methods
