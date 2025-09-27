import streamlit as st
import pandas as pd

from govdata_analyser.preprocessor import Cleaner, EDA
from govdata_analyser.clustering import Clustering
from govdata_analyser.regression import Regression
from govdata_analyser.evaluator import Evaluator
from govdata_analyser.visualiser import Visualizer

st.set_page_config(page_title="AST : Government Data Analysis", layout="wide")
st.title("AST : Government Data Analysis")

# -----------------------
# File Upload + Auto Preprocessing
# -----------------------
uploaded_file = st.file_uploader("Upload CSV dataset", type="csv")

if "df" not in st.session_state:
    st.session_state.df = None

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    st.info(f"Loaded raw dataset with shape: {raw_df.shape}")

    cleaner = Cleaner(raw_df)

    try:
        cleaned_df = cleaner.handle_missing_values(strategy="mean")
        cleaned_df = cleaner.remove_duplicates()

        outliers = cleaner.detect_outliers(method="zscore", threshold=3.0)
        if isinstance(outliers, pd.DataFrame) and not outliers.empty:
            cleaned_df = cleaned_df.drop(index=outliers.index, errors="ignore")

        cleaner.cleaned_data = cleaned_df
        cleaned_df = cleaner.encode_categorical(method="onehot")
    except Exception as e:
        st.warning(f"Cleaning pipeline issue: {e}")
        cleaned_df = raw_df.copy()

    st.session_state.df = cleaned_df
    st.success(f"✅ Dataset uploaded and preprocessed. Final shape: {cleaned_df.shape}")

df = st.session_state.df

# -----------------------
# Main App
# -----------------------
if df is not None:
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Data Preview", "EDA", "Clustering", "Regression"]
    )

    # Tab 1: Data Preview
    with tab1:
        st.header("Data Preview (Cleaned)")
        st.dataframe(df.head())
        st.write("### Summary Statistics (after cleaning)")
        st.write(df.describe())

    # Tab 2: EDA
    with tab2:
        st.header("Exploratory Data Analysis")
        try:
            eda = EDA(df)
            st.write("### Missing Values")
            st.bar_chart(df.isnull().sum())
        except Exception as e:
            st.warning(f"EDA utilities failed: {e}")
            st.write(df.describe())

    # Tab 3: Clustering
    with tab3:
        st.header("Clustering")
        clustering = Clustering(df)
        visualiser = Visualizer()

        algo = st.selectbox("Choose Clustering Algorithm", ["KMeans", "DBSCAN", "Agglomerative"])

        if algo == "KMeans":
            n_clusters = st.slider("Number of clusters", 2, 10, 3)
            if st.button("Run KMeans"):
                labels, model = clustering.kmeans_clustering(df, n_clusters=n_clusters)
                evaluator = Evaluator(df)  # ✅ fix
                metrics = evaluator.evaluate_clustering(df, labels)
                st.write("### Evaluation Metrics", metrics)
                visualiser.plot_clusters(df, labels, centers=model.cluster_centers_)
                visualiser.silhouette_plot(df, labels)

        elif algo == "DBSCAN":
            eps = st.slider("eps", 0.1, 5.0, 0.5)
            min_samples = st.slider("min_samples", 2, 20, 5)
            if st.button("Run DBSCAN"):
                labels, model = clustering.dbscan_clustering(df, eps=eps, min_samples=min_samples)
                evaluator = Evaluator(df)  # ✅ fix
                metrics = evaluator.evaluate_clustering(df, labels)
                st.write("### Evaluation Metrics", metrics)
                visualiser.plot_clusters(df, labels)
                visualiser.silhouette_plot(df, labels)

        elif algo == "Agglomerative":
            n_clusters = st.slider("Number of clusters", 2, 10, 3)
            if st.button("Run Agglomerative"):
                from sklearn.cluster import AgglomerativeClustering
                agg = AgglomerativeClustering(n_clusters=n_clusters)
                labels = agg.fit_predict(df)
                evaluator = Evaluator(df)  # ✅ fix
                metrics = evaluator.evaluate_clustering(df, labels)
                st.write("### Evaluation Metrics", metrics)
                visualiser.plot_clusters(df, labels)
                visualiser.silhouette_plot(df, labels)

    # Tab 4: Regression
    with tab4:
        st.header("Regression")
        regression = Regression()
        visualiser = Visualizer()

        target_col = st.selectbox("Select Target Variable", df.columns)
        X = df.drop(columns=[target_col])
        y = df[target_col]

        algo = st.selectbox("Choose Regression Algorithm", ["Linear", "Polynomial", "Ridge"])

        if algo == "Linear":
            if st.button("Run Linear Regression"):
                model, _ = regression.linear_regression(X, y)
                evaluator = Evaluator((X, y))  # ✅ fix
                metrics = evaluator.evaluate_regression(model, X, y)
                st.write("### Evaluation Metrics", metrics)
                visualiser.plot_regression_results(y, model.predict(X))
                visualiser.plot_residuals(y, model.predict(X))

        elif algo == "Polynomial":
            degree = st.slider("Degree", 2, 5, 2)
            if st.button("Run Polynomial Regression"):
                model, _ = regression.polynomial_regression(X, y, degree=degree)
                evaluator = Evaluator((X, y))  # ✅ fix
                metrics = evaluator.evaluate_regression(model, X, y)
                st.write("### Evaluation Metrics", metrics)
                visualiser.plot_regression_results(y, model.predict(X))
                visualiser.plot_residuals(y, model.predict(X))

        elif algo == "Ridge":
            alpha = st.slider("Alpha", 0.1, 10.0, 1.0)
            if st.button("Run Ridge Regression"):
                model, _ = regression.ridge_regression(X, y, alpha=alpha)
                evaluator = Evaluator((X, y))  # ✅ fix
                metrics = evaluator.evaluate_regression(model, X, y)
                st.write("### Evaluation Metrics", metrics)
                visualiser.plot_regression_results(y, model.predict(X))
                visualiser.plot_residuals(y, model.predict(X))
