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

    # Select target variable
    target_col = st.selectbox("Select target column (y)", df.columns)
    feature_cols = st.multiselect(
        "Select feature columns (X)", 
        [col for col in df.columns if col != target_col]
    )

    if target_col and feature_cols:
        X = df[feature_cols]
        y = df[target_col]

        model_type = st.selectbox(
            "Choose regression model",
            ["linear", "polynomial", "ridge", "lasso", "logistic"]
        )

        # Params
        kwargs = {}
        if model_type == "polynomial":
            kwargs["degree"] = st.slider("Polynomial degree", 2, 5, 2)
        if model_type in ["ridge", "lasso"]:
            kwargs["alpha"] = st.number_input("Alpha (regularization strength)", 0.01, 10.0, 1.0)

        # Train model
        reg_analysis = RegressionAnalysis()
        model, metrics = reg_analysis.perform_regression(X, y, model_type, **kwargs)

        st.subheader("Model Evaluation")
        st.json(metrics)

        # Predictions
        y_pred = model.predict(X)

        # Plot Actual vs Predicted
        st.subheader("Predicted vs Actual")
        Visualizer.plot_regression_results(y, y_pred)

        # Plot residuals (skip for classification)
        if model_type != "logistic":
            st.subheader("Residuals Plot")
            Visualizer.plot_residuals(y, y_pred)