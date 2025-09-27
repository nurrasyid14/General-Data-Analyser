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
# File Upload + Auto Preprocessing (uses existing Cleaner methods)
# -----------------------
uploaded_file = st.file_uploader("Upload CSV dataset", type="csv")

if "df" not in st.session_state:
    st.session_state.df = None

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    st.info(f"Loaded raw dataset with shape: {raw_df.shape}")

    cleaner = Cleaner(raw_df)

    # 1) Fill missing values (default: mean)
    try:
        cleaned_df = cleaner.handle_missing_values(strategy="mean")
        st.write("Missing values handled (strategy='mean').")
    except Exception as e:
        st.warning(f"handle_missing_values failed: {e}. Using raw data as fallback.")
        cleaned_df = raw_df.copy()

    # 2) Remove duplicates
    try:
        # cleaner.remove_duplicates uses cleaner.cleaned_data if present
        cleaned_df = cleaner.remove_duplicates()
        st.write("Duplicate rows removed.")
    except Exception as e:
        st.warning(f"remove_duplicates failed: {e}. Continuing with current dataframe.")
        # ensure cleaned_df exists
        if cleaned_df is None:
            cleaned_df = raw_df.copy()

    # 3) Detect outliers (detect only) and remove them from cleaned_df
    outliers_removed = 0
    try:
        outliers = cleaner.detect_outliers(method="zscore", threshold=3.0)
        if isinstance(outliers, pd.DataFrame) and not outliers.empty:
            # drop outliers by index from the current cleaned_df
            before = len(cleaned_df)
            cleaned_df = cleaned_df.drop(index=outliers.index, errors="ignore")
            outliers_removed = before - len(cleaned_df)
            st.write(f"Detected {len(outliers)} outlier rows and removed {outliers_removed}.")
        else:
            st.write("No outliers detected (or detect_outliers returned an empty DataFrame).")
    except Exception as e:
        st.info(f"detect_outliers skipped due to: {e}")

    # Make sure cleaner's internal state matches the cleaned_df before encoding
    try:
        cleaner.cleaned_data = cleaned_df
    except Exception:
        # If Cleaner does not allow assignment (unlikely), proceed without setting it
        pass

    # 4) Encode categorical columns (one-hot by default)
    try:
        cleaned_df = cleaner.encode_categorical(method="onehot")
        st.write("Categorical columns encoded (one-hot).")
    except Exception as e:
        st.warning(f"encode_categorical failed: {e}. Using current cleaned_df as-is.")

    st.success(f"✅ Dataset uploaded and preprocessed. Final shape: {cleaned_df.shape}")
    st.session_state.df = cleaned_df

df = st.session_state.df

# -----------------------
# Main App (Data Preview, EDA, Clustering, Regression, Visualization)
# -----------------------
if df is not None:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Data Preview", "EDA", "Clustering", "Regression", "Visualization"]
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
            st.warning(f"EDA utilities failed to initialize: {e}")
            st.write("Showing basic stats instead:")
            st.write(df.describe())

    # Tab 3: Clustering
    with tab3:
        st.header("Clustering")
        clustering = Clustering(df)
        evaluator = Evaluator()
        metrics = evaluator.evaluate_clustering(df, labels)
        visualiser = Visualizer()

        algo = st.selectbox("Choose Clustering Algorithm", ["KMeans", "DBSCAN", "Agglomerative"])

        if algo == "KMeans":
            n_clusters = st.slider("Number of clusters", 2, 10, 3)
            if st.button("Run KMeans"):
                labels, model = clustering.kmeans_clustering(df, n_clusters=n_clusters)
                metrics = evaluator.evaluate_clustering(df, labels)
                st.write("### Evaluation Metrics", metrics)
                # guard cluster_centers_ presence
                centers = getattr(model, "cluster_centers_", None)
                visualiser.plot_clusters(df, labels, centers=centers)

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
                labels, model = clustering.hierarchical_clustering(df, method="ward")
                # hierarchical_clustering returns linkage Z; transform to labels if needed
                try:
                    # try to fit Agglomerative to get labels as well
                    from sklearn.cluster import AgglomerativeClustering
                    agg = AgglomerativeClustering(n_clusters=n_clusters)
                    labels = agg.fit_predict(df)
                except Exception:
                    labels = None
                if labels is not None:
                    metrics = evaluator.evaluate_clustering(df, labels)
                    st.write("### Evaluation Metrics", metrics)
                    visualiser.plot_clusters(df, labels)
                else:
                    st.info("Dendrogram generated, but labels unavailable for metrics/plot.")
                    Z = model  # model name is actually Z if returned from hierarchical_clustering
                    st.pyplot(clustering.plot_dendrogram(Z))

    # Tab 4: Regression
    with tab4:
        st.header("Regression")
        regression = Regression()
        evaluator = Evaluator()
        metrics = evaluator.evaluate_regression(y, model.predict(X))

        # select target column
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
