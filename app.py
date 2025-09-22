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

# Persist data
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
        ["Data & Cleaning", "EDA", "Clustering", "Regression / Classification", "Visualization"]
    )

    # -----------------------
    # Tab 1: Data & Cleaning
    # -----------------------
    with tab1:
        st.header("Data & Cleaning")
        st.dataframe(df.head())

        cleaner = Cleaner(df)

        st.subheader("Handle Missing Values")
        strategy = st.selectbox("Strategy", ["mean", "median", "mode", "constant"])
        fill_value = None
        if strategy == "constant":
            fill_value = st.text_input("Fill Value")

        if st.button("Apply Missing Value Handling"):
            st.session_state.df = cleaner.handle_missing_values(strategy=strategy, fill_value=fill_value)
            st.success("✅ Missing values handled!")
            st.rerun()

        if st.button("Remove Duplicates"):
            st.session_state.df = cleaner.remove_duplicates()
            st.success("✅ Duplicates removed!")
            st.rerun()

        st.subheader("Encode Categorical Features")
        encoding = st.selectbox("Encoding Method", ["onehot", "label"])
        if st.button("Apply Encoding"):
            st.session_state.df = cleaner.encode_categorical(method=encoding)
            st.success("✅ Encoding applied!")
            st.rerun()

        st.subheader("Detect Outliers")
        method = st.selectbox("Method", ["zscore", "iqr"])
        if st.button("Find Outliers"):
            outliers = cleaner.detect_outliers(method=method)
            st.write(outliers)

    # -----------------------
    # Tab 2: EDA
    # -----------------------
    with tab2:
        st.header("Exploratory Data Analysis (EDA)")
        eda = EDA(df)

        # Summary statistics
        st.subheader("Summary Statistics")
        summary_df = eda.summary_statistics()
        st.write(summary_df)
        csv_summary = summary_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Summary CSV", csv_summary, "summary_statistics.csv", "text/csv")

        # Missing values
        st.subheader("Missing Values")
        missing_df = eda.missing_values()
        st.write(missing_df)
        csv_missing = missing_df.to_csv(index=True).encode('utf-8')
        st.download_button("📥 Download Missing Values CSV", csv_missing, "missing_values.csv", "text/csv")

        # Correlation matrix
        st.subheader("Correlation Matrix")
        if st.button("Show Correlation Matrix"):
            fig = eda.correlation_matrix()
            st.pyplot(fig)

        # Custom plot
        st.subheader("Custom Plot")
        plot_type = st.selectbox("Plot Type", ["hist", "box", "scatter", "count", "bar", "violin", "line"])
        x_col = st.selectbox("X Column", df.columns)
        y_col = None
        if plot_type in ["scatter", "bar", "violin", "line"]:
            y_col = st.selectbox("Y Column", df.columns)
        if st.button("Generate Plot"):
            fig = eda.plot(kind=plot_type, x=x_col, y=y_col)
            st.pyplot(fig)

    # -----------------------
    # Tab 3: Clustering
    # -----------------------
    with tab3:
        st.header("Clustering")
        num_df = df.select_dtypes(include="number").dropna()
        if num_df.shape[1] < 2:
            st.warning("⚠️ Need at least 2 numeric columns for clustering.")
        else:
            cluster = Clustering(num_df)
            evaluator = Evaluator(df)

            algo = st.selectbox("Clustering Algorithm", ["KMeans", "DBSCAN", "Agglomerative", "Hierarchical"])
            labels, model = None, None

            if algo == "KMeans":
                k = st.slider("Number of Clusters", 2, 10, 3)
                labels, model = cluster.kmeans_clustering(num_df, n_clusters=k)

            elif algo == "DBSCAN":
                eps = st.slider("Epsilon", 0.1, 10.0, 0.5)
                min_samples = st.slider("Min Samples", 1, 10, 5)
                labels, model = cluster.dbscan_clustering(num_df, eps=eps, min_samples=min_samples)

            elif algo == "Agglomerative":
                k = st.slider("Number of Clusters", 2, 10, 3)
                labels, model = cluster.agglomerative_clustering(num_df, n_clusters=k)

            elif algo == "Hierarchical":
                Z = cluster.hierarchical_clustering(num_df)
                fig = cluster.plot_dendrogram(Z)
                st.pyplot(fig)

            if labels is not None:
                st.subheader("Clustering Evaluation")
                scores = evaluator.evaluate_clustering(num_df, labels)
                for metric, value in scores.items():
                    st.metric(metric.replace("_"," ").title(), "N/A" if value is None else f"{value:.3f}")

                st.subheader("🗂️ Clustered Data Preview")
                df["cluster"] = labels
                st.dataframe(df.head(20))

    # -----------------------
    # Tab 4: Regression / Classification
    # -----------------------
    with tab4:
        st.header("Regression / Classification")
        reg = Regression()
        evalr = Evaluator(df)

        target = st.selectbox("Select Target Column", df.columns)
        features = df.drop(columns=[target])
        X = features.select_dtypes(include="number")
        y = df[target]

        if X.empty:
            st.warning("⚠️ No numeric features available for regression/classification.")
        else:
            model_type = st.selectbox("Choose Model", ["Linear", "Polynomial", "Ridge", "Lasso", "Logistic"])
            metrics = {}

            if model_type == "Linear":
                model, _ = reg.linear_regression(X, y)
                metrics = evalr.evaluate_regression(model, X, y)
            elif model_type == "Polynomial":
                degree = st.slider("Degree", 2, 5, 2)
                model, _ = reg.polynomial_regression(X, y, degree=degree)
                metrics = evalr.evaluate_regression(model, X, y)
            elif model_type == "Ridge":
                alpha = st.slider("Alpha", 0.1, 10.0, 1.0)
                model, _ = reg.ridge_regression(X, y, alpha=alpha)
                metrics = evalr.evaluate_regression(model, X, y)
            elif model_type == "Lasso":
                alpha = st.slider("Alpha", 0.1, 10.0, 1.0)
                model, _ = reg.lasso_regression(X, y, alpha=alpha)
                metrics = evalr.evaluate_regression(model, X, y)
            elif model_type == "Logistic":
                model, _ = reg.logistic_regression(X, y)
                metrics = evalr.evaluate_classification(model, X, y)

            st.subheader("Model Evaluation Metrics")
            for metric_name, value in metrics.items():
                st.metric(metric_name.replace("_"," ").title(), f"{value:.3f}")

    # -----------------------
    # Tab 5: Visualization
    # -----------------------
    with tab5:
        st.header("Visualization (PCA & t-SNE)")
        vis = Visualizer()
        num_df = df.select_dtypes(include="number").dropna()

        if num_df.shape[1] < 2:
            st.warning("⚠️ Need at least 2 numeric columns for visualization.")
        else:
            if st.button("PCA"):
                fig = vis.plot_pca(num_df, labels=df.get("cluster"))
                st.pyplot(fig)

            if st.button("t-SNE"):
                fig = vis.plot_tsne(num_df, labels=df.get("cluster"))
                st.pyplot(fig)

else:
    st.warning("⚠️ Please upload a CSV file to begin.")
