# General-Data-Analyser

`govdata_analyser` adalah library Python untuk **analisis data umum** yang mencakup preprocessing, exploratory data analysis (EDA), clustering, regresi, klasifikasi, evaluasi model, dan visualisasi.
Library ini dibuat untuk mempermudah proses analisis data pemerintahan maupun data umum dalam workflow berbasis Python.

---

## Fitur Utama

* **Preprocessor**

  * Pembersihan data: missing values, duplikasi, outlier.
  * Encoding kategorikal (one-hot & label encoding).
  * ETL pipeline (Extract, Transform, Load).
  * Visualisasi data dasar.

* **EDA (Exploratory Data Analysis)**

  * Statistik ringkasan.
  * Analisis missing values.
  * Heatmap korelasi.
  * Plotting (histogram, boxplot, scatter, countplot, dll.).

* **Clustering**

  * KMeans
  * DBSCAN
  * Agglomerative Clustering
  * Hierarchical Clustering (dendrogram)
  * (opsional) Fuzzy C-Means

* **Regression & Classification**

  * Model umum berbasis scikit-learn.
  * Evaluasi performa (R², MSE, MAE, accuracy, precision, recall, F1).

* **Evaluator**

  * Skor untuk clustering: Silhouette, Davies-Bouldin, Calinski-Harabasz.
  * Skor untuk regresi dan klasifikasi.

* **Visualiser**

  * Scatter plot, cluster plot, dll.

---

## Instalasi

### Dari GitHub

```bash
pip install git+https://github.com/nurrasyid14/General-Data-Analyser.git#egg=govdata_analyser
```

### Dari lokal (editable mode)

Clone repo:

```bash
git clone https://github.com/nurrasyid14/General-Data-Analyser.git
cd General-Data-Analyser
pip install -e .
```

---

## 🚀 Penggunaan

### 1. Preprocessing

```python
import pandas as pd
from govdata_analyser.preprocessor import Cleaner

# load data
df = pd.read_csv("data.csv")

# cleaning
cleaner = Cleaner(df)
cleaner.inspect()
df_cleaned = cleaner.handle_missing_values(strategy="mean")
df_cleaned = cleaner.remove_duplicates()
```

### 2. Clustering

```python
from govdata_analyser.clustering import Clustering

cluster = Clustering(df_cleaned)
labels, model = cluster.kmeans_clustering(df_cleaned, n_clusters=3)
```

### 3. Evaluasi Clustering

```python
from govdata_analyser.evaluator import Evaluator

evaluator = Evaluator(df_cleaned)
metrics = evaluator.e
```
