# A5_clustering.py
"""
Lab07 - A5
Clustering with Hierarchical (Agglomerative) and Density-based (DBSCAN) algorithms

Run:
    python A5_clustering.py

Requirements:
    pip install scikit-learn matplotlib seaborn
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ----------------- Load dataset -----------------
DATA_PATH = "dataset.xlsx"
df = pd.read_excel(DATA_PATH, sheet_name="Sheet1")

# Drop label column if it exists (unsupervised learning)
if "label" in df.columns:
    X = df.drop(columns=["label"])
else:
    X = df.copy()

# Scale features for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------- Hierarchical Clustering -----------------
hier_model = AgglomerativeClustering(n_clusters=2)  # you can try other k values
hier_labels = hier_model.fit_predict(X_scaled)

sil_hier = silhouette_score(X_scaled, hier_labels)
print(f"Hierarchical Clustering Silhouette Score: {sil_hier:.4f}")

# ----------------- DBSCAN -----------------
dbscan_model = DBSCAN(eps=0.8, min_samples=5)  # tune eps & min_samples
dbscan_labels = dbscan_model.fit_predict(X_scaled)

# Filter out noise (-1 label) when calculating silhouette
if len(set(dbscan_labels)) > 1 and -1 not in set(dbscan_labels):
    sil_dbscan = silhouette_score(X_scaled, dbscan_labels)
    print(f"DBSCAN Silhouette Score: {sil_dbscan:.4f}")
else:
    print("DBSCAN produced only 1 cluster or noise, silhouette not defined.")

# ----------------- Save Results -----------------
df_clusters = df.copy()
df_clusters["Hierarchical_Label"] = hier_labels
df_clusters["DBSCAN_Label"] = dbscan_labels

df_clusters.to_csv("A5_clusters.csv", index=False)
print("\nCluster labels saved to A5_clusters.csv")

# ----------------- Visualization -----------------
# Use first 2 features for scatterplot visualization
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1], hue=hier_labels, palette="Set1")
plt.title("Hierarchical Clustering (k=2)")

plt.subplot(1,2,2)
sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1], hue=dbscan_labels, palette="Set2")
plt.title("DBSCAN Clustering")

plt.tight_layout()
plt.savefig("A5_clustering_plots.png")
print("Cluster plots saved to A5_clustering_plots.png")
