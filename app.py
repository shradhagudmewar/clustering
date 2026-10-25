# 1. Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

# 2. Load the Iris dataset
iris = load_iris()
X = iris.data

df = pd.DataFrame(X, columns=iris.feature_names)

print("First five rows of the dataset:")
print(df.head())

print("\nDataset shape:", X.shape)

# 3. Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# 5. Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.8, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# 6. Reduce dimensions to two for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 7. Evaluate clustering results
def evaluate_clustering(name, labels, data):
    # Exclude noise points labelled -1 (relevant to DBSCAN)
    valid = labels != -1
    valid_labels = labels[valid]
    valid_data = data[valid]

    print(f"\n{name} Results")

    n_clusters = len(set(valid_labels))
    n_noise = np.sum(labels == -1)

    print("Number of clusters:", n_clusters)
    print("Number of noise points:", n_noise)

    if n_clusters >= 2 and n_clusters < len(valid_labels):
        sil = silhouette_score(valid_data, valid_labels)
        dbi = davies_bouldin_score(valid_data, valid_labels)

        print("Silhouette Score:", round(sil, 3))
        print("Davies-Bouldin Index:", round(dbi, 3))
    else:
        print("Evaluation metrics cannot be calculated reliably.")

evaluate_clustering("K-Means", kmeans_labels, X_scaled)
evaluate_clustering("DBSCAN", dbscan_labels, X_scaled)

# 8. Visualize the clustering results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=kmeans_labels, cmap="viridis", s=50
)
axes[0].set_title("K-Means Clustering")
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")

axes[1].scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=dbscan_labels, cmap="viridis", s=50
)
axes[1].set_title("DBSCAN Clustering")
axes[1].set_xlabel("Principal Component 1")
axes[1].set_ylabel("Principal Component 2")

plt.tight_layout()
plt.show()
