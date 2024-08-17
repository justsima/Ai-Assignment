#normalaize iris dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X = iris.data

# Normalize the Iris dataset
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

#  Apply k-means clustering with k=3
def kmeans(X, k, max_iters=1000, plot_steps=False):
    idx = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[idx, :]
    
    # Optimize clusters
    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        if plot_steps:
            plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
            plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.5)
            plt.show()
        
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    return centroids, labels

centroids, labels = kmeans(X_normalized, 3, max_iters=150, plot_steps=False)

# Visualize the results
plt.figure(figsize=(8, 6))
plt.scatter(X_normalized[:, 0], X_normalized[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.5)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering on Normalized Iris Dataset')
plt.show()





#Apply k-means clustering with k=3. 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # we only take the first two features.

# Define the k-means clustering function
def kmeans(X, k, max_iter=100):
    # Initialize centroids randomly
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(max_iter):
        # Assign each data point to the closest centroid
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)

        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, labels

# Apply k-means clustering with k=3
centroids, labels = kmeans(X, k=3)

# Visualize the results
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=200)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('K-means Clustering on Iris Dataset')
plt.show()


#Visualize the cluster assignments and the centroids in 2D and 3D plots.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data

# Define the k-means clustering function
def kmeans(X, k, max_iter=100):
    # Initialize centroids randomly
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(max_iter):
        # Assign each data point to the closest centroid
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)

        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, labels

# Apply k-means clustering with k=3
centroids, labels = kmeans(X, k=3)

# 2D plot
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=200)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('K-means Clustering on Iris Dataset (2D)')
plt.show()

# 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels)
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', marker='*', s=200)
ax.set_xlabel('Sepal Length (cm)')
ax.set_ylabel('Sepal Width (cm)')
ax.set_zlabel('Petal Length (cm)')
ax.set_title('K-means Clustering on Iris Dataset (3D)')
plt.show()



#Compare the cluster assignments with the actual class labels.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Define the k-means clustering function
def kmeans(X, k, max_iter=100):
    # Initialize centroids randomly
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(max_iter):
        # Assign each data point to the closest centroid
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)

        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, labels

# Apply k-means clustering with k=3
centroids, cluster_labels = kmeans(X, k=3)

# Visualize the cluster assignments
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=200)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('K-means Clustering on Iris Dataset')
plt.show()

# Compare the cluster assignments with the actual class labels
print("Adjusted Rand Index:", adjusted_rand_score(y, cluster_labels))

# Visualize the actual class labels
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Actual Class Labels')
plt.show()



# Plot the silhouette scores to evaluate the quality of the clusters.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data

# Apply k-means clustering with k=3
def kmeans(X, k, max_iters=1000, plot_steps=False):
    idx = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[idx, :]
    
    # Optimize clusters
    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # check for convergence
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
        
        # plot steps
        if plot_steps:
            plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
            plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.5)
            plt.show()
    
    return centroids, labels

centroids, labels = kmeans(X, 3)

# Visualize the cluster assignments
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=200)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('K-means Clustering on Iris Dataset')
plt.show()

# Calculate the silhouette scores
def silhouette_score(X, labels, centroids):
    silhouettes = []
    for i in range(X.shape[0]):
        a = np.linalg.norm(X[i] - centroids[labels[i]])
        b = np.min([np.linalg.norm(X[i] - centroids[j]) for j in range(3) if j != labels[i]])
        silhouettes.append((b - a) / max(a, b))
    return silhouettes

silhouette_values = silhouette_score(X, labels, centroids)

# Plot the silhouette scores
plt.figure(figsize=(8, 6))
plt.hist(silhouette_values, bins=50)
plt.xlabel('Silhouette Score')
plt.ylabel('Frequency')
plt.title('Silhouette Score Distribution')
plt.show()

# Plot the silhouette scores for each cluster
silhouette_values_by_cluster = [silhouette_values[labels == i] for i in range(3)]
plt.figure(figsize=(8, 6))
for i, values in enumerate(silhouette_values_by_cluster):
    plt.subplot(1, 3, i+1)
    plt.hist(values, bins=50)
    plt.title(f'Cluster {i+1}')
    plt.xlabel('Silhouette Score')
    plt.ylabel('Frequency')
plt.show()

