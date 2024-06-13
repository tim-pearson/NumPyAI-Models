import numpy as np
import matplotlib.pyplot as plt

# import the dataset we have saved in the current directory
url = "data/iris.data"
data = np.genfromtxt(url, delimiter=",", dtype="float", usecols=(0, 1, 2, 3))


class StandardScaler:
    def fit_transform(self, X):
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        return (X - self.means) / self.stds


scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

cov_matrix = np.cov(data_scaled, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
# the eigenvalues are ordered and the corresponding eigenvectors are ordered accorningly

total_variance = np.sum(eigenvalues)
explained_variance_ratio = eigenvalues / total_variance
cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)

threshold = 0.95
num_components = np.argmax(cumulative_explained_variance_ratio >= threshold) + 1
principal_components = eigenvectors[:, :num_components]
data_projected = data_scaled.dot(principal_components)


plt.figure(figsize=(8, 6))
colors = ["r", "g", "b"]
class_labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

for i in range(3):
    plt.scatter(
        x=data_projected[i * 50 : (i + 1) * 50, 0],
        y=data_projected[i * 50 : (i + 1) * 50, 1],
        color=colors[i],
        label=class_labels[i],
    )
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.title("Projection of the Dataset onto the 2 Principle components")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# histogram plot for each attribute in the orinal data
# bins specify the number of intervals to split the data
plt.figure(figsize=(8, 6))
plt.hist(
    data,
    bins=10,
    alpha=0.7,
    label=["sepal_length", "sepal_width", "petal_length", "petal_width"],
)
plt.legend(loc="upper right")
plt.grid(True)
plt.show()
