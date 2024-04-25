from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

# loading data
iris = datasets.load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='FlowerType')
print(X.head())

# creating PCA
pca_iris = PCA(n_components=3).fit(iris.data)
# print(pca_iris)

# variance analysis
explained_variance = pca_iris.explained_variance_ratio_
explained_variance_cumulative = explained_variance.cumsum()

# check for min num of components
components_num = (explained_variance_cumulative < 0.95).sum() + 1
print(f"Number of required components to keep 95% variance: {components_num}")


if components_num == 2:
    X_reduced = PCA(n_components=2).fit_transform(X)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis')
    plt.title('PCA of IRIS dataset (2D)')
    plt.show()
elif components_num == 3:
    X_reduced = PCA(n_components=3).fit_transform(X)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, cmap='viridis')
    ax.set_title('PCA of IRIS dataset (3D)')
    plt.show()
