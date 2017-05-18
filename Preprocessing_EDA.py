import pandas as pd
from sklearn import decomposition
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

training_data = pd.read_csv('numerai_training_data.csv', header=0)
prediction_data = pd.read_csv('numerai_tournament_data.csv', header=0)

# Transform the loaded CSV data into numpy arrays
features = [f for f in list(training_data) if "feature" in f]
X = training_data[features]
Y = training_data["target"]
tournament = prediction_data[features]
ids = prediction_data["id"]

# PCA intrinsic dimensions
pca = decomposition.PCA()
pca.fit(X)
components = range(pca.n_components_)
plt.bar(components, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()

# TSNE Visualization
tsne = TSNE(learning_rate=200)
tsne_data = tsne.fit_transform(X=X)
tsne_x = tsne_data[:, 0]
tsne_y = tsne_data[:, 1]
plt.scatter(tsne_x, tsne_y, c=Y)
plt.show()
