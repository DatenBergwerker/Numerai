from sklearn import decomposition
import matplotlib.pyplot as plt

training_data = pd.read_csv('numerai_training_data.csv', header=0)
prediction_data = pd.read_csv('numerai_tournament_data.csv', header=0)

# Transform the loaded CSV data into numpy arrays
features = [f for f in list(training_data) if "feature" in f]
X = training_data[features]
Y = training_data["target"]
tournament = prediction_data[features]
ids = prediction_data["id"]

pca = decomposition.PCA()
pca.fit(X)
components = range(pca.n_components_)
plt.bar(components, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()