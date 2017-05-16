from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV


np.random.seed(0)

print("Loading data...")
# Load the data from the CSV files
training_data = pd.read_csv('numerai_training_data.csv', header=0)
prediction_data = pd.read_csv('numerai_tournament_data.csv', header=0)

# Transform the loaded CSV data into numpy arrays
features = [f for f in list(training_data) if "feature" in f]
X = training_data[features]
Y = training_data["target"]
ind = train_test_split(X.index)
x_train, y_train = X.iloc[ind[0]], Y.iloc[ind[0]]
x_test, y_test = X.iloc[ind[1]], Y.iloc[ind[1]]
tournament = prediction_data[features]
ids = prediction_data["id"]

print("Training...")
# Random Forest
param_grid = {"max_features": range(int(sqrt(X.shape[1])), int(X.shape[1]*0.5)),
              "n_estimators": [val for sublist in [[10, 25], list(range(50, 501, 50))] for val in sublist]}
model = GridSearchCV(RandomForestClassifier(n_jobs=-1), param_grid=param_grid, cv=10)
model = model.fit(X=x_train, y=y_train)
print("Best parameters on training set: {0}".format(model.best_params_))
print("Best cross-validated accuracy: {0}".format(model.best_score_))
print("Accuracy on holdout test set: {0}".format(model.score(x_test, y_test)))

print("Predicting...")
# Your trained model is now used to make predictions on the numerai_tournament_data
# The model returns two columns: [probability of 0, probability of 1]
# We are just interested in the probability that the target is 1.
y_prediction = model.predict_proba(tournament)
results = y_prediction[:, 1]
results_df = pd.DataFrame(data={'probability': results})
joined = pd.DataFrame(ids).join(results_df)

print("Writing predictions to predictions.csv")
# Save the predictions out to a CSV file
joined.to_csv("predictions.csv", index=False)
# Now you can upload these predictions on numer.ai
