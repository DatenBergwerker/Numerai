import logging
from math import sqrt
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC

logging.basicConfig(filename="numerai_logger.log",
                    level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%d.%m.%Y %H:%M:%S")
logging.info("Loading Data")
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
logging.info("Data loading and transformation finished")

logging.info("Beginning training")

# Random Forest
logging.info("Training Start: Model ExtraTrees")
param_grid = {"max_features": range(int(sqrt(X.shape[1])), int(X.shape[1]*0.5)),
              "n_estimators": [val for sublist in [[10, 25], list(range(50, 501, 50))] for val in sublist]}
extra_tree = GridSearchCV(ExtraTreesClassifier(n_jobs=-1), param_grid=param_grid, cv=10)
extra_tree = extra_tree.fit(X=x_train, y=y_train)
logging.info("Training End: Model ExtraTrees")
logging.info("Extreme Random Tree Results:")
logging.info("Best parameters on training set: {0}".format(extra_tree.best_params_))
logging.info("Best cross-validated accuracy: {0}".format(extra_tree.best_score_))
logging.info("Accuracy on holdout test set: {0}".format(extra_tree.score(x_test, y_test)))

# Linear SVC
param_grid = {}



print("Predicting...")
# Your trained model is now used to make predictions on the numerai_tournament_data
# The model returns two columns: [probability of 0, probability of 1]
# We are just interested in the probability that the target is 1.
y_prediction = extra_tree.predict_proba(tournament)
results = y_prediction[:, 1]
results_df = pd.DataFrame(data={'probability': results})
joined = pd.DataFrame(ids).join(results_df)

print("Writing predictions to predictions.csv")
# Save the predictions out to a CSV file
joined.to_csv("predictions.csv", index=False)
# Now you can upload these predictions on numer.ai
