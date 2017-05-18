import logging
from math import sqrt
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

logging.basicConfig(filename="numerai_logger.log",
                    level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%d.%m.%Y %H:%M:%S")

# models = {"names": ["ExtraTrees",
#                     "Nystroem LinearSVM",
#                     "K Nearest Neighbors"],
#           "classes": [ExtraTreesClassifier(n_jobs=-1),
#                       Pipeline([("Nystroem_feature_map", Nystroem()),
#                                 ("svm", LinearSVC())]),
#                       KNeighborsClassifier()],
#           "params": [{"max_features": range(int(sqrt(X.shape[1])), int(X.shape[1] * 0.5)),
#                       "n_estimators": [val for sublist in [[10, 25], list(range(50, 501, 50))] for val in sublist]},
#                      {"svm__C": [0.1, 10, 100]}]}

models = {"ExtraTrees": [ExtraTreesClassifier(n_jobs=-1),
                         {"max_features": range(int(sqrt(X.shape[1])), int(X.shape[1] * 0.5)),
                          "n_estimators": [val for sublist in [[10, 25], list(range(50, 501, 50))] for val in sublist]}],
          "Nystroem-LinearSVM": [Pipeline([("Nystroem_feature_map", Nystroem()),
                                           ("svm", LinearSVC())]),
                                 {"svm__C": [0.1, 10, 100]}],
          "KNearestNeighbors": [KNeighborsClassifier(algorithm="kd_tree", n_jobs=-1),
                                {"n_neighbors": range(5, 250, 5)}]}

# TODO: Add mean CV score
def grid_search_report(model, params):
    logging.info("Training End: Model {}".format(model))
    logging.info("""{model} Results:\n
                    Best parameters on training set: \n
                    {best_params}\n
                    Best cross-validated accuracy:\n
                    {cv_accuracy}\n
                    Mean cross-validated accuracy:\n
                    {mean_cv_accuracy}
                    Accuracy on holdout test set:\n
                    {holdout_accuracy}
                 """.format(model=model,
                            best_params=params["best_parameters"],
                            cv_accuracy=params["best_validation_score"],
                            mean_cv_accuracy=params["mean_cv_score"],
                            holdout_accuracy=params["holdout_accuracy"])
                 )
# TODO: Check out cv_results dict returned from GridSearchCV
# TODO: Combine Predictions
for model in models.keys():
    logging.info("Begin Training: {model}".format(model=model))
    cur_model = GridSearchCV(models[model][0], param_grid=models[model][1], cv=10)
    cur_model.fit(X=X, y=Y)
    params = {"best_parameters": cur_model.best_params_,
              "best_validation_score": cur_model.best_score_,
              "mean_cv_score": cur_model.cv_results_["mean_training_score"],
              "holdout_accuracy": cur_model.score(x_test, y_test)}
    grid_search_report(model, params=params)


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

# Extra Trees
logging.info("Training Start: Model ExtraTrees")

model = GridSearchCV(ExtraTreesClassifier(n_jobs=-1), param_grid=param_grid, cv=10)
model = extra_tree.fit(X=x_train, y=y_train)
extra_tree = model
logging.info("Training End: Model ExtraTrees")
logging.info("Extreme Random Tree Results:")
logging.info("Best parameters on training set: {0}".format(extra_tree.best_params_))
logging.info("Best cross-validated accuracy: {0}".format(extra_tree.best_score_))
logging.info("Accuracy on holdout test set: {0}".format(extra_tree.score(x_test, y_test)))

# Linear SVC
logging.info("Training Start: Model Linear SVC")
param_grid = {}
linear_svm = GridSearchCV(LinearSVC())

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
