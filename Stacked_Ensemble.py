import logging
from math import sqrt
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


def grid_search_report(model):
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


def gen_param_dict(model, x_test, y_test):
    cv_results = model.cv_results_["mean_test_score"]
    cv_results = {"mean": round(np.mean(cv_results), 5),
                  "std": round(np.std(cv_results), 5),
                  "min": round(np.min(cv_results), 5),
                  "max": round(np.max(cv_results), 5)}
    params = {"best_parameters": model.best_params_,
              "best_validation_score": model.best_score_,
              "cv_score_stats": cv_results,
              "holdout_accuracy": model.score(x_test, y_test)}
    return params

logging.basicConfig(filename="numerai_logger.log",
                    level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%d.%m.%Y %H:%M:%S")

# Data loading
logging.info("Loading Data")
training_data = pd.read_csv('numerai_training_data.csv', header=0)
prediction_data = pd.read_csv('numerai_tournament_data.csv', header=0)
training_data = training_data.sample(n=15)
prediction_data = prediction_data.sample(n=15)
features = [f for f in list(training_data) if "feature" in f]
X = training_data[features]
Y = training_data["target"]
tournament = prediction_data[features]
ids = prediction_data["id"]
logging.info("Data loading and transformation finished")

models = {"ExtraTrees": [ExtraTreesClassifier(n_jobs=-1),
                         {"max_features": range(int(sqrt(X.shape[1])), int(X.shape[1] * 0.5)),
                          "n_estimators": [val for sublist in [[10, 25], list(range(50, 501, 50))] for val in sublist]}],
          "Nystroem-LinearSVM": [Pipeline([("Nystroem_feature_map", Nystroem()),
                                           ("svm", LinearSVC())]),
                                 {"svm__C": [0.1, 10, 100]}],
          "KNearestNeighbors": [KNeighborsClassifier(algorithm="kd_tree", n_jobs=-1),
                                {"n_neighbors": range(5, 250, 5)}]}

skf = StratifiedKFold(n_splits=5)
no_of_models = len(models.keys())
predictions_train = np.zeros([X.shape[0], no_of_models])
predictions_submission = np.zeros([prediction_data.shape[0], no_of_models])
for j, (train, test) in enumerate(skf.split(X=X, y=Y)):
    logging.info("Train / Test Split, Split {j} of {split}".format(j=j+1,
                                                                   split=skf.get_n_splits()))
    x_train, y_train = X.iloc[train], Y.iloc[train]
    x_test, y_test = X.iloc[test], Y.iloc[test]
    for i, model in enumerate(models.keys()):
        logging.info("Begin Training: {model}, Model {i} of {len}".format(model=model,
                                                                          i=i+1,
                                                                          len=no_of_models))
        cur_model = GridSearchCV(models[model][0], param_grid=models[model][1], cv=5)
        cur_model.fit(X=x_train, y=y_train)
        params = gen_param_dict(cur_model, x_test=x_test, y_test=y_test)
        grid_search_report(model, params=params)
        predictions_train[train, i] = cur_model.predict_proba(X=x_test)
        predictions_submission[:, i] = cur_model.predict_proba(X=tournament)

# stacked classifier
logging.info("Base model training complete. Starting meta classifier training.")
sfk = StratifiedKFold(n_splits=2)
ensemble_tree = RandomForestClassifier(n_jobs=-1)
meta_classifier = GridSearchCV(ensemble_tree, param_grid=models["ExtraTrees"])
params = {"best_parameters": meta_classifier.best_params_,
          "best_validation_score": meta_classifier.best_score_,
          "mean_cv_score": meta_classifier.cv_results_["mean_test_score"],
          "holdout_accuracy": meta_classifier.score(x_test, y_test)}
grid_search_report(meta_classifier, params=params)

logging.info("Meta classifier training complete. Predicting tournament propabilities.")
y_prediction = ensemble_tree.predict_proba(predictions_submission)
logging.info("Writing predictions to predictions.csv")
final = pd.DataFrame(ids, y_prediction)
final.to_csv("predictions.csv", index=False)
