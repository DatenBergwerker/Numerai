import logging
from math import sqrt
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


def grid_search_report(model, params):
    logging.info(f"Training End: Model {model}")
    logging.info(f"""
                    {model} Results:\n
                    Best parameters on training set: \n
                    {params["best_parameters"]}\n
                    Best cross-validated accuracy:\n
                    {params["best_validation_score"]}\n
                    Cross-validation stats:\n
                    {params["cv_score_stats"]}\n
                    Accuracy on holdout test set:\n
                    {params["holdout_accuracy"]}
                 """
                 )


def gen_param_dict(model, score):
    cv_results = model.cv_results_["mean_test_score"]
    cv_results = {"mean": round(np.mean(cv_results), 5),
                  "std": round(np.std(cv_results), 5),
                  "min": round(np.min(cv_results), 5),
                  "max": round(np.max(cv_results), 5)}
    params = {"best_parameters": model.best_params_,
              "best_validation_score": model.best_score_,
              "cv_score_stats": cv_results,
              "holdout_accuracy": score}
    return params


logging.basicConfig(filename="numerai_logger.log",
                    level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt="%d.%m.%Y %H:%M:%S")

# Data loading
logging.info("Loading Data")
training_data = pd.read_csv('numerai_training_data.csv', header=0)
prediction_data = pd.read_csv('numerai_tournament_data.csv', header=0)
training_data = training_data.sample(n=1000)
prediction_data = prediction_data.sample(n=1000)
features = [f for f in list(training_data) if "feature" in f]
X = training_data[features]
Y = training_data["target"]
tournament = prediction_data[features]
ids = prediction_data["id"]
logging.info("Data loading and transformation finished.")

models = {"ExtraTrees": [ExtraTreesClassifier(n_jobs=-1),
                         {"max_features": range(int(sqrt(X.shape[1])), int(X.shape[1] * 0.5)),
                          "n_estimators": [val for sublist in [[10, 25], list(range(50, 501, 50))] for val in sublist]}],
          "Nystroem-LogitRegression": [Pipeline([("Nystroem_feature_map", Nystroem()),
                                                 ("logisticRegression", LogisticRegression(solver="sag", n_jobs=-1))]),
                                       {"logisticRegression__C": [0.1, 1, 10]}],
          "KNearestNeighbors": [KNeighborsClassifier(algorithm="kd_tree", n_jobs=-1),
                                {"n_neighbors": range(5, 250, 5)}]}
meta_learner = {"n_estimators": [val for sublist in [[10, 25], list(range(50, 501, 50))] for val in sublist]}
skf = StratifiedKFold(n_splits=3)
no_of_models = len(models.keys())
fitted_models = {key: [0, None] for key in models.keys()}

predictions_train = np.zeros([X.shape[0], no_of_models])
predictions_submission = np.zeros([prediction_data.shape[0], no_of_models])

#
for j, (train, test) in enumerate(skf.split(X=X, y=Y)):
    logging.info(f"Train / Test Split, Split {j + 1} of {skf.get_n_splits()}")
    x_train, y_train = X.iloc[train], Y.iloc[train]
    x_test, y_test = X.iloc[test], Y.iloc[test]
    for i, model in enumerate(models.keys()):
        logging.info(f"Begin Training: {model}, Model {i + 1} of {no_of_models}")
        cur_model = GridSearchCV(models[model][0], param_grid=models[model][1], cv=5)
        cur_model.fit(X=x_train, y=y_train)
        score = cur_model.score(X=x_test, y=y_test)
        params = gen_param_dict(cur_model, score=score)
        grid_search_report(model, params=params)
        predictions_train[test, i] = cur_model.predict_proba(X=x_test)[:, 1]

        # Keep best models for predicting first level test set values
        if score > fitted_models[model][0]:
            fitted_models[model] = [score, cur_model]

logging.info("Base model training complete. Starting base model prediction.")
# prediction run
for i, model in enumerate(fitted_models.keys()):
    cur_model = fitted_models[model][1]
    predictions_submission[:, i] = cur_model.predict_proba(X=tournament)[:, 1]
logging.info("Base model prediction complete. Starting Meta Learner Training.")

# stacked classifier
meta_classifier = GridSearchCV(RandomForestClassifier(n_jobs=-1), param_grid=meta_learner, cv=5)
meta_classifier.fit(X=predictions_train, y=Y)
score = meta_classifier.best_score_
params = gen_param_dict(meta_classifier, score=score)
grid_search_report("Meta Classifier", params=params)
logging.info("Meta classifier training complete. Predicting tournament probabilities.")
y_prediction = meta_classifier.predict_proba(predictions_submission)[:, 1]
logging.info("Writing predictions to predictions.csv")
final = pd.DataFrame({"id": ids, "pred": y_prediction})
final.to_csv("Numerai_predictions.csv", index=False)
