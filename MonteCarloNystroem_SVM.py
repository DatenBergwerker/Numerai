import logging
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC

logging.basicConfig(filename="MC_Ny_SVM.log",
                    level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt="%d.%m.%Y %H:%M:%S")

# Data loading
logging.info("Loading Data")
training_data = pd.read_csv('numerai_training_data.csv', header=0)
prediction_data = pd.read_csv('numerai_tournament_data.csv', header=0)
features = [f for f in list(training_data) if "feature" in f]
X = training_data[features]
Y = training_data["target"]
tournament = prediction_data[features]
ids = prediction_data["id"]

# Preliminaries
iterations = 100
model_name = "Nystroem-LogitRegression"
model = {model_name: Pipeline([("Nystroem_feature_map", Nystroem()),
                               ("logisticRegression", LogisticRegression(solver="sag"))]),
         "param_grid": {"logisticRegression__C": np.linspace(start=0.1, stop=10,
                                                             num=100, endpoint=True)}}
# model_name = "Nystroem-LinearSupportVectorMachine"
# model = {model_name: Pipeline([("Nystroem_feature_map", Nystroem()),
#                                 ("SVC", SVC(kernel="linear"))]),
#          "param_grid": {"SVC__C": np.linspace(start=0.1, stop=100,
#                                               num=1000, endpoint=True)}}


results = np.zeros(shape=(tournament.shape[0], iterations))
start_t = datetime.now()

# Monte Carlo Leave One Out Cross Validation
logging.info(f"Training Start Model: {model_name} {iterations} Iterations.")
for i in range(iterations):
    X_train, _, Y_train, _ = train_test_split(X, Y, train_size=0.7, test_size=0.3)
    cur_model = GridSearchCV(model[model_name], cv=5, param_grid=model["param_grid"], n_jobs=-1)
    cur_model.fit(X=X_train, y=Y_train)
    pred = cur_model.predict_proba(X=tournament)
    results[:, i] = pred[:, 1]
    report = (np.mean(cur_model.cv_results_["mean_test_score"]), np.std(cur_model.cv_results_["mean_test_score"]))
    logging.info(f"Run {i} finished. Mean CV Test score: {report[0]} CV test score SD: {report[1]}")

logging.info(f"Training finished. Elapsed Time: {(datetime.now() - start_t).total_seconds()/60} Minutes.")
final = pd.DataFrame({"id": ids, "probability": results.mean(axis=0)})
final.to_csv("MC_Ny_SVM_predictions", index=False)
