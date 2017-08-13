import logging
import numpy as np


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